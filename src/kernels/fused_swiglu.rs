// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! Fused SwiGLU activation CubeCL kernels.
//!
//! This module provides GPU-accelerated implementations of:
//! - SwiGLU activation (Swish-Gated Linear Unit)
//! - Fused FFN block with SwiGLU (optional advanced optimization)
//!
//! ## What is SwiGLU?
//!
//! SwiGLU is a gated activation function used in modern LLMs like LLaMA, PaLM:
//! ```text
//! SwiGLU(gate, up) = SiLU(gate) * up
//!                 = (gate * sigmoid(gate)) * up
//! ```
//!
//! In transformer FFN blocks:
//! ```text
//! hidden = input @ W1.T  (gate projection)
//! up = input @ W3.T      (up projection)
//! activated = SwiGLU(hidden, up)
//! output = activated @ W2.T  (down projection)
//! ```
//!
//! ## Why Fuse?
//!
//! The SwiGLU computation is memory-bound. Fusing the activation:
//! - Reduces memory bandwidth by ~50% (no intermediate writes)
//! - Enables better instruction-level parallelism
//! - Reduces kernel launch overhead

use crate::error::{Result as UnslothResult, UnslothError};
use candle_core::Tensor;

#[cfg(feature = "cuda")]
use cubecl::prelude::*;
#[cfg(feature = "cuda")]
use cubecl_cuda::CudaRuntime;

/// Maximum block size for kernel launches
const _MAX_BLOCK_SIZE: u32 = 1024;

// ============================================================================
// CubeCL Kernel Definitions
// ============================================================================

/// SiLU (Swish) activation: x * sigmoid(x)
#[cfg(feature = "cuda")]
#[cube]
fn silu<F: Float + CubeElement>(x: F) -> F {
    let sigmoid = F::cast_from(1.0f32) / (F::cast_from(1.0f32) + F::exp(-x));
    x * sigmoid
}

/// Fused SwiGLU activation kernel.
///
/// Computes: output = SiLU(gate) * up = (gate * sigmoid(gate)) * up
///
/// This is a simple element-wise kernel that processes gate and up tensors
/// in parallel, producing the SwiGLU activation without intermediate memory writes.
///
/// Grid: (num_elements / block_size, 1, 1)
/// Block: (block_size, 1, 1)
#[cfg(feature = "cuda")]
#[cube(launch)]
fn swiglu_kernel<F: Float + CubeElement>(
    gate: &Array<F>,       // Gate projection output [*, intermediate_dim]
    up: &Array<F>,         // Up projection output [*, intermediate_dim]
    output: &mut Array<F>, // SwiGLU output [*, intermediate_dim]
    num_elements: u32,
) {
    let idx = ABSOLUTE_POS;

    if idx >= (num_elements as usize) {
        terminate!();
    }

    let gate_val = gate[idx];
    let up_val = up[idx];

    // SiLU(gate) = gate * sigmoid(gate)
    let sigmoid_gate = F::cast_from(1.0f32) / (F::cast_from(1.0f32) + F::exp(-gate_val));
    let silu_gate = gate_val * sigmoid_gate;

    // SwiGLU = SiLU(gate) * up
    output[idx] = silu_gate * up_val;
}

/// Backward pass for SwiGLU.
///
/// Given grad_output (dL/d_output), computes:
/// - grad_gate = grad_output * up * silu'(gate)
/// - grad_up = grad_output * silu(gate)
///
/// Where silu'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
///                = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
#[cfg(feature = "cuda")]
#[cube(launch)]
fn swiglu_backward_kernel<F: Float + CubeElement>(
    grad_output: &Array<F>, // Gradient from downstream
    gate: &Array<F>,        // Original gate values
    up: &Array<F>,          // Original up values
    grad_gate: &mut Array<F>,
    grad_up: &mut Array<F>,
    num_elements: u32,
) {
    let idx = ABSOLUTE_POS;

    if idx >= (num_elements as usize) {
        terminate!();
    }

    let g_out = grad_output[idx];
    let gate_val = gate[idx];
    let up_val = up[idx];

    // Compute sigmoid(gate)
    let sigmoid_gate = F::cast_from(1.0f32) / (F::cast_from(1.0f32) + F::exp(-gate_val));

    // Compute silu(gate) = gate * sigmoid(gate)
    let silu_gate = gate_val * sigmoid_gate;

    // grad_up = grad_output * silu(gate)
    grad_up[idx] = g_out * silu_gate;

    // silu'(gate) = sigmoid(gate) * (1 + gate * (1 - sigmoid(gate)))
    let silu_grad =
        sigmoid_gate * (F::cast_from(1.0f32) + gate_val * (F::cast_from(1.0f32) - sigmoid_gate));

    // grad_gate = grad_output * up * silu'(gate)
    grad_gate[idx] = g_out * up_val * silu_grad;
}

/// Vectorized SwiGLU kernel for better memory throughput.
///
/// Processes 4 elements at a time using vector loads/stores.
/// This improves memory coalescing and bandwidth utilization.
#[cfg(feature = "cuda")]
#[cube(launch)]
fn swiglu_vectorized_kernel<F: Float + CubeElement>(
    gate: &Array<Line<F>>,       // Gate as 4-element vectors
    up: &Array<Line<F>>,         // Up as 4-element vectors
    output: &mut Array<Line<F>>, // Output as 4-element vectors
    #[comptime] num_vectors: u32,
) {
    let idx = ABSOLUTE_POS;

    if idx >= (num_vectors as usize) {
        terminate!();
    }

    let gate_vec = gate[idx];
    let up_vec = up[idx];

    // Process each element of the vector
    // SwiGLU element-wise using vectorized = 4 elements
    #[unroll]
    for i in 0..4 {
        let g = gate_vec[i];
        let u = up_vec[i];
        let sigmoid_g = F::cast_from(1.0f32) / (F::cast_from(1.0f32) + F::exp(-g));
        output[idx][i] = g * sigmoid_g * u;
    }
}

/// Fused FFN with SwiGLU kernel (advanced optimization).
///
/// Combines the entire FFN computation:
/// 1. gate = x @ W1.T
/// 2. up = x @ W3.T
/// 3. hidden = SwiGLU(gate, up)
/// 4. output = hidden @ W2.T
///
/// This is a tiled matrix multiplication kernel with fused activation.
/// Uses shared memory for tile caching.
///
/// Note: This kernel is more complex and only beneficial for specific
/// shapes. For most cases, separate GEMM + SwiGLU is faster due to
/// highly optimized cuBLAS/cuDNN GEMM implementations.
#[cfg(feature = "cuda")]
#[cube(launch)]
fn fused_ffn_swiglu_tiled_kernel<F: Float + CubeElement>(
    input: &Array<F>,      // [M, K] where M = batch*seq, K = hidden_dim
    w1: &Array<F>,         // [K, N] gate projection
    w3: &Array<F>,         // [K, N] up projection
    _w2: &Array<F>,        // [N, K] down projection (unused in this phase)
    output: &mut Array<F>, // [M, K]
    m_val: u32,            // batch * seq_len
    k_val: u32,            // hidden_dim
    n_val: u32,            // intermediate_dim
    tile_size: u32,        // Tile size for shared memory
) {
    // This is a complex kernel - for production use, we typically
    // use separate optimized GEMM calls followed by element-wise SwiGLU.
    // This implementation is provided for reference.

    let row = CUBE_POS_X * tile_size + UNIT_POS_Y;
    let col = CUBE_POS_Y * tile_size + UNIT_POS_X;
    let tid_x = UNIT_POS_X as usize;
    let tid_y = UNIT_POS_Y as usize;

    // Shared memory for tiles (fixed size for CubeCL 0.9)
    let tile_size_usize = tile_size as usize;
    let _tile_size_sq = tile_size_usize * tile_size_usize;
    let mut input_tile = SharedMemory::<F>::new(1024usize);
    let mut w1_tile = SharedMemory::<F>::new(1024usize);
    let mut w3_tile = SharedMemory::<F>::new(1024usize);
    let mut swiglu_tile = SharedMemory::<F>::new(1024usize);

    // Bounds check
    if row >= m_val || col >= k_val {
        terminate!();
    }

    // Step 1: Compute gate = input @ W1 and up = input @ W3 tile by tile
    let mut gate_acc = F::cast_from(0.0f32);
    let mut up_acc = F::cast_from(0.0f32);

    let num_tiles = (k_val + tile_size - 1) / tile_size;
    for t in 0..num_tiles {
        let tile_start = t * tile_size;

        // Load input tile
        let input_row = row;
        let input_col = (tile_start as usize) + tid_x;
        if input_row < m_val && (input_col as u32) < k_val {
            let input_idx = (input_row as usize * (k_val as usize)) + input_col;
            input_tile[(tid_y * tile_size_usize) + tid_x] = input[input_idx];
        } else {
            input_tile[(tid_y * tile_size_usize) + tid_x] = F::cast_from(0.0f32);
        }

        // Load W1 tile
        let w1_row = (tile_start as usize) + tid_y;
        let w1_col = col as usize; // Actually N dimension
        if (w1_row as u32) < k_val && (w1_col as u32) < n_val {
            let w1_idx = ((w1_row as u32) as usize * (n_val as usize)) + w1_col;
            w1_tile[(tid_y * tile_size_usize) + tid_x] = w1[w1_idx];
        } else {
            w1_tile[(tid_y * tile_size_usize) + tid_x] = F::cast_from(0.0f32);
        }

        // Load W3 tile
        if (w1_row as u32) < k_val && (w1_col as u32) < n_val {
            let w3_idx = ((w1_row as u32) as usize * (n_val as usize)) + w1_col;
            w3_tile[(tid_y * tile_size_usize) + tid_x] = w3[w3_idx];
        } else {
            w3_tile[(tid_y * tile_size_usize) + tid_x] = F::cast_from(0.0f32);
        }

        sync_cube();

        // Compute partial dot products
        for k in 0..tile_size {
            let k_usize = k as usize;
            let input_val = input_tile[(tid_y * tile_size_usize) + k_usize];
            let w1_val = w1_tile[(k_usize * tile_size_usize) + tid_x];
            let w3_val = w3_tile[(k_usize * tile_size_usize) + tid_x];
            gate_acc = gate_acc + input_val * w1_val;
            up_acc = up_acc + input_val * w3_val;
        }

        sync_cube();
    }

    // Step 2: Apply SwiGLU
    let sigmoid_gate = F::cast_from(1.0f32) / (F::cast_from(1.0f32) + F::exp(-gate_acc));
    let swiglu_val = gate_acc * sigmoid_gate * up_acc;

    // Store SwiGLU result in shared memory for next matmul
    swiglu_tile[(tid_y * tile_size_usize) + tid_x] = swiglu_val;
    sync_cube();

    // Step 3: Compute output = swiglu_result @ W2
    // (This would require another tiled matmul - simplified here)
    // For production, use separate GEMM call

    // Write intermediate result (in production, would continue with W2 matmul)
    if row < m_val && col < n_val {
        // This is the intermediate activation
        // In a full implementation, we'd do another matmul with W2
        let output_idx = (row as usize * (n_val as usize)) + (col as usize);
        output[output_idx] = swiglu_val;
    }
}

// ============================================================================
// Public API Functions
// ============================================================================

/// Apply SwiGLU activation using CubeCL GPU kernel.
///
/// Computes: output = SiLU(gate) * up = (gate * sigmoid(gate)) * up
///
/// # Arguments
/// * `gate` - Gate projection output tensor
/// * `up` - Up projection output tensor (same shape as gate)
///
/// # Returns
/// SwiGLU activation output with same shape as inputs
///
/// # Example
/// ```rust,ignore
/// let hidden = linear(&input, &w1)?;  // Gate projection
/// let up = linear(&input, &w3)?;      // Up projection
/// let activated = swiglu(&hidden, &up)?;
/// let output = linear(&activated, &w2)?;  // Down projection
/// ```
pub fn swiglu(gate: &Tensor, up: &Tensor) -> UnslothResult<Tensor> {
    // Validate shapes match
    if gate.dims() != up.dims() {
        return Err(UnslothError::InvalidConfig(format!(
            "Gate and up tensor shapes must match: {:?} vs {:?}",
            gate.dims(),
            up.dims()
        )));
    }

    #[cfg(feature = "cuda")]
    {
        if gate.device().is_cuda() {
            return launch_swiglu_kernel(gate, up);
        }
    }

    // CPU fallback
    swiglu_cpu(gate, up)
}

/// Apply SwiGLU backward pass using CubeCL GPU kernel.
///
/// # Arguments
/// * `grad_output` - Gradient from downstream layer
/// * `gate` - Original gate values (saved from forward pass)
/// * `up` - Original up values (saved from forward pass)
///
/// # Returns
/// Tuple of (grad_gate, grad_up) gradients
pub fn swiglu_backward(
    grad_output: &Tensor,
    gate: &Tensor,
    up: &Tensor,
) -> UnslothResult<(Tensor, Tensor)> {
    #[cfg(feature = "cuda")]
    {
        if grad_output.device().is_cuda() {
            return launch_swiglu_backward_kernel(grad_output, gate, up);
        }
    }

    // CPU fallback
    swiglu_backward_cpu(grad_output, gate, up)
}

/// Fused FFN block with SwiGLU activation.
///
/// Computes the complete feed-forward network:
/// ```text
/// hidden = input @ W1.T
/// up = input @ W3.T
/// activated = SwiGLU(hidden, up)
/// output = activated @ W2.T
/// ```
///
/// # Arguments
/// * `input` - Input tensor [batch, seq_len, hidden_dim]
/// * `w1` - Gate projection weights [intermediate_dim, hidden_dim]
/// * `w3` - Up projection weights [intermediate_dim, hidden_dim]
/// * `w2` - Down projection weights [hidden_dim, intermediate_dim]
///
/// # Returns
/// FFN output [batch, seq_len, hidden_dim]
///
/// # Note
/// For best performance, this uses separate optimized GEMM operations
/// followed by fused SwiGLU, rather than a fully fused kernel.
pub fn fused_ffn_swiglu(
    input: &Tensor,
    w1: &Tensor,
    w3: &Tensor,
    w2: &Tensor,
) -> UnslothResult<Tensor> {
    // Validate dimensions
    let input_shape = input.dims();
    let w1_shape = w1.dims();
    let w3_shape = w3.dims();
    let w2_shape = w2.dims();

    if w1_shape.len() != 2 || w3_shape.len() != 2 || w2_shape.len() != 2 {
        return Err(UnslothError::InvalidConfig(
            "Weight matrices must be 2D".to_string(),
        ));
    }

    let hidden_dim = input_shape[input_shape.len() - 1];
    let intermediate_dim = w1_shape[0];

    if w1_shape[1] != hidden_dim || w3_shape[1] != hidden_dim {
        return Err(UnslothError::InvalidConfig(format!(
            "W1/W3 input dim {} doesn't match hidden_dim {}",
            w1_shape[1], hidden_dim
        )));
    }

    if w2_shape[1] != intermediate_dim || w2_shape[0] != hidden_dim {
        return Err(UnslothError::InvalidConfig(format!(
            "W2 shape {:?} incompatible with intermediate_dim {} and hidden_dim {}",
            w2_shape, intermediate_dim, hidden_dim
        )));
    }

    // Use optimized path: separate GEMMs + fused SwiGLU
    // This is typically faster than a fully fused kernel because
    // cuBLAS/cuDNN GEMMs are highly optimized

    // Step 1: Gate projection
    let gate = input.broadcast_matmul(&w1.t()?)?;

    // Step 2: Up projection
    let up = input.broadcast_matmul(&w3.t()?)?;

    // Step 3: Fused SwiGLU activation
    let activated = swiglu(&gate, &up)?;

    // Step 4: Down projection
    let output = activated.broadcast_matmul(&w2.t()?)?;

    Ok(output)
}

// ============================================================================
// Kernel Launch Functions
// ============================================================================

#[cfg(feature = "cuda")]
fn launch_swiglu_kernel(gate: &Tensor, up: &Tensor) -> UnslothResult<Tensor> {
    use crate::kernels::cubecl::interop::{candle_to_cubecl_handle, cubecl_to_candle_tensor};

    let num_elements: usize = gate.dims().iter().product();

    // Convert to bytes
    let (gate_bytes, _, _) = candle_to_cubecl_handle(gate)?;
    let (up_bytes, _, _) = candle_to_cubecl_handle(up)?;

    // Get CUDA client
    let device = cubecl_cuda::CudaDevice::new(0);
    let client = CudaRuntime::client(&device);

    // Create handles
    let gate_handle = client.create(cubecl::bytes::Bytes::from_bytes_vec(gate_bytes));
    let up_handle = client.create(cubecl::bytes::Bytes::from_bytes_vec(up_bytes));
    let output_handle = client.empty(num_elements * std::mem::size_of::<f32>());

    // Launch configuration
    let block_size = 256u32;
    let num_blocks = (num_elements as u32 + block_size - 1) / block_size;
    let cube_count = CubeCount::Static(num_blocks, 1, 1);
    let cube_dim = CubeDim::new(&client, block_size as usize);

    // SAFETY: Handles are valid and properly sized for the kernel operation
    unsafe {
        swiglu_kernel::launch::<f32, CudaRuntime>(
            &client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts::<f32>(&gate_handle, num_elements, 1),
            ArrayArg::from_raw_parts::<f32>(&up_handle, num_elements, 1),
            ArrayArg::from_raw_parts::<f32>(&output_handle, num_elements, 1),
            ScalarArg::new(num_elements as u32),
        )
        .map_err(|e| UnslothError::Kernel(format!("swiglu_kernel launch failed: {e}")))?;
    }

    let output_bytes = client.read_one(output_handle);
    cubecl_to_candle_tensor(&output_bytes, gate.dims(), gate.device())
}

#[cfg(feature = "cuda")]
fn launch_swiglu_backward_kernel(
    grad_output: &Tensor,
    gate: &Tensor,
    up: &Tensor,
) -> UnslothResult<(Tensor, Tensor)> {
    use crate::kernels::cubecl::interop::{candle_to_cubecl_handle, cubecl_to_candle_tensor};

    let num_elements: usize = gate.dims().iter().product();

    // Convert to bytes
    let (grad_bytes, _, _) = candle_to_cubecl_handle(grad_output)?;
    let (gate_bytes, _, _) = candle_to_cubecl_handle(gate)?;
    let (up_bytes, _, _) = candle_to_cubecl_handle(up)?;

    // Get CUDA client
    let device = cubecl_cuda::CudaDevice::new(0);
    let client = CudaRuntime::client(&device);

    // Create handles
    let grad_handle = client.create(cubecl::bytes::Bytes::from_bytes_vec(grad_bytes));
    let gate_handle = client.create(cubecl::bytes::Bytes::from_bytes_vec(gate_bytes));
    let up_handle = client.create(cubecl::bytes::Bytes::from_bytes_vec(up_bytes));
    let grad_gate_handle = client.empty(num_elements * std::mem::size_of::<f32>());
    let grad_up_handle = client.empty(num_elements * std::mem::size_of::<f32>());

    // Launch configuration
    let block_size = 256u32;
    let num_blocks = (num_elements as u32 + block_size - 1) / block_size;
    let cube_count = CubeCount::Static(num_blocks, 1, 1);
    let cube_dim = CubeDim::new(&client, block_size as usize);

    // SAFETY: Handles are valid and properly sized for the kernel operation
    unsafe {
        swiglu_backward_kernel::launch::<f32, CudaRuntime>(
            &client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts::<f32>(&grad_handle, num_elements, 1),
            ArrayArg::from_raw_parts::<f32>(&gate_handle, num_elements, 1),
            ArrayArg::from_raw_parts::<f32>(&up_handle, num_elements, 1),
            ArrayArg::from_raw_parts::<f32>(&grad_gate_handle, num_elements, 1),
            ArrayArg::from_raw_parts::<f32>(&grad_up_handle, num_elements, 1),
            ScalarArg::new(num_elements as u32),
        )
        .map_err(|e| UnslothError::Kernel(format!("swiglu_backward_kernel launch failed: {e}")))?;
    }

    let grad_gate_bytes = client.read_one(grad_gate_handle);
    let grad_up_bytes = client.read_one(grad_up_handle);

    let grad_gate = cubecl_to_candle_tensor(&grad_gate_bytes, gate.dims(), gate.device())?;
    let grad_up = cubecl_to_candle_tensor(&grad_up_bytes, up.dims(), up.device())?;

    Ok((grad_gate, grad_up))
}

// ============================================================================
// CPU Fallback Implementations
// ============================================================================

fn swiglu_cpu(gate: &Tensor, up: &Tensor) -> UnslothResult<Tensor> {
    // SiLU(gate) = gate * sigmoid(gate)
    let silu_gate = candle_nn::ops::silu(gate)?;

    // SwiGLU = SiLU(gate) * up
    let output = (silu_gate * up)?;

    Ok(output)
}

fn swiglu_backward_cpu(
    grad_output: &Tensor,
    gate: &Tensor,
    up: &Tensor,
) -> UnslothResult<(Tensor, Tensor)> {
    // sigmoid(gate)
    let sigmoid_gate = candle_nn::ops::sigmoid(gate)?;

    // silu(gate) = gate * sigmoid(gate)
    let silu_gate = (gate * &sigmoid_gate)?;

    // grad_up = grad_output * silu(gate)
    let grad_up = (grad_output * &silu_gate)?;

    // silu'(gate) = sigmoid(gate) * (1 + gate * (1 - sigmoid(gate)))
    let ones = Tensor::ones_like(&sigmoid_gate)?;
    let one_minus_sigmoid = (&ones - &sigmoid_gate)?;
    let gate_times_one_minus = (gate * &one_minus_sigmoid)?;
    let one_plus_term = (&ones + &gate_times_one_minus)?;
    let silu_grad = (&sigmoid_gate * &one_plus_term)?;

    // grad_gate = grad_output * up * silu'(gate)
    let grad_gate = (grad_output * up)?.broadcast_mul(&silu_grad)?;

    Ok((grad_gate, grad_up))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_swiglu_cpu() {
        let device = Device::Cpu;
        let shape = (2, 4, 256);

        let gate = Tensor::randn(0.0f32, 1.0, shape, &device).unwrap();
        let up = Tensor::randn(0.0f32, 1.0, shape, &device).unwrap();

        let output = swiglu(&gate, &up).unwrap();
        assert_eq!(output.dims(), gate.dims());

        // Verify no NaN/Inf
        let vals: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();
        for v in vals {
            assert!(!v.is_nan() && !v.is_infinite());
        }
    }

    #[test]
    fn test_swiglu_matches_reference() {
        let device = Device::Cpu;
        let shape = (2, 4, 64);

        let gate = Tensor::randn(0.0f32, 1.0, shape, &device).unwrap();
        let up = Tensor::randn(0.0f32, 1.0, shape, &device).unwrap();

        // Our implementation
        let output = swiglu(&gate, &up).unwrap();

        // Reference: manual computation
        let silu_gate = candle_nn::ops::silu(&gate).unwrap();
        let reference = (&silu_gate * &up).unwrap();

        // Compare
        let diff = (&output - &reference).unwrap().abs().unwrap();
        let max_diff = diff.max_all().unwrap().to_scalar::<f32>().unwrap();

        assert!(max_diff < 1e-5, "Max diff {} exceeds tolerance", max_diff);
    }

    #[test]
    fn test_fused_ffn_swiglu_shape() {
        let device = Device::Cpu;
        let batch = 2;
        let seq_len = 4;
        let hidden_dim = 64;
        let intermediate_dim = 128;

        let input = Tensor::randn(0.0f32, 1.0, (batch, seq_len, hidden_dim), &device).unwrap();
        let w1 = Tensor::randn(0.0f32, 0.1, (intermediate_dim, hidden_dim), &device).unwrap();
        let w3 = Tensor::randn(0.0f32, 0.1, (intermediate_dim, hidden_dim), &device).unwrap();
        let w2 = Tensor::randn(0.0f32, 0.1, (hidden_dim, intermediate_dim), &device).unwrap();

        let output = fused_ffn_swiglu(&input, &w1, &w3, &w2).unwrap();

        assert_eq!(output.dims(), &[batch, seq_len, hidden_dim]);
    }

    #[test]
    fn test_swiglu_backward_cpu() {
        let device = Device::Cpu;
        let shape = (2, 4, 64);

        let gate = Tensor::randn(0.0f32, 1.0, shape, &device).unwrap();
        let up = Tensor::randn(0.0f32, 1.0, shape, &device).unwrap();
        let grad_output = Tensor::randn(0.0f32, 1.0, shape, &device).unwrap();

        let (grad_gate, grad_up) = swiglu_backward(&grad_output, &gate, &up).unwrap();

        assert_eq!(grad_gate.dims(), gate.dims());
        assert_eq!(grad_up.dims(), up.dims());

        // Verify no NaN/Inf
        for tensor in [&grad_gate, &grad_up] {
            let vals: Vec<f32> = tensor.flatten_all().unwrap().to_vec1().unwrap();
            for v in vals {
                assert!(!v.is_nan() && !v.is_infinite());
            }
        }
    }
}
