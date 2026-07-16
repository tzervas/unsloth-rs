// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! Fused RMSNorm + Rotary Position Embedding (RoPE) CubeCL kernels.
//!
//! This module provides GPU-accelerated implementations of:
//! - Standalone RMSNorm for post-attention layers
//! - Fused RMSNorm + RoPE for pre-attention layers (huge optimization)
//!
//! ## Why Fuse RMSNorm and RoPE?
//!
//! In transformer inference, we often apply RMSNorm followed immediately by RoPE
//! to query and key tensors. Fusing these operations:
//! - Eliminates one global memory round-trip (saves ~2TB/s on modern GPUs)
//! - Reduces kernel launch overhead
//! - Improves cache utilization
//!
//! ## Algorithm
//!
//! ### RMSNorm
//! ```text
//! rms = sqrt(mean(x^2) + eps)
//! output = (x / rms) * weight
//! ```
//!
//! ### RoPE (Rotary Position Embedding)
//! ```text
//! For each pair (x_i, x_{i+d/2}):
//!     x'_i = x_i * cos(θ_i) - x_{i+d/2} * sin(θ_i)
//!     x'_{i+d/2} = x_{i+d/2} * cos(θ_i) + x_i * sin(θ_i)
//! ```
//!
//! ### Fused Operation
//! Applies RMSNorm first, then RoPE, in a single kernel pass.

use crate::error::{Result as UnslothResult, UnslothError};
use candle_core::Tensor;

#[cfg(feature = "cuda")]
use cubecl::prelude::*;
#[cfg(feature = "cuda")]
use cubecl_cuda::CudaRuntime;

/// Maximum block size for kernel launches (used when GPU dispatch is added)
#[allow(dead_code)]
const MAX_BLOCK_SIZE: u32 = 1024;

/// Warp size for NVIDIA GPUs (used when GPU dispatch is added)
#[allow(dead_code)]
const WARP_SIZE: u32 = 32;

// ============================================================================
// CubeCL Kernel Definitions
// ============================================================================

/// Standalone RMSNorm CubeCL kernel.
///
/// Each block processes one row (one token position).
/// Threads cooperatively compute sum of squares, then apply normalization.
///
/// Grid: (num_rows, 1, 1)
/// Block: (min(hidden_dim, MAX_BLOCK_SIZE), 1, 1)
#[cfg(feature = "cuda")]
#[cube(launch)]
fn rmsnorm_kernel<F: Float + CubeElement>(
    input: &Array<F>,      // [num_rows, hidden_dim]
    weight: &Array<F>,     // [hidden_dim]
    output: &mut Array<F>, // [num_rows, hidden_dim]
    hidden_dim: u32,
    eps: F,
    block_size: u32,
) {
    let row_idx = CUBE_POS_X;
    let tid = UNIT_POS_X;

    let base_idx = (row_idx as usize) * (hidden_dim as usize);
    let is_active = (tid as usize) < (hidden_dim as usize);

    // Shared memory for reduction
    let mut shared_sq = SharedMemory::<F>::new(1024usize);

    // Step 1: Compute sum of squares for this row
    let mut local_sum = F::cast_from(0.0f32);
    if is_active {
        // Handle hidden_dim > block_size with striding
        let mut i = tid as usize;
        while i < (hidden_dim as usize) {
            let val = input[base_idx + i];
            local_sum = local_sum + val * val;
            i = i + (block_size as usize);
        }
    }
    shared_sq[tid as usize] = local_sum;
    sync_cube();

    // Tree reduction for sum of squares
    let mut stride = (block_size / 2) as usize;
    while stride > 0 {
        if (tid as usize) < stride {
            let partner_idx = (tid as usize) + stride;
            if partner_idx < (block_size as usize) {
                shared_sq[tid as usize] = shared_sq[tid as usize] + shared_sq[partner_idx];
            }
        }
        sync_cube();
        stride = stride / 2;
    }

    // Compute inverse RMS and broadcast
    let sum_sq = shared_sq[0];
    let mean_sq = sum_sq / F::cast_from(hidden_dim as f32);
    let rms = F::sqrt(mean_sq + eps);
    let inv_rms = F::cast_from(1.0f32) / rms;

    // Store inv_rms in shared memory for all threads
    if tid as usize == 0 {
        shared_sq[0] = inv_rms;
    }
    sync_cube();
    let inv_rms_val = shared_sq[0];

    // Step 2: Apply normalization with striding
    if is_active {
        let mut i = tid as usize;
        while i < (hidden_dim as usize) {
            let val = input[base_idx + i];
            let w = weight[i];
            output[base_idx + i] = val * inv_rms_val * w;
            i = i + (block_size as usize);
        }
    }
}

/// Fused RMSNorm + RoPE CubeCL kernel.
///
/// Applies RMSNorm normalization followed by Rotary Position Embedding
/// in a single GPU kernel pass, avoiding intermediate memory writes.
///
/// Grid: (batch_size, seq_len, 1)
/// Block: (min(hidden_dim, MAX_BLOCK_SIZE), 1, 1)
///
/// The kernel handles the common transformer pattern where:
/// 1. Input goes through RMSNorm
/// 2. Normalized output is split into Q, K heads
/// 3. Each head gets RoPE applied based on sequence position
#[cfg(feature = "cuda")]
#[cube(launch)]
fn fused_rmsnorm_rope_kernel<F: Float + CubeElement>(
    input: &Array<F>,      // [batch, seq_len, hidden_dim]
    weight: &Array<F>,     // RMSNorm weight [hidden_dim]
    cos_cache: &Array<F>,  // Precomputed cos [max_seq, head_dim/2]
    sin_cache: &Array<F>,  // Precomputed sin [max_seq, head_dim/2]
    output: &mut Array<F>, // [batch, seq_len, hidden_dim]
    batch_size: u32,
    seq_len: u32,
    hidden_dim: u32,
    head_dim: u32,
    _num_heads: u32,
    eps: F,
    block_size: u32,
) {
    let batch_idx = CUBE_POS_X;
    let seq_idx = CUBE_POS_Y;
    let tid = UNIT_POS_X;

    // Bounds check
    if batch_idx >= batch_size || seq_idx >= seq_len {
        terminate!();
    }

    let base_idx =
        ((batch_idx as usize) * (seq_len as usize) + (seq_idx as usize)) * (hidden_dim as usize);
    let is_active = (tid as usize) < (hidden_dim as usize);
    let half_head = (head_dim / 2) as usize;

    // Shared memory for reduction and intermediate values
    let mut shared = SharedMemory::<F>::new(1024usize);

    // ========== Step 1: Compute RMS ==========
    let mut local_sum = F::cast_from(0.0f32);
    if is_active {
        let mut i = tid as usize;
        while i < (hidden_dim as usize) {
            let val = input[base_idx + i];
            local_sum = local_sum + val * val;
            i = i + (block_size as usize);
        }
    }
    shared[tid as usize] = local_sum;
    sync_cube();

    // Tree reduction
    let mut stride = (block_size / 2) as usize;
    while stride > 0 {
        if (tid as usize) < stride {
            let partner_idx = (tid as usize) + stride;
            if partner_idx < (block_size as usize) {
                shared[tid as usize] = shared[tid as usize] + shared[partner_idx];
            }
        }
        sync_cube();
        stride = stride / 2;
    }

    // Compute and broadcast inv_rms
    let sum_sq = shared[0];
    let mean_sq = sum_sq / F::cast_from(hidden_dim as f32);
    let rms = F::sqrt(mean_sq + eps);
    let inv_rms = F::cast_from(1.0f32) / rms;

    if tid as usize == 0 {
        shared[0] = inv_rms;
    }
    sync_cube();
    let inv_rms_val = shared[0];

    // ========== Step 2: Apply RMSNorm and RoPE together ==========
    // Process elements in pairs for RoPE
    if is_active {
        let mut i = tid as usize;
        while i < (hidden_dim as usize) {
            // First apply RMSNorm
            let input_val = input[base_idx + i];
            let normed = input_val * inv_rms_val * weight[i];

            // Determine head and position within head
            let _head_idx = i / (head_dim as usize);
            let pos_in_head = i % (head_dim as usize);

            // Apply RoPE based on position in head
            if pos_in_head < half_head {
                // First half: needs value from second half
                let pair_idx = i + half_head;

                // Get the pair value (also normalized)
                let pair_input = input[base_idx + pair_idx];
                let pair_normed = pair_input * inv_rms_val * weight[pair_idx];

                // Get cos/sin for this position
                let cache_idx = (seq_idx as usize) * half_head + pos_in_head;
                let cos_val = cos_cache[cache_idx];
                let sin_val = sin_cache[cache_idx];

                // x' = x * cos - y * sin
                output[base_idx + i] = normed * cos_val - pair_normed * sin_val;
            } else {
                // Second half: needs value from first half
                let pair_idx = i - half_head;

                // Get the pair value (also normalized)
                let pair_input = input[base_idx + pair_idx];
                let pair_normed = pair_input * inv_rms_val * weight[pair_idx];

                // Get cos/sin for this position
                let cache_idx = (seq_idx as usize) * half_head + (pos_in_head - half_head);
                let cos_val = cos_cache[cache_idx];
                let sin_val = sin_cache[cache_idx];

                // y' = x * sin + y * cos
                output[base_idx + i] = pair_normed * sin_val + normed * cos_val;
            }

            i = i + (block_size as usize);
        }
    }
}

/// Standalone RoPE kernel for when RMSNorm is not needed.
///
/// Applies rotary position embeddings to pre-normalized Q/K tensors.
#[cfg(feature = "cuda")]
#[cube(launch)]
fn rope_kernel<F: Float + CubeElement>(
    input: &Array<F>,      // [batch, num_heads, seq_len, head_dim]
    cos_cache: &Array<F>,  // [max_seq, head_dim/2]
    sin_cache: &Array<F>,  // [max_seq, head_dim/2]
    output: &mut Array<F>, // [batch, num_heads, seq_len, head_dim]
    batch_size: u32,
    num_heads: u32,
    seq_len: u32,
    head_dim: u32,
    block_size: u32,
) {
    // Grid: (batch * num_heads, seq_len, 1)
    let batch_head_idx = CUBE_POS_X;
    let seq_idx = CUBE_POS_Y;
    let tid = UNIT_POS_X;

    let total_batch_heads = batch_size * num_heads;
    if batch_head_idx >= total_batch_heads || seq_idx >= seq_len {
        terminate!();
    }

    let half_head = (head_dim / 2) as usize;
    let base_idx =
        ((batch_head_idx as usize) * (seq_len as usize) + (seq_idx as usize)) * (head_dim as usize);
    let is_active = (tid as usize) < (head_dim as usize);

    if is_active {
        let mut i = tid as usize;
        while i < (head_dim as usize) {
            let pos_in_head = i;

            if pos_in_head < half_head {
                // First half: x' = x * cos - y * sin
                let x = input[base_idx + pos_in_head];
                let y = input[base_idx + pos_in_head + half_head];

                let cache_idx = (seq_idx as usize) * half_head + pos_in_head;
                let cos_val = cos_cache[cache_idx];
                let sin_val = sin_cache[cache_idx];

                output[base_idx + pos_in_head] = x * cos_val - y * sin_val;
            } else {
                // Second half: y' = x * sin + y * cos
                let local_pos = pos_in_head - half_head;
                let x = input[base_idx + local_pos];
                let y = input[base_idx + pos_in_head];

                let cache_idx = (seq_idx as usize) * half_head + local_pos;
                let cos_val = cos_cache[cache_idx];
                let sin_val = sin_cache[cache_idx];

                output[base_idx + pos_in_head] = x * sin_val + y * cos_val;
            }

            i = i + (block_size as usize);
        }
    }
}

// ============================================================================
// Public API Functions
// ============================================================================

/// Apply RMSNorm to input tensor using CubeCL GPU kernel.
///
/// # Arguments
/// * `input` - Input tensor [..., hidden_dim]
/// * `weight` - Normalization weights [hidden_dim]
/// * `eps` - Epsilon for numerical stability
///
/// # Returns
/// Normalized tensor with same shape as input
pub fn rmsnorm(input: &Tensor, weight: &Tensor, eps: f64) -> UnslothResult<Tensor> {
    // Validate inputs
    let input_shape = input.dims();
    let weight_shape = weight.dims();

    if weight_shape.len() != 1 {
        return Err(UnslothError::InvalidConfig(format!(
            "RMSNorm weight must be 1D, got shape {:?}",
            weight_shape
        )));
    }

    let hidden_dim = weight_shape[0];
    if input_shape.last() != Some(&hidden_dim) {
        return Err(UnslothError::InvalidConfig(format!(
            "Input last dim {} doesn't match weight dim {}",
            input_shape.last().unwrap_or(&0),
            hidden_dim
        )));
    }

    // Check for CUDA support
    #[cfg(feature = "cuda")]
    {
        if input.device().is_cuda() {
            return launch_rmsnorm_kernel(input, weight, eps);
        }
    }

    // CPU fallback
    rmsnorm_cpu(input, weight, eps)
}

/// Apply fused RMSNorm + RoPE using CubeCL GPU kernel.
///
/// This is the primary optimization for transformer inference.
/// Combines normalization and position encoding in a single kernel pass.
///
/// # Arguments
/// * `input` - Input tensor [batch, seq_len, hidden_dim]
/// * `weight` - RMSNorm weights [hidden_dim]
/// * `cos_cache` - Precomputed cosine values [max_seq, head_dim/2]
/// * `sin_cache` - Precomputed sine values [max_seq, head_dim/2]
/// * `head_dim` - Dimension per attention head
/// * `num_heads` - Number of attention heads
/// * `eps` - Epsilon for numerical stability
///
/// # Returns
/// Tensor with RMSNorm and RoPE applied
pub fn fused_rmsnorm_rope(
    input: &Tensor,
    weight: &Tensor,
    cos_cache: &Tensor,
    sin_cache: &Tensor,
    head_dim: usize,
    num_heads: usize,
    eps: f64,
) -> UnslothResult<Tensor> {
    let input_shape = input.dims();
    if input_shape.len() != 3 {
        return Err(UnslothError::InvalidConfig(format!(
            "Expected 3D input [batch, seq_len, hidden_dim], got {:?}",
            input_shape
        )));
    }

    #[cfg(feature = "cuda")]
    let batch_size = input_shape[0];
    #[cfg(feature = "cuda")]
    let seq_len = input_shape[1];
    let hidden_dim = input_shape[2];

    // Validate dimensions
    if hidden_dim != head_dim * num_heads {
        return Err(UnslothError::InvalidConfig(format!(
            "hidden_dim {} != head_dim {} * num_heads {}",
            hidden_dim, head_dim, num_heads
        )));
    }

    #[cfg(feature = "cuda")]
    {
        if input.device().is_cuda() {
            return launch_fused_rmsnorm_rope_kernel(
                input, weight, cos_cache, sin_cache, batch_size, seq_len, hidden_dim, head_dim,
                num_heads, eps,
            );
        }
    }

    // CPU fallback: apply RMSNorm then RoPE separately
    fused_rmsnorm_rope_cpu(input, weight, cos_cache, sin_cache, head_dim, eps)
}

/// Apply RoPE to Q/K tensors using CubeCL GPU kernel.
///
/// # Arguments
/// * `input` - Input tensor [batch, num_heads, seq_len, head_dim]
/// * `cos_cache` - Precomputed cos [max_seq, head_dim/2]
/// * `sin_cache` - Precomputed sin [max_seq, head_dim/2]
///
/// # Returns
/// Tensor with RoPE applied
pub fn rope(input: &Tensor, cos_cache: &Tensor, sin_cache: &Tensor) -> UnslothResult<Tensor> {
    let input_shape = input.dims();
    if input_shape.len() != 4 {
        return Err(UnslothError::InvalidConfig(format!(
            "Expected 4D input [batch, heads, seq, dim], got {:?}",
            input_shape
        )));
    }

    #[cfg(feature = "cuda")]
    {
        if input.device().is_cuda() {
            return launch_rope_kernel(input, cos_cache, sin_cache);
        }
    }

    // CPU fallback
    rope_cpu(input, cos_cache, sin_cache)
}

// ============================================================================
// Kernel Launch Functions
// ============================================================================

#[cfg(feature = "cuda")]
fn launch_rmsnorm_kernel(input: &Tensor, weight: &Tensor, eps: f64) -> UnslothResult<Tensor> {
    use crate::kernels::cubecl::interop::{candle_to_cubecl_handle, cubecl_to_candle_tensor};

    let input_shape = input.dims();
    let hidden_dim = input_shape[input_shape.len() - 1];
    let num_rows: usize = input_shape[..input_shape.len() - 1].iter().product();

    // Convert to bytes
    let (input_bytes, _, _) = candle_to_cubecl_handle(input)?;
    let (weight_bytes, _, _) = candle_to_cubecl_handle(weight)?;

    let num_elements = num_rows * hidden_dim;

    // Get CUDA client
    let device = cubecl_cuda::CudaDevice::new(0);
    let client = CudaRuntime::client(&device);

    // Create handles - CubeCL 0.9 requires Bytes type
    let input_handle = client.create(cubecl::bytes::Bytes::from_bytes_vec(input_bytes));
    let weight_handle = client.create(cubecl::bytes::Bytes::from_bytes_vec(weight_bytes));
    let output_handle = client.empty(num_elements * std::mem::size_of::<f32>());

    // Launch configuration
    let block_size = (hidden_dim as u32).min(MAX_BLOCK_SIZE).next_power_of_two();
    let cube_count = CubeCount::Static(num_rows as u32, 1, 1);
    let cube_dim = CubeDim::new(&client, block_size as usize);

    // SAFETY: Handles are valid and properly sized for the kernel operation
    unsafe {
        rmsnorm_kernel::launch::<f32, CudaRuntime>(
            &client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts::<f32>(&input_handle, num_elements, 1),
            ArrayArg::from_raw_parts::<f32>(&weight_handle, hidden_dim, 1),
            ArrayArg::from_raw_parts::<f32>(&output_handle, num_elements, 1),
            ScalarArg::new(hidden_dim as u32),
            ScalarArg::new(eps as f32),
            ScalarArg::new(block_size),
        )
        .map_err(|e| UnslothError::Kernel(format!("rmsnorm_kernel launch failed: {e}")))?;
    }

    let output_bytes = client.read_one(output_handle);
    cubecl_to_candle_tensor(&output_bytes, input_shape, input.device())
}

#[cfg(feature = "cuda")]
fn launch_fused_rmsnorm_rope_kernel(
    input: &Tensor,
    weight: &Tensor,
    cos_cache: &Tensor,
    sin_cache: &Tensor,
    batch_size: usize,
    seq_len: usize,
    hidden_dim: usize,
    head_dim: usize,
    num_heads: usize,
    eps: f64,
) -> UnslothResult<Tensor> {
    use crate::kernels::cubecl::interop::{candle_to_cubecl_handle, cubecl_to_candle_tensor};

    // Convert to bytes
    let (input_bytes, _, _) = candle_to_cubecl_handle(input)?;
    let (weight_bytes, _, _) = candle_to_cubecl_handle(weight)?;
    let (cos_bytes, _, _) = candle_to_cubecl_handle(cos_cache)?;
    let (sin_bytes, _, _) = candle_to_cubecl_handle(sin_cache)?;

    let num_elements = batch_size * seq_len * hidden_dim;
    let cache_elements = cos_cache.dims().iter().product::<usize>();

    // Get CUDA client
    let device = cubecl_cuda::CudaDevice::new(0);
    let client = CudaRuntime::client(&device);

    // Create handles - CubeCL 0.9 requires Bytes type
    let input_handle = client.create(cubecl::bytes::Bytes::from_bytes_vec(input_bytes));
    let weight_handle = client.create(cubecl::bytes::Bytes::from_bytes_vec(weight_bytes));
    let cos_handle = client.create(cubecl::bytes::Bytes::from_bytes_vec(cos_bytes));
    let sin_handle = client.create(cubecl::bytes::Bytes::from_bytes_vec(sin_bytes));
    let output_handle = client.empty(num_elements * std::mem::size_of::<f32>());

    // Launch configuration
    let block_size = (hidden_dim as u32).min(MAX_BLOCK_SIZE).next_power_of_two();
    let cube_count = CubeCount::Static(batch_size as u32, seq_len as u32, 1);
    let cube_dim = CubeDim::new(&client, block_size as usize);

    // SAFETY: Handles are valid and properly sized for the kernel operation
    unsafe {
        fused_rmsnorm_rope_kernel::launch::<f32, CudaRuntime>(
            &client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts::<f32>(&input_handle, num_elements, 1),
            ArrayArg::from_raw_parts::<f32>(&weight_handle, hidden_dim, 1),
            ArrayArg::from_raw_parts::<f32>(&cos_handle, cache_elements, 1),
            ArrayArg::from_raw_parts::<f32>(&sin_handle, cache_elements, 1),
            ArrayArg::from_raw_parts::<f32>(&output_handle, num_elements, 1),
            ScalarArg::new(batch_size as u32),
            ScalarArg::new(seq_len as u32),
            ScalarArg::new(hidden_dim as u32),
            ScalarArg::new(head_dim as u32),
            ScalarArg::new(num_heads as u32),
            ScalarArg::new(eps as f32),
            ScalarArg::new(block_size),
        )
        .map_err(|e| {
            UnslothError::Kernel(format!("fused_rmsnorm_rope_kernel launch failed: {e}"))
        })?;
    }

    let output_bytes = client.read_one(output_handle);
    cubecl_to_candle_tensor(&output_bytes, input.dims(), input.device())
}

#[cfg(feature = "cuda")]
fn launch_rope_kernel(
    input: &Tensor,
    cos_cache: &Tensor,
    sin_cache: &Tensor,
) -> UnslothResult<Tensor> {
    use crate::kernels::cubecl::interop::{candle_to_cubecl_handle, cubecl_to_candle_tensor};

    let dims = input.dims();
    let batch_size = dims[0];
    let num_heads = dims[1];
    let seq_len = dims[2];
    let head_dim = dims[3];

    // Convert to bytes
    let (input_bytes, _, _) = candle_to_cubecl_handle(input)?;
    let (cos_bytes, _, _) = candle_to_cubecl_handle(cos_cache)?;
    let (sin_bytes, _, _) = candle_to_cubecl_handle(sin_cache)?;

    let num_elements = batch_size * num_heads * seq_len * head_dim;
    let cache_elements = cos_cache.dims().iter().product::<usize>();

    // Get CUDA client
    let device = cubecl_cuda::CudaDevice::new(0);
    let client = CudaRuntime::client(&device);

    // Create handles - CubeCL 0.9 requires Bytes type
    let input_handle = client.create(cubecl::bytes::Bytes::from_bytes_vec(input_bytes));
    let cos_handle = client.create(cubecl::bytes::Bytes::from_bytes_vec(cos_bytes));
    let sin_handle = client.create(cubecl::bytes::Bytes::from_bytes_vec(sin_bytes));
    let output_handle = client.empty(num_elements * std::mem::size_of::<f32>());

    // Launch configuration
    let block_size = (head_dim as u32).min(MAX_BLOCK_SIZE).next_power_of_two();
    let cube_count = CubeCount::Static((batch_size * num_heads) as u32, seq_len as u32, 1);
    let cube_dim = CubeDim::new(&client, block_size as usize);

    // SAFETY: Handles are valid and properly sized for the kernel operation
    unsafe {
        rope_kernel::launch::<f32, CudaRuntime>(
            &client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts::<f32>(&input_handle, num_elements, 1),
            ArrayArg::from_raw_parts::<f32>(&cos_handle, cache_elements, 1),
            ArrayArg::from_raw_parts::<f32>(&sin_handle, cache_elements, 1),
            ArrayArg::from_raw_parts::<f32>(&output_handle, num_elements, 1),
            ScalarArg::new(batch_size as u32),
            ScalarArg::new(num_heads as u32),
            ScalarArg::new(seq_len as u32),
            ScalarArg::new(head_dim as u32),
            ScalarArg::new(block_size),
        )
        .map_err(|e| UnslothError::Kernel(format!("rope_kernel launch failed: {e}")))?;
    }

    let output_bytes = client.read_one(output_handle);
    cubecl_to_candle_tensor(&output_bytes, dims, input.device())
}

// ============================================================================
// CPU Fallback Implementations
// ============================================================================

fn rmsnorm_cpu(input: &Tensor, weight: &Tensor, eps: f64) -> UnslothResult<Tensor> {
    // RMS = sqrt(mean(x^2) + eps)
    let x_sq = input.sqr()?;
    let mean_sq = x_sq.mean_keepdim(input.rank() - 1)?;
    let rms = (mean_sq + eps)?.sqrt()?;

    // Normalize and scale
    let normalized = input.broadcast_div(&rms)?;
    let output = normalized.broadcast_mul(weight)?;

    Ok(output)
}

fn fused_rmsnorm_rope_cpu(
    input: &Tensor,
    weight: &Tensor,
    cos_cache: &Tensor,
    sin_cache: &Tensor,
    head_dim: usize,
    eps: f64,
) -> UnslothResult<Tensor> {
    // Step 1: Apply RMSNorm
    let normalized = rmsnorm_cpu(input, weight, eps)?;

    // Step 2: Apply RoPE
    // Input is [batch, seq_len, hidden_dim]
    // Need to split into heads and apply rotation
    let dims = normalized.dims();
    let batch = dims[0];
    let seq_len = dims[1];
    let hidden_dim = dims[2];
    let num_heads = hidden_dim / head_dim;
    let half_dim = head_dim / 2;

    // Reshape to [batch, seq_len, num_heads, head_dim]
    let reshaped = normalized.reshape((batch, seq_len, num_heads, head_dim))?;

    // Get cos/sin for positions
    let cos = cos_cache.narrow(0, 0, seq_len)?;
    let sin = sin_cache.narrow(0, 0, seq_len)?;

    // Reshape cos/sin for broadcast: [seq_len, half_dim] -> [1, seq_len, 1, half_dim]
    let cos = cos.unsqueeze(0)?.unsqueeze(2)?;
    let sin = sin.unsqueeze(0)?.unsqueeze(2)?;

    // Split into halves
    let x1 = reshaped.narrow(3, 0, half_dim)?;
    let x2 = reshaped.narrow(3, half_dim, half_dim)?;

    // Apply rotation
    let rotated_x1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
    let rotated_x2 = (x2.broadcast_mul(&cos)? + x1.broadcast_mul(&sin)?)?;

    // Concatenate and reshape back
    let rotated = Tensor::cat(&[&rotated_x1, &rotated_x2], 3)?;
    let output = rotated.reshape((batch, seq_len, hidden_dim))?;

    Ok(output)
}

fn rope_cpu(input: &Tensor, cos_cache: &Tensor, sin_cache: &Tensor) -> UnslothResult<Tensor> {
    let dims = input.dims();
    let seq_len = dims[2];
    let head_dim = dims[3];
    let half_dim = head_dim / 2;

    // Get cos/sin for positions
    let cos = cos_cache.narrow(0, 0, seq_len)?;
    let sin = sin_cache.narrow(0, 0, seq_len)?;

    // Reshape cos/sin for broadcast: [seq_len, half_dim] -> [1, 1, seq_len, half_dim]
    let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
    let sin = sin.unsqueeze(0)?.unsqueeze(0)?;

    // Split into halves
    let x1 = input.narrow(3, 0, half_dim)?;
    let x2 = input.narrow(3, half_dim, half_dim)?;

    // Apply rotation
    let rotated_x1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
    let rotated_x2 = (x2.broadcast_mul(&cos)? + x1.broadcast_mul(&sin)?)?;

    // Concatenate
    Tensor::cat(&[&rotated_x1, &rotated_x2], 3).map_err(Into::into)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_rmsnorm_cpu() {
        let device = Device::Cpu;
        let hidden_dim = 64;
        let batch_size = 2;
        let seq_len = 4;

        let input = Tensor::randn(0.0f32, 1.0, (batch_size, seq_len, hidden_dim), &device).unwrap();
        let weight = Tensor::ones((hidden_dim,), DType::F32, &device).unwrap();

        let output = rmsnorm(&input, &weight, 1e-5).unwrap();
        assert_eq!(output.dims(), input.dims());

        // Check no NaN/Inf
        let vals: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();
        for v in vals {
            assert!(!v.is_nan() && !v.is_infinite());
        }
    }

    #[test]
    fn test_rope_cpu() {
        let device = Device::Cpu;
        let batch = 2;
        let num_heads = 4;
        let seq_len = 8;
        let head_dim = 64;
        let half_dim = head_dim / 2;

        let input =
            Tensor::randn(0.0f32, 1.0, (batch, num_heads, seq_len, head_dim), &device).unwrap();
        let cos_cache = Tensor::ones((seq_len, half_dim), DType::F32, &device).unwrap();
        let sin_cache = Tensor::zeros((seq_len, half_dim), DType::F32, &device).unwrap();

        let output = rope(&input, &cos_cache, &sin_cache).unwrap();
        assert_eq!(output.dims(), input.dims());
    }

    #[test]
    fn test_fused_rmsnorm_rope_cpu() {
        let device = Device::Cpu;
        let batch = 2;
        let seq_len = 8;
        let num_heads = 4;
        let head_dim = 64;
        let hidden_dim = num_heads * head_dim;
        let half_dim = head_dim / 2;

        let input = Tensor::randn(0.0f32, 1.0, (batch, seq_len, hidden_dim), &device).unwrap();
        let weight = Tensor::ones((hidden_dim,), DType::F32, &device).unwrap();
        let cos_cache = Tensor::ones((seq_len, half_dim), DType::F32, &device).unwrap();
        let sin_cache = Tensor::zeros((seq_len, half_dim), DType::F32, &device).unwrap();

        let output = fused_rmsnorm_rope(
            &input, &weight, &cos_cache, &sin_cache, head_dim, num_heads, 1e-5,
        )
        .unwrap();
        assert_eq!(output.dims(), input.dims());
    }
}
