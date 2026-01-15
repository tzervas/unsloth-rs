// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! Flash Attention `CubeCL` kernel implementation.
//!
//! This module contains the actual `CubeCL` kernel for Flash Attention 2.
//! The implementation uses tiled computation with online softmax for
//! O(N) memory complexity.
//!
//! ## Algorithm Overview
//!
//! Flash Attention processes attention in tiles:
//!
//! ```text
//! For each Q tile (i = 0..Tr):
//!     Load Q_i into shared memory
//!     Initialize accumulators: O_i = 0, m_i = -∞, l_i = 0
//!     
//!     For each KV tile (j = 0..Tc):
//!         Load K_j, V_j into shared memory
//!         S_ij = Q_i @ K_j^T / sqrt(d)
//!         
//!         # Online softmax update
//!         m_ij = max(m_i, rowmax(S_ij))
//!         P_ij = exp(S_ij - m_ij)
//!         l_ij = exp(m_i - m_ij) * l_i + rowsum(P_ij)
//!         
//!         # Output update with correction
//!         O_i = (l_i * exp(m_i - m_ij) * O_i + P_ij @ V_j) / l_ij
//!         
//!         m_i = m_ij
//!         l_i = l_ij
//!     
//!     Store O_i to global memory
//! ```
//!
//! ## `CubeCL` API (v0.8.1)
//!
//! Key constructs used:
//! - `#[cube(launch_unchecked)]` - Kernel definition (unchecked for performance)
//! - `Array<Line<F>>` - Vectorized 4-element loads for coalescing
//! - `SharedMemory::<F>::new(comptime_size)` - 1D shared memory allocation
//! - `ABSOLUTE_POS`, `UNIT_POS`, `CUBE_POS` - Thread indexing primitives
//! - `sync_cube()` - Block-level barrier
//! - `warp_reduce` - Warp-level reductions for max/sum
//!
//! ## Implementation Status
//!
//! Phase 1 (Minimal Viable Kernel): In Progress
//! - Target: RTX 5080, f32, non-masked, equal Q/K/V lengths
//! - Goal: Correct results, 2x speedup vs Candle fallback

use super::config::FlashAttentionConfig;
use super::interop::has_cubecl_cuda_support;
use crate::error::{Result, UnslothError};
use candle_core::Tensor;

// CubeCL imports for kernel implementation
#[cfg(feature = "cuda")]
use cubecl::prelude::*;
#[cfg(feature = "cuda")]
use cubecl_cuda::CudaRuntime;

// ============================================================================
// CubeCL Kernel Definition (v0.8.1 API)
// ============================================================================

/// Compile-time configuration for Flash Attention kernels.
/// This is used by the launch wrapper, not directly by CubeCL.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg(feature = "cuda")]
pub struct TileConfig {
    /// Size of tiles for tiled attention
    pub tile_size: u32,
    /// Head dimension (must match tensor dim)
    pub head_dim: u32,
    /// Sequence length for this batch
    pub seq_len: u32,
    /// Number of KV tiles to iterate over
    pub num_kv_tiles: u32,
    /// Whether to apply causal masking (upper triangular mask)
    pub causal: bool,
}

/// Flash Attention forward kernel - Phase 1 simplified implementation.
///
/// This kernel computes attention for a single tile row.
/// For Phase 1, we use a simplified approach:
/// - Each block handles one (batch, head, q_tile) combination
/// - Threads within block cooperatively process elements
///
/// Memory layout: [batch, heads, seq_len, head_dim] stored contiguously.
#[cfg(feature = "cuda")]
#[cube(launch)]
fn flash_attention_tile<F: Float>(
    q: &Array<F>,       // Query [batch * heads * seq_len * head_dim]
    k: &Array<F>,       // Key [batch * heads * seq_len * head_dim]
    v: &Array<F>,       // Value [batch * heads * seq_len * head_dim]
    out: &mut Array<F>, // Output [batch * heads * seq_len * head_dim]
    scale: F,           // 1/sqrt(head_dim)
    // Runtime parameters for dimensions
    seq_len_val: u32,
    head_dim_val: u32,
) {
    // Thread/block indices
    let batch_head_idx = CUBE_POS_X; // Which (batch, head) pair
    let q_row_idx = CUBE_POS_Y; // Which Q row within this batch-head
    let tid = UNIT_POS_X; // Thread within block

    // Strides for [batch*heads, seq_len, head_dim] layout
    let head_stride = seq_len_val * head_dim_val;

    // Base offset for this batch-head
    let base_offset = batch_head_idx * head_stride;

    // This thread handles one element in the head_dim dimension
    // Each row has head_dim elements; we process them in parallel
    if tid >= head_dim_val {
        terminate!();
    }

    // Initialize running statistics for online softmax
    let mut running_max = F::new(-1e30); // Running max of attention scores
    let mut running_sum = F::new(0.0); // Running sum for normalization
    let mut running_out = F::new(0.0); // Running output accumulator

    // Get Q value for this thread's position
    let q_offset = base_offset + q_row_idx * head_dim_val + tid;
    let q_val = q[q_offset];

    // Iterate over all K/V positions (tile-less for Phase 1)
    for kv_idx in 0..seq_len_val {
        // Compute dot product Q[row] @ K[kv_idx] (single element contribution)
        let k_offset = base_offset + kv_idx * head_dim_val + tid;
        let k_val = k[k_offset];

        // Each thread computes one term of the dot product
        // We need to reduce across threads to get the full score
        let score_contrib = q_val * k_val;

        // Warp-level reduction for dot product (sum across head_dim threads)
        // Note: For simplicity in Phase 1, we use shared memory reduction
        // Full implementation will use warp_reduce
        // FIXME: Hardcoded shared memory size of 256 elements
        // WARNING: These limitations will cause INCORRECT RESULTS in the following cases:
        // - Threads with tid >= 256 will be out of bounds when writing to score_tile[tid]
        // - Tree reduction assumes head_dim_val is a power of 2, starting with stride = head_dim_val / 2
        // - For non-power-of-2 head_dim (e.g., 80), some elements won't be included in reduction
        // - Shared memory size should match block size, and reduction should handle non-power-of-2
        // These limitations will be addressed in Phase 2 with proper warp-level primitives
        let mut score_tile = SharedMemory::<F>::new(256); // Block size
        score_tile[tid] = score_contrib;
        sync_cube();

        // Tree reduction for sum
        let mut stride: u32 = head_dim_val / 2;
        while stride > 0 {
            if tid < stride {
                score_tile[tid] = score_tile[tid] + score_tile[tid + stride];
            }
            sync_cube();
            stride = stride / 2;
        }

        // Thread 0 has the full dot product
        let score = score_tile[0] * scale;

        // Broadcast score to all threads via shared memory
        if tid == 0 {
            score_tile[0] = score;
        }
        sync_cube();
        let attn_score = score_tile[0];

        // Online softmax update
        let new_max = F::max(running_max, attn_score);
        let exp_old = F::exp(running_max - new_max);
        let exp_new = F::exp(attn_score - new_max);

        // Update running sum
        let new_sum = exp_old * running_sum + exp_new;

        // Update output: scale old output and add new contribution
        let v_offset = base_offset + kv_idx * head_dim_val + tid;
        let v_val = v[v_offset];
        running_out = (exp_old * running_sum * running_out + exp_new * v_val) / new_sum;

        // Update statistics
        running_max = new_max;
        running_sum = new_sum;
    }

    // Write output
    let out_offset = base_offset + q_row_idx * head_dim_val + tid;
    out[out_offset] = running_out;
}

/// Improved Flash Attention kernel with proper tiling (Phase 1.5).
///
/// This kernel implements a more efficient tiled approach:
/// - Each block processes a tile of Q rows (tile_size rows)
/// - Threads cooperatively load Q/K/V tiles into shared memory
/// - Supports causal masking for autoregressive attention
///
/// Algorithm (per Q tile):
/// ```text
/// For q_tile_idx in 0..num_q_tiles:
///     Load Q[q_tile_idx] to shared memory
///     Initialize accumulators: m_i = -inf, l_i = 0, O_i = 0
///     
///     For kv_tile_idx in 0..num_kv_tiles:
///         Load K[kv_tile_idx], V[kv_tile_idx] to shared memory
///         Compute S_ij = Q_tile @ K_tile^T * scale
///         Apply causal mask if needed
///         
///         # Online softmax update
///         m_new = max(m_i, rowmax(S_ij))
///         P_ij = exp(S_ij - m_new)
///         l_new = exp(m_i - m_new) * l_i + rowsum(P_ij)
///         O_i = (exp(m_i - m_new) * l_i * O_i + P_ij @ V_tile) / l_new
///         
///         m_i = m_new, l_i = l_new
///     
///     Write O_i to global memory
/// ```
// TODO: Re-enable once CubeCL comptime bool issue is resolved
#[cfg(all(feature = "cuda", feature = "_phase2_tiled_kernel"))]
#[cube(launch)]
#[allow(dead_code)]
fn flash_attention_tiled<F: Float>(
    q: &Array<F>,       // Query [batch * heads * seq_len * head_dim]
    k: &Array<F>,       // Key [batch * heads * seq_len * head_dim]
    v: &Array<F>,       // Value [batch * heads * seq_len * head_dim]
    out: &mut Array<F>, // Output [batch * heads * seq_len * head_dim]
    scale: F,           // 1/sqrt(head_dim)
    #[comptime] tile_size: u32,
    #[comptime] head_dim: u32,
    #[comptime] seq_len: u32,
    #[comptime] num_kv_tiles: u32,
    #[comptime] causal: bool,
) {
    // Grid: (batch * heads, num_q_tiles, 1)
    // Block: (tile_size, 1, 1) - each thread handles one Q row

    let batch_head_idx = CUBE_POS_X;
    let q_tile_idx = CUBE_POS_Y;
    let thread_in_tile = UNIT_POS_X;

    // Calculate which Q row this thread handles globally
    let q_row_global = q_tile_idx * tile_size + thread_in_tile;

    // Early exit if beyond sequence length
    if q_row_global >= seq_len {
        terminate!();
    }

    // Stride calculations
    let head_stride = seq_len * head_dim;
    let base_offset = batch_head_idx * head_stride;

    // Allocate shared memory for Q tile (tile_size × head_dim)
    // Each thread loads head_dim elements for its Q row
    let mut q_tile = SharedMemory::<F>::new(tile_size * head_dim);

    // Load Q tile cooperatively
    // Thread i loads Q[q_row_global, :] into q_tile[thread_in_tile, :]
    for dim_idx in 0..head_dim {
        let q_offset = base_offset + q_row_global * head_dim + dim_idx;
        let tile_offset = thread_in_tile * head_dim + dim_idx;
        q_tile[tile_offset] = q[q_offset];
    }
    sync_cube();

    // Initialize per-row accumulators (in registers)
    let mut running_max = F::new(-1e30);
    let mut running_sum = F::new(0.0);

    // Output accumulator (head_dim elements per thread)
    let mut out_acc = SharedMemory::<F>::new(tile_size * head_dim);
    for dim_idx in 0..head_dim {
        out_acc[thread_in_tile * head_dim + dim_idx] = F::new(0.0);
    }

    // Iterate over KV tiles
    for kv_tile_idx in 0..num_kv_tiles {
        let kv_start = kv_tile_idx * tile_size;
        // Compute actual tile size (min of tile_size and remaining seq_len)
        let kv_end = if kv_start + tile_size < seq_len {
            kv_start + tile_size
        } else {
            seq_len
        };
        let kv_tile_actual_size = kv_end - kv_start;

        // Allocate shared memory for K and V tiles
        let mut k_tile = SharedMemory::<F>::new(tile_size * head_dim);
        let mut v_tile = SharedMemory::<F>::new(tile_size * head_dim);

        // Load K and V tiles cooperatively
        // Each thread loads multiple rows if needed
        for local_kv_idx in 0..kv_tile_actual_size {
            if local_kv_idx % tile_size == thread_in_tile {
                let kv_row_global = kv_start + local_kv_idx;
                for dim_idx in 0..head_dim {
                    let k_offset = base_offset + kv_row_global * head_dim + dim_idx;
                    let v_offset = base_offset + kv_row_global * head_dim + dim_idx;
                    let tile_offset = local_kv_idx * head_dim + dim_idx;
                    k_tile[tile_offset] = k[k_offset];
                    v_tile[tile_offset] = v[v_offset];
                }
            }
        }
        sync_cube();

        // Compute attention scores for this thread's Q row against all KV rows in tile
        for local_kv_idx in 0..kv_tile_actual_size {
            let kv_row_global = kv_start + local_kv_idx;

            // Apply causal mask: if causal and q_row < kv_row, skip this KV row
            // Note: CubeCL doesn't support `continue`, so we use if-else instead
            let should_process = if causal {
                q_row_global >= kv_row_global
            } else {
                true
            };

            if should_process {
                // Compute dot product Q[thread_in_tile] @ K[local_kv_idx]
                let mut score = F::new(0.0);
                for dim_idx in 0..head_dim {
                    let q_val = q_tile[thread_in_tile * head_dim + dim_idx];
                    let k_val = k_tile[local_kv_idx * head_dim + dim_idx];
                    score = score + q_val * k_val;
                }
                score = score * scale;

                // Online softmax update
                let new_max = F::max(running_max, score);
                let exp_old = F::exp(running_max - new_max);
                let exp_new = F::exp(score - new_max);

                // Update running sum with correction factor
                let new_sum = exp_old * running_sum + exp_new;

                // Update output: O_new = (exp_old * l_old * O_old + exp_new * V) / l_new
                for dim_idx in 0..head_dim {
                    let out_offset = thread_in_tile * head_dim + dim_idx;
                    let old_out = out_acc[out_offset];
                    let v_val = v_tile[local_kv_idx * head_dim + dim_idx];

                    // Apply correction and add new contribution
                    let corrected_old = exp_old * running_sum * old_out;
                    let new_contrib = exp_new * v_val;
                    out_acc[out_offset] = (corrected_old + new_contrib) / new_sum;
                }

                // Update statistics
                running_max = new_max;
                running_sum = new_sum;
            }
        }
        sync_cube();
    }

    // Write output to global memory
    for dim_idx in 0..head_dim {
        let out_offset = base_offset + q_row_global * head_dim + dim_idx;
        let tile_offset = thread_in_tile * head_dim + dim_idx;
        out[out_offset] = out_acc[tile_offset];
    }
}

/// Launch Flash Attention `CubeCL` kernel.
///
/// This is the main entry point for the `CubeCL` Flash Attention implementation.
/// It handles device detection, tensor conversion, kernel launch, and result
/// conversion back to Candle tensors.
///
/// # Arguments
///
/// * `q` - Query tensor `[batch, num_heads, seq_len, head_dim]`
/// * `k` - Key tensor `[batch, num_kv_heads, seq_len, head_dim]`
/// * `v` - Value tensor `[batch, num_kv_heads, seq_len, head_dim]`
/// * `scale` - Attention scale factor (typically `1/sqrt(head_dim)`)
/// * `mask` - Optional attention mask
/// * `config` - Kernel configuration
///
/// # Returns
///
/// Attention output tensor `[batch, num_heads, seq_len, head_dim]`
///
/// # Errors
///
/// Returns error if:
/// - `CubeCL` CUDA is not available (falls back to Candle)
/// - Tensor shapes are incompatible
/// - Kernel launch fails
///
/// # Example
///
/// ```rust,ignore
/// use unsloth_rs::kernels::cubecl::{flash_attention_kernel, FlashAttentionConfig};
///
/// let config = FlashAttentionConfig::for_rtx_5080().with_causal_mask();
/// let output = flash_attention_kernel(&q, &k, &v, scale, None, &config)?;
/// ```
pub fn flash_attention_kernel(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    mask: Option<&Tensor>,
    config: &FlashAttentionConfig,
) -> Result<Tensor> {
    // Validate input shapes
    validate_attention_inputs(q, k, v)?;

    // Check for CubeCL support
    if !has_cubecl_cuda_support() {
        tracing::debug!("CubeCL CUDA not available, using fallback implementation");
        return fallback_attention(q, k, v, scale, mask, config);
    }

    // Try to launch CubeCL kernel; fall back on any error
    #[cfg(feature = "cuda")]
    {
        match launch_cubecl_attention(q, k, v, scale, config) {
            Ok(output) => return Ok(output),
            Err(e) => {
                tracing::warn!("CubeCL kernel launch failed: {}, using fallback", e);
            }
        }
    }

    fallback_attention(q, k, v, scale, mask, config)
}

/// Launch the actual CubeCL Flash Attention kernel.
///
/// This is the internal implementation that handles:
/// 1. CubeCL runtime initialization
/// 2. Tensor conversion to CubeCL handles
/// 3. Kernel launch with proper grid/block configuration
/// 4. Result conversion back to Candle tensor
#[cfg(feature = "cuda")]
fn launch_cubecl_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    config: &FlashAttentionConfig,
) -> Result<Tensor> {
    use super::interop::{candle_to_cubecl_handle, cubecl_to_candle_tensor};

    // Extract dimensions
    let dims = q.dims();
    let batch = dims[0];
    let num_heads = dims[1];
    let seq_len = dims[2];
    let head_dim = dims[3];

    tracing::debug!(
        "Launching CubeCL Flash Attention: batch={}, heads={}, seq={}, dim={}",
        batch,
        num_heads,
        seq_len,
        head_dim
    );

    // Convert tensors to byte arrays
    // PERF: This performs inefficient GPU→CPU→GPU transfers
    // candle_to_cubecl_handle() copies CUDA tensor data to CPU (see interop.rs:131),
    // then client.create() copies it back to GPU. For CUDA tensors, we should
    // extract the GPU pointer directly without round-tripping through CPU memory.
    // This significantly impacts performance and will be optimized in Phase 2.
    // TODO: Implement direct GPU pointer extraction for CUDA tensors
    let (q_bytes, _, _) = candle_to_cubecl_handle(q)?;
    let (k_bytes, _, _) = candle_to_cubecl_handle(k)?;
    let (v_bytes, _, _) = candle_to_cubecl_handle(v)?;

    // Allocate output buffer
    let num_elements = batch * num_heads * seq_len * head_dim;

    // Get CubeCL CUDA client
    let device = cubecl_cuda::CudaDevice::new(0);
    let client = CudaRuntime::client(&device);

    // Create CubeCL handles - data is already in bytes from candle_to_cubecl_handle
    let q_handle = client.create(&q_bytes);
    let k_handle = client.create(&k_bytes);
    let v_handle = client.create(&v_bytes);
    let out_handle = client.empty(num_elements * std::mem::size_of::<f32>());

    // Configure kernel launch (tile_config preserved for future tiled kernel)
    let _tile_config = TileConfig {
        tile_size: config.tile_size,
        head_dim: head_dim as u32,
        seq_len: seq_len as u32,
        num_kv_tiles: config.num_kv_tiles(seq_len as u32),
        causal: config.causal_mask,
    };

    // Grid: (batch * heads, seq_len) - one block per (batch-head, q_row)
    let cube_count = CubeCount::Static((batch * num_heads) as u32, seq_len as u32, 1);

    // Block: head_dim threads (each handles one element)
    // FIXME: Clamping to 256 threads is incorrect for head_dim > 256
    // The kernel's reduction logic assumes all threads up to head_dim participate.
    // Currently only supports head_dim <= 256. For larger head_dim, the kernel
    // needs rewriting to handle multi-pass reduction or tiled computation.
    // See kernel.rs lines 144-162 for the reduction code that makes this assumption.
    let block_size = Ord::min(head_dim as u32, 256);
    if head_dim > 256 {
        tracing::warn!(
            "head_dim={} exceeds maximum supported block size of 256. \
             Results WILL BE INCORRECT. This will be fixed in Phase 2 tiled kernel.",
            head_dim
        );
    }
    let cube_dim = CubeDim::new(block_size, 1, 1);

    // Scale as f32
    let scale_f32 = scale as f32;

    // Launch kernel with runtime dimension parameters
    // SAFETY: ArrayArg::from_raw_parts requires handles to be valid and num_elements to match
    unsafe {
        flash_attention_tile::launch::<f32, CudaRuntime>(
            &client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts::<f32>(&q_handle, num_elements, 1),
            ArrayArg::from_raw_parts::<f32>(&k_handle, num_elements, 1),
            ArrayArg::from_raw_parts::<f32>(&v_handle, num_elements, 1),
            ArrayArg::from_raw_parts::<f32>(&out_handle, num_elements, 1),
            ScalarArg::new(scale_f32),
            ScalarArg::new(seq_len as u32),
            ScalarArg::new(head_dim as u32),
        );
    }

    // Synchronize and read output
    let output_bytes = client.read_one(out_handle);

    // Convert back to Candle tensor
    cubecl_to_candle_tensor(
        &output_bytes,
        &[batch, num_heads, seq_len, head_dim],
        q.device(),
    )
}

/// Validate attention input tensor shapes.
fn validate_attention_inputs(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<()> {
    let q_dims = q.dims();
    let k_dims = k.dims();
    let v_dims = v.dims();

    // Check 4D
    if q_dims.len() != 4 || k_dims.len() != 4 || v_dims.len() != 4 {
        return Err(UnslothError::InvalidConfig(format!(
            "Expected 4D tensors [batch, heads, seq, dim], got Q: {q_dims:?}, K: {k_dims:?}, V: {v_dims:?}"
        )));
    }

    // Check batch dimension matches
    if q_dims[0] != k_dims[0] || q_dims[0] != v_dims[0] {
        return Err(UnslothError::InvalidConfig(format!(
            "Batch size mismatch: Q={}, K={}, V={}",
            q_dims[0], k_dims[0], v_dims[0]
        )));
    }

    // Check head_dim matches
    if q_dims[3] != k_dims[3] || q_dims[3] != v_dims[3] {
        return Err(UnslothError::InvalidConfig(format!(
            "Head dimension mismatch: Q={}, K={}, V={}",
            q_dims[3], k_dims[3], v_dims[3]
        )));
    }

    // Check K and V have same shape (for now; GQA handled later)
    if k_dims != v_dims {
        return Err(UnslothError::InvalidConfig(format!(
            "K and V shape mismatch: K={k_dims:?}, V={v_dims:?}"
        )));
    }

    Ok(())
}

/// Fallback attention using Candle operations.
///
/// This provides correct results while the `CubeCL` kernel is being developed.
/// Uses O(N²) memory but serves as a reference for validation.
fn fallback_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    mask: Option<&Tensor>,
    config: &FlashAttentionConfig,
) -> Result<Tensor> {
    // Q @ K^T, scaled by 1/sqrt(head_dim)
    let scores = q.matmul(&k.transpose(2, 3)?.contiguous()?)?;
    let scores = (scores * scale)?;

    // Apply causal mask if configured
    let scores = if config.causal_mask {
        let seq_len = q.dims()[2];
        let causal_mask = create_causal_mask_tensor(seq_len, q.device())?;
        scores.broadcast_add(&causal_mask)?
    } else {
        scores
    };

    // Apply additional mask if provided
    let scores = match mask {
        Some(m) => scores.broadcast_add(m)?,
        None => scores,
    };

    // Softmax along key dimension
    let attn_weights = candle_nn::ops::softmax(&scores, 3)?;

    // Attention @ V
    let output = attn_weights.matmul(v)?;

    Ok(output)
}

/// Create a causal mask tensor with -inf in upper triangle.
fn create_causal_mask_tensor(seq_len: usize, device: &candle_core::Device) -> Result<Tensor> {
    let mut mask_data = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            if j > i {
                mask_data[i * seq_len + j] = f32::NEG_INFINITY;
            }
        }
    }

    let mask = Tensor::from_vec(mask_data, (1, 1, seq_len, seq_len), device)?;
    Ok(mask)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_validate_attention_inputs_valid() {
        let device = Device::Cpu;
        let q = Tensor::zeros((2, 4, 8, 64), candle_core::DType::F32, &device).unwrap();
        let k = Tensor::zeros((2, 4, 8, 64), candle_core::DType::F32, &device).unwrap();
        let v = Tensor::zeros((2, 4, 8, 64), candle_core::DType::F32, &device).unwrap();

        assert!(validate_attention_inputs(&q, &k, &v).is_ok());
    }

    #[test]
    fn test_validate_attention_inputs_wrong_dims() {
        let device = Device::Cpu;
        let q = Tensor::zeros((2, 8, 64), candle_core::DType::F32, &device).unwrap(); // 3D
        let k = Tensor::zeros((2, 4, 8, 64), candle_core::DType::F32, &device).unwrap();
        let v = Tensor::zeros((2, 4, 8, 64), candle_core::DType::F32, &device).unwrap();

        assert!(validate_attention_inputs(&q, &k, &v).is_err());
    }

    #[test]
    fn test_validate_attention_inputs_batch_mismatch() {
        let device = Device::Cpu;
        let q = Tensor::zeros((2, 4, 8, 64), candle_core::DType::F32, &device).unwrap();
        let k = Tensor::zeros((3, 4, 8, 64), candle_core::DType::F32, &device).unwrap(); // Different batch
        let v = Tensor::zeros((3, 4, 8, 64), candle_core::DType::F32, &device).unwrap();

        assert!(validate_attention_inputs(&q, &k, &v).is_err());
    }

    #[test]
    fn test_flash_attention_kernel_shape() {
        let device = Device::Cpu;
        let batch = 2;
        let num_heads = 4;
        let seq_len = 8;
        let head_dim = 64;

        let q = Tensor::randn(0.0f32, 1.0, (batch, num_heads, seq_len, head_dim), &device).unwrap();
        let k = Tensor::randn(0.0f32, 1.0, (batch, num_heads, seq_len, head_dim), &device).unwrap();
        let v = Tensor::randn(0.0f32, 1.0, (batch, num_heads, seq_len, head_dim), &device).unwrap();

        let scale = 1.0 / (head_dim as f64).sqrt();
        let config = FlashAttentionConfig::default();

        let output = flash_attention_kernel(&q, &k, &v, scale, None, &config).unwrap();

        assert_eq!(output.dims(), &[batch, num_heads, seq_len, head_dim]);
    }

    #[test]
    fn test_flash_attention_kernel_numerical_stability() {
        let device = Device::Cpu;
        // Use larger variance to stress numerical stability
        let q = Tensor::randn(0.0f32, 10.0, (1, 2, 4, 64), &device).unwrap();
        let k = Tensor::randn(0.0f32, 10.0, (1, 2, 4, 64), &device).unwrap();
        let v = Tensor::randn(0.0f32, 10.0, (1, 2, 4, 64), &device).unwrap();

        let scale = 1.0 / 8.0; // 1/sqrt(64)
        let config = FlashAttentionConfig::default();

        let output = flash_attention_kernel(&q, &k, &v, scale, None, &config).unwrap();

        // Verify no NaN or Inf
        let values: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();
        for val in values {
            assert!(!val.is_nan(), "Output contains NaN");
            assert!(!val.is_infinite(), "Output contains Inf");
        }
    }

    #[test]
    fn test_flash_attention_with_config() {
        let device = Device::Cpu;
        let q = Tensor::randn(0.0f32, 1.0, (1, 2, 16, 64), &device).unwrap();
        let k = Tensor::randn(0.0f32, 1.0, (1, 2, 16, 64), &device).unwrap();
        let v = Tensor::randn(0.0f32, 1.0, (1, 2, 16, 64), &device).unwrap();

        let scale = 1.0 / 8.0;
        let config = FlashAttentionConfig::for_rtx_5080().with_causal_mask();

        let output = flash_attention_kernel(&q, &k, &v, scale, None, &config).unwrap();

        assert_eq!(output.dims(), &[1, 2, 16, 64]);
    }

    // =========================================================================
    // Small-Size Numerical Equivalence Tests (Phase 1)
    // Test CubeCL/fallback output matches reference implementation
    // =========================================================================

    /// Reference attention implementation using explicit matmul for validation.
    fn reference_attention(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64) -> Tensor {
        let scores = q
            .matmul(&k.transpose(2, 3).unwrap().contiguous().unwrap())
            .unwrap();
        let scores = (scores * scale).unwrap();
        let attn_weights = candle_nn::ops::softmax(&scores, 3).unwrap();
        attn_weights.matmul(v).unwrap()
    }

    /// Compute mean absolute error between two tensors.
    fn mean_absolute_error(a: &Tensor, b: &Tensor) -> f32 {
        let diff = (a - b).unwrap().abs().unwrap();
        let mean = diff.mean_all().unwrap();
        mean.to_scalar::<f32>().unwrap()
    }

    #[test]
    fn test_numerical_equivalence_batch2_heads4_seq8_dim64() {
        let device = Device::Cpu;
        let (batch, heads, seq, dim) = (2, 4, 8, 64);
        let scale = 1.0 / (dim as f64).sqrt();

        let q = Tensor::randn(0.0f32, 1.0, (batch, heads, seq, dim), &device).unwrap();
        let k = Tensor::randn(0.0f32, 1.0, (batch, heads, seq, dim), &device).unwrap();
        let v = Tensor::randn(0.0f32, 1.0, (batch, heads, seq, dim), &device).unwrap();

        let config = FlashAttentionConfig::default();
        let output = flash_attention_kernel(&q, &k, &v, scale, None, &config).unwrap();
        let reference = reference_attention(&q, &k, &v, scale);

        let mae = mean_absolute_error(&output, &reference);
        assert!(
            mae < 1e-5,
            "MAE {} exceeds tolerance 1e-5 for batch={}, heads={}, seq={}, dim={}",
            mae,
            batch,
            heads,
            seq,
            dim
        );
    }

    #[test]
    fn test_numerical_equivalence_batch2_heads4_seq16_dim64() {
        let device = Device::Cpu;
        let (batch, heads, seq, dim) = (2, 4, 16, 64);
        let scale = 1.0 / (dim as f64).sqrt();

        let q = Tensor::randn(0.0f32, 1.0, (batch, heads, seq, dim), &device).unwrap();
        let k = Tensor::randn(0.0f32, 1.0, (batch, heads, seq, dim), &device).unwrap();
        let v = Tensor::randn(0.0f32, 1.0, (batch, heads, seq, dim), &device).unwrap();

        let config = FlashAttentionConfig::default();
        let output = flash_attention_kernel(&q, &k, &v, scale, None, &config).unwrap();
        let reference = reference_attention(&q, &k, &v, scale);

        let mae = mean_absolute_error(&output, &reference);
        assert!(mae < 1e-5, "MAE {} exceeds tolerance 1e-5", mae);
    }

    #[test]
    fn test_numerical_equivalence_batch2_heads4_seq32_dim64() {
        let device = Device::Cpu;
        let (batch, heads, seq, dim) = (2, 4, 32, 64);
        let scale = 1.0 / (dim as f64).sqrt();

        let q = Tensor::randn(0.0f32, 1.0, (batch, heads, seq, dim), &device).unwrap();
        let k = Tensor::randn(0.0f32, 1.0, (batch, heads, seq, dim), &device).unwrap();
        let v = Tensor::randn(0.0f32, 1.0, (batch, heads, seq, dim), &device).unwrap();

        let config = FlashAttentionConfig::default();
        let output = flash_attention_kernel(&q, &k, &v, scale, None, &config).unwrap();
        let reference = reference_attention(&q, &k, &v, scale);

        let mae = mean_absolute_error(&output, &reference);
        assert!(mae < 1e-5, "MAE {} exceeds tolerance 1e-5", mae);
    }

    #[test]
    fn test_numerical_equivalence_batch2_heads4_seq64_dim64() {
        let device = Device::Cpu;
        let (batch, heads, seq, dim) = (2, 4, 64, 64);
        let scale = 1.0 / (dim as f64).sqrt();

        let q = Tensor::randn(0.0f32, 1.0, (batch, heads, seq, dim), &device).unwrap();
        let k = Tensor::randn(0.0f32, 1.0, (batch, heads, seq, dim), &device).unwrap();
        let v = Tensor::randn(0.0f32, 1.0, (batch, heads, seq, dim), &device).unwrap();

        let config = FlashAttentionConfig::default();
        let output = flash_attention_kernel(&q, &k, &v, scale, None, &config).unwrap();
        let reference = reference_attention(&q, &k, &v, scale);

        let mae = mean_absolute_error(&output, &reference);
        assert!(mae < 1e-5, "MAE {} exceeds tolerance 1e-5", mae);
    }

    #[test]
    fn test_numerical_equivalence_batch1_heads1_seq8_dim64() {
        // Minimal case: single batch, single head
        let device = Device::Cpu;
        let (batch, heads, seq, dim) = (1, 1, 8, 64);
        let scale = 1.0 / (dim as f64).sqrt();

        let q = Tensor::randn(0.0f32, 1.0, (batch, heads, seq, dim), &device).unwrap();
        let k = Tensor::randn(0.0f32, 1.0, (batch, heads, seq, dim), &device).unwrap();
        let v = Tensor::randn(0.0f32, 1.0, (batch, heads, seq, dim), &device).unwrap();

        let config = FlashAttentionConfig::default();
        let output = flash_attention_kernel(&q, &k, &v, scale, None, &config).unwrap();
        let reference = reference_attention(&q, &k, &v, scale);

        let mae = mean_absolute_error(&output, &reference);
        assert!(mae < 1e-5, "MAE {} exceeds tolerance 1e-5", mae);
    }

    #[test]
    fn test_numerical_equivalence_with_different_configs() {
        let device = Device::Cpu;
        let (batch, heads, seq, dim) = (2, 4, 16, 64);
        let scale = 1.0 / (dim as f64).sqrt();

        let q = Tensor::randn(0.0f32, 1.0, (batch, heads, seq, dim), &device).unwrap();
        let k = Tensor::randn(0.0f32, 1.0, (batch, heads, seq, dim), &device).unwrap();
        let v = Tensor::randn(0.0f32, 1.0, (batch, heads, seq, dim), &device).unwrap();

        // Test with different tile sizes
        let configs = [
            FlashAttentionConfig::default(),
            FlashAttentionConfig::for_rtx_5080(),
            FlashAttentionConfig::for_rtx_3090_ti(),
        ];

        let reference = reference_attention(&q, &k, &v, scale);

        for config in configs {
            let output = flash_attention_kernel(&q, &k, &v, scale, None, &config).unwrap();
            let mae = mean_absolute_error(&output, &reference);
            assert!(
                mae < 1e-5,
                "MAE {} exceeds tolerance for config {:?}",
                mae,
                config
            );
        }
    }

    #[test]
    fn test_determinism_multiple_runs() {
        // Ensure same input produces same output across runs
        let device = Device::Cpu;
        let (batch, heads, seq, dim) = (2, 4, 8, 64);
        let scale = 1.0 / (dim as f64).sqrt();

        // Fixed seed via from_slice
        let data: Vec<f32> = (0..(batch * heads * seq * dim))
            .map(|i| (i as f32 * 0.001).sin())
            .collect();
        let q = Tensor::from_vec(data.clone(), (batch, heads, seq, dim), &device).unwrap();
        let k = Tensor::from_vec(data.clone(), (batch, heads, seq, dim), &device).unwrap();
        let v = Tensor::from_vec(data, (batch, heads, seq, dim), &device).unwrap();

        let config = FlashAttentionConfig::default();

        let output1 = flash_attention_kernel(&q, &k, &v, scale, None, &config).unwrap();
        let output2 = flash_attention_kernel(&q, &k, &v, scale, None, &config).unwrap();

        let mae = mean_absolute_error(&output1, &output2);
        assert!(mae < 1e-10, "Non-deterministic output: MAE = {}", mae);
    }

    #[test]
    fn test_identity_attention_pattern() {
        // When Q == K, diagonal of attention matrix should be highest
        let device = Device::Cpu;
        let (batch, heads, seq, dim) = (1, 1, 4, 64);
        let scale = 1.0 / (dim as f64).sqrt();

        // Create orthogonal-ish rows (each row very different)
        let mut data = vec![0.0f32; seq * dim];
        for i in 0..seq {
            for j in 0..dim {
                data[i * dim + j] = if j == i { 1.0 } else { 0.0 };
            }
        }
        let q = Tensor::from_vec(data.clone(), (batch, heads, seq, dim), &device).unwrap();
        let k = q.clone();
        // V is identity-like to make output predictable
        let v = q.clone();

        let config = FlashAttentionConfig::default();
        let output = flash_attention_kernel(&q, &k, &v, scale, None, &config).unwrap();

        // Output should be close to V when Q==K (since attention to self is high)
        let mae = mean_absolute_error(&output, &v);
        // With softmax and scaling, we expect some deviation but should be reasonable
        assert!(mae < 0.5, "Identity pattern MAE {} too high", mae);
    }

    // =========================================================================
    // Causal Masking Tests
    // =========================================================================

    #[test]
    fn test_causal_masking_basic() {
        // Test that causal masking prevents attending to future positions
        let device = Device::Cpu;
        let (batch, heads, seq, dim) = (1, 1, 4, 64);
        let scale = 1.0 / (dim as f64).sqrt();

        let q = Tensor::randn(0.0f32, 1.0, (batch, heads, seq, dim), &device).unwrap();
        let k = Tensor::randn(0.0f32, 1.0, (batch, heads, seq, dim), &device).unwrap();
        let v = Tensor::randn(0.0f32, 1.0, (batch, heads, seq, dim), &device).unwrap();

        let config = FlashAttentionConfig::default().with_causal_mask();
        let output = flash_attention_kernel(&q, &k, &v, scale, None, &config).unwrap();

        // Verify output shape
        assert_eq!(output.dims(), &[batch, heads, seq, dim]);

        // Verify no NaN/Inf
        let values: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();
        for val in values {
            assert!(!val.is_nan() && !val.is_infinite());
        }
    }

    #[test]
    fn test_causal_vs_non_causal_difference() {
        // Causal and non-causal should produce different outputs
        let device = Device::Cpu;
        let (batch, heads, seq, dim) = (1, 2, 8, 64);
        let scale = 1.0 / (dim as f64).sqrt();

        let q = Tensor::randn(0.0f32, 1.0, (batch, heads, seq, dim), &device).unwrap();
        let k = Tensor::randn(0.0f32, 1.0, (batch, heads, seq, dim), &device).unwrap();
        let v = Tensor::randn(0.0f32, 1.0, (batch, heads, seq, dim), &device).unwrap();

        let config_normal = FlashAttentionConfig::default();
        let config_causal = FlashAttentionConfig::default().with_causal_mask();

        let output_normal =
            flash_attention_kernel(&q, &k, &v, scale, None, &config_normal).unwrap();
        let output_causal =
            flash_attention_kernel(&q, &k, &v, scale, None, &config_causal).unwrap();

        // Outputs should differ (causal mask changes attention)
        let mae = mean_absolute_error(&output_normal, &output_causal);
        assert!(mae > 1e-4, "Causal and non-causal outputs are too similar");
    }

    #[test]
    fn test_causal_masking_numerical_equivalence() {
        // Test causal masking against reference implementation
        let device = Device::Cpu;
        let (batch, heads, seq, dim) = (2, 4, 16, 64);
        let scale = 1.0 / (dim as f64).sqrt();

        let q = Tensor::randn(0.0f32, 1.0, (batch, heads, seq, dim), &device).unwrap();
        let k = Tensor::randn(0.0f32, 1.0, (batch, heads, seq, dim), &device).unwrap();
        let v = Tensor::randn(0.0f32, 1.0, (batch, heads, seq, dim), &device).unwrap();

        // Create causal mask manually
        let causal_mask = create_causal_mask(seq, &device);

        let config = FlashAttentionConfig::default().with_causal_mask();
        let output = flash_attention_kernel(&q, &k, &v, scale, None, &config).unwrap();

        // Reference with explicit mask
        let reference = reference_attention_with_mask(&q, &k, &v, scale, Some(&causal_mask));

        let mae = mean_absolute_error(&output, &reference);
        assert!(mae < 1e-5, "Causal MAE {} exceeds tolerance 1e-5", mae);
    }

    /// Create causal mask (upper triangular with -inf)
    fn create_causal_mask(seq_len: usize, device: &Device) -> Tensor {
        let mut mask_data = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                if j > i {
                    mask_data[i * seq_len + j] = f32::NEG_INFINITY;
                }
            }
        }
        Tensor::from_vec(mask_data, (1, 1, seq_len, seq_len), device)
            .unwrap()
            .broadcast_as((1, 1, seq_len, seq_len))
            .unwrap()
    }

    /// Reference attention with explicit mask
    fn reference_attention_with_mask(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        scale: f64,
        mask: Option<&Tensor>,
    ) -> Tensor {
        let scores = q
            .matmul(&k.transpose(2, 3).unwrap().contiguous().unwrap())
            .unwrap();
        let scores = (scores * scale).unwrap();

        let scores = if let Some(m) = mask {
            scores.broadcast_add(m).unwrap()
        } else {
            scores
        };

        let attn_weights = candle_nn::ops::softmax(&scores, 3).unwrap();
        attn_weights.matmul(v).unwrap()
    }
}
