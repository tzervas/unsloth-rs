// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! `CubeCL` GPU kernel implementation for Flash Attention.
//!
//! This module provides a memory-efficient GPU implementation of multi-head attention
//! using the Flash Attention algorithm. The implementation uses `CubeCL` for cross-platform
//! GPU support (CUDA, `ROCm`, Vulkan).
//!
//! ## Flash Attention Algorithm
//!
//! Traditional attention materializes the full attention matrix (seq × seq), which is
//! quadratic in memory. Flash Attention computes attention in tiles with online softmax,
//! achieving linear memory complexity.
//!
//! ### Memory Complexity
//! - Traditional: O(batch × heads × seq²)
//! - Flash Attention: O(batch × heads × seq × dim)
//!
//! ### Key Optimizations
//! 1. **Tiling:** Process attention in blocks that fit in shared memory
//! 2. **Online Softmax:** Compute softmax incrementally without full materialization
//! 3. **Fused Operations:** Combine Q·K^T, softmax, and attention·V in single pass
//! 4. **IO-Aware:** Minimize slow HBM access, maximize fast SRAM usage
//!
//! ## Implementation Status
//!
//! Production-ready CubeCL kernel with:
//! - Arbitrary head dimensions (not just power-of-2)
//! - Causal masking support
//! - Proper bounds checking and thread synchronization

use crate::error::Result;
use candle_core::Tensor;

/// Check if `CubeCL` GPU support is available.
///
/// Returns true if:
/// 1. The `cuda` feature is enabled at compile time
/// 2. A CUDA-capable GPU is available at runtime
#[must_use]
pub fn has_cubecl_support() -> bool {
    #[cfg(feature = "cuda")]
    {
        use candle_core::Device;
        // Check if CUDA device is available via Candle
        matches!(Device::cuda_if_available(0), Ok(Device::Cuda(_)))
    }

    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

/// Flash Attention computation using `CubeCL` GPU kernel.
///
/// This function implements the Flash Attention algorithm using tiled computation
/// and online softmax for memory efficiency. It automatically dispatches to
/// the CubeCL kernel when CUDA is available, otherwise falls back to Candle ops.
///
/// # Arguments
/// * `q` - Query tensor [batch, `num_heads`, `seq_len`, `head_dim`]
/// * `k` - Key tensor [batch, `num_kv_heads`, `seq_len`, `head_dim`]
/// * `v` - Value tensor [batch, `num_kv_heads`, `seq_len`, `head_dim`]
/// * `scale` - Attention scale factor (typically `1/sqrt(head_dim)`)
/// * `mask` - Optional attention mask
///
/// # Returns
/// Attention output tensor [batch, `num_heads`, `seq_len`, `head_dim`]
///
/// # Errors
/// Returns error if:
/// - Tensor shapes are incompatible
/// - GPU kernel launch fails
/// - Memory allocation fails
///
/// # Example
/// ```rust,ignore
/// use unsloth_rs::kernels::flash_attention_cubecl;
///
/// let output = flash_attention_cubecl(&q, &k, &v, scale, None)?;
/// ```
pub fn flash_attention_cubecl(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    mask: Option<&Tensor>,
) -> Result<Tensor> {
    // Validate inputs
    let q_shape = q.dims();
    let k_shape = k.dims();
    let v_shape = v.dims();

    if q_shape.len() != 4 || k_shape.len() != 4 || v_shape.len() != 4 {
        return Err(crate::error::UnslothError::InvalidConfig(format!(
            "Expected 4D tensors, got Q: {q_shape:?}, K: {k_shape:?}, V: {v_shape:?}"
        )));
    }

    let (batch, num_heads, seq_len, head_dim) = (q_shape[0], q_shape[1], q_shape[2], q_shape[3]);
    let num_kv_heads = k_shape[1];

    tracing::debug!(
        "Flash Attention CubeCL: batch={}, heads={}/{}, seq={}, dim={}",
        batch,
        num_heads,
        num_kv_heads,
        seq_len,
        head_dim
    );

    // Use CubeCL kernel when available and on CUDA device
    #[cfg(feature = "cuda")]
    {
        if q.device().is_cuda() && has_cubecl_support() {
            // Use the kernel from cubecl module
            use super::cubecl::{flash_attention_kernel, FlashAttentionConfig};

            let config = FlashAttentionConfig::default().with_head_dim(head_dim as u32);
            return flash_attention_kernel(q, k, v, scale, mask, &config);
        }
    }

    // Fallback for CPU or when CubeCL is not available
    flash_attention_fallback(q, k, v, scale, mask)
}

/// Fallback implementation using Candle operations.
fn flash_attention_fallback(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    mask: Option<&Tensor>,
) -> Result<Tensor> {
    let scores = q.matmul(&k.transpose(2, 3)?.contiguous()?)?;
    let scores = (scores * scale)?;

    let scores = match mask {
        Some(m) => scores.broadcast_add(m)?,
        None => scores,
    };

    let attn_weights = candle_nn::ops::softmax(&scores, 3)?;
    let output = attn_weights.matmul(v)?;

    Ok(output)
}

/// Estimate VRAM usage for Flash Attention.
#[must_use]
pub fn estimate_flash_attention_vram(
    batch_size: usize,
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
    tile_size: usize,
) -> usize {
    let bytes_per_elem = 4;
    let qkv_size = 3 * batch_size * num_heads * seq_len * head_dim * bytes_per_elem;
    let output_size = batch_size * num_heads * seq_len * head_dim * bytes_per_elem;
    let workspace = batch_size * num_heads * (2 * tile_size) * bytes_per_elem;
    qkv_size + output_size + workspace
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_has_cubecl_support() {
        let _ = has_cubecl_support();
    }

    #[test]
    fn test_flash_attention_shape() {
        let device = Device::Cpu;
        let (batch, num_heads, seq_len, head_dim) = (2, 4, 8, 64);

        let q = Tensor::randn(0.0f32, 1.0, (batch, num_heads, seq_len, head_dim), &device).unwrap();
        let k = Tensor::randn(0.0f32, 1.0, (batch, num_heads, seq_len, head_dim), &device).unwrap();
        let v = Tensor::randn(0.0f32, 1.0, (batch, num_heads, seq_len, head_dim), &device).unwrap();

        let scale = 1.0 / (head_dim as f64).sqrt();
        let output = flash_attention_cubecl(&q, &k, &v, scale, None).unwrap();

        assert_eq!(output.dims(), &[batch, num_heads, seq_len, head_dim]);
    }

    #[test]
    fn test_flash_attention_numerical_stability() {
        let device = Device::Cpu;
        let q = Tensor::randn(0.0f32, 10.0, (1, 2, 4, 64), &device).unwrap();
        let k = Tensor::randn(0.0f32, 10.0, (1, 2, 4, 64), &device).unwrap();
        let v = Tensor::randn(0.0f32, 10.0, (1, 2, 4, 64), &device).unwrap();

        let scale = 1.0 / 8.0;
        let output = flash_attention_cubecl(&q, &k, &v, scale, None).unwrap();

        let values: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();
        for v in values {
            assert!(!v.is_nan() && !v.is_infinite());
        }
    }

    #[test]
    fn test_flash_attention_invalid_shape() {
        let device = Device::Cpu;
        let q = Tensor::randn(0.0f32, 1.0, (2, 8, 64), &device).unwrap();
        let k = Tensor::randn(0.0f32, 1.0, (2, 8, 64), &device).unwrap();
        let v = Tensor::randn(0.0f32, 1.0, (2, 8, 64), &device).unwrap();

        let result = flash_attention_cubecl(&q, &k, &v, 1.0, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_estimate_flash_attention_vram() {
        let vram = estimate_flash_attention_vram(4, 12, 2048, 64, 128);
        assert!(vram > 1_000_000);
        assert!(vram < 10_000_000_000);

        let vram_2x_batch = estimate_flash_attention_vram(8, 12, 2048, 64, 128);
        assert!(vram_2x_batch > vram);
    }
}
