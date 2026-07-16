// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! Multi-head attention implementation.
//!
//! This module provides a multi-head attention layer commonly used in transformer
//! architectures. The implementation supports grouped-query attention (GQA) which
//! reduces memory usage by sharing key-value heads across multiple query heads.
//!
//! ## Why Multi-Head Attention?
//!
//! Multi-head attention allows the model to jointly attend to information from
//! different representation subspaces at different positions. This is more
//! effective than single-head attention with the same total dimension.
//!
//! ## Implementation Notes
//!
//! - Uses Candle's tensor operations for both CPU and GPU execution
//! - GPU dispatch uses Candle's CUDA backend (not custom fused kernels yet)
//! - Supports optional attention masking for causal/padded sequences
//! - KV cache parameter is a placeholder for future inference optimization

use candle_core::{Device, Tensor};

use crate::error::Result;

fn warn_cpu_fallback(device: &Device) {
    static WARN_ONCE: std::sync::Once = std::sync::Once::new();
    if matches!(device, Device::Cpu) {
        WARN_ONCE.call_once(|| {
            eprintln!(
                "unsloth-rs: CPU device in use. CUDA is the intended default; enable the 'cuda' feature and use Device::cuda_if_available(0) when possible."
            );
        });
    }
}

/// Configuration for fused attention.
#[derive(Debug, Clone)]
pub struct FusedAttentionConfig {
    /// Hidden dimension.
    pub hidden_size: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Dimension per head.
    pub head_dim: usize,
    /// Number of key-value heads (for GQA).
    pub num_kv_heads: Option<usize>,
    /// Dropout probability.
    pub dropout: f64,
    /// Whether to use flash attention algorithm.
    pub use_flash: bool,
}

impl Default for FusedAttentionConfig {
    fn default() -> Self {
        Self {
            hidden_size: 768,
            num_heads: 12,
            head_dim: 64,
            num_kv_heads: None,
            dropout: 0.0,
            use_flash: true,
        }
    }
}

/// Multi-head attention layer.
///
/// This implementation provides:
/// 1. QKV projection
/// 2. Scaled dot-product attention
/// 3. Output projection
/// 4. Support for grouped-query attention (GQA)
pub struct FusedAttention {
    /// QKV projection weights [3 * hidden, hidden]
    qkv_weight: Tensor,
    /// Output projection weights [hidden, hidden]
    o_weight: Tensor,
    /// Configuration
    config: FusedAttentionConfig,
}

impl FusedAttention {
    /// Create a new fused attention layer.
    ///
    /// # Arguments
    /// * `config` - Attention configuration
    /// * `device` - Device for tensors
    ///
    /// # Errors
    /// Returns error if tensor creation fails
    pub fn new(config: FusedAttentionConfig, device: &Device) -> Result<Self> {
        warn_cpu_fallback(device);
        let hidden = config.hidden_size;
        let num_kv_heads = config.num_kv_heads.unwrap_or(config.num_heads);

        // QKV: Q has num_heads, K and V have num_kv_heads
        let qkv_size = config.num_heads * config.head_dim + 2 * num_kv_heads * config.head_dim;

        let qkv_weight = Tensor::randn(
            0.0f32,
            (1.0 / (hidden as f64).sqrt()) as f32,
            (qkv_size, hidden),
            device,
        )?;

        let o_weight = Tensor::randn(
            0.0f32,
            (1.0 / (hidden as f64).sqrt()) as f32,
            (hidden, hidden),
            device,
        )?;

        Ok(Self {
            qkv_weight,
            o_weight,
            config,
        })
    }

    /// Forward pass with optional KV cache.
    ///
    /// # Arguments
    /// * `hidden_states` - Input tensor [batch, `seq_len`, hidden]
    /// * `attention_mask` - Optional attention mask
    /// * `kv_cache` - Optional KV cache for inference
    ///
    /// # Returns
    /// Output tensor [batch, `seq_len`, hidden]
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        _kv_cache: Option<(&Tensor, &Tensor)>,
    ) -> Result<Tensor> {
        let device = hidden_states.device();

        // Use optimized GPU path if available
        if device.is_cuda() {
            self.forward_cuda(hidden_states, attention_mask)
        } else {
            self.forward_cpu(hidden_states, attention_mask)
        }
    }

    /// CPU reference implementation.
    fn forward_cpu(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (batch, seq_len, _hidden) = hidden_states.dims3()?;
        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim;

        // QKV projection - use broadcast_matmul for 3D tensor with 2D weight
        let qkv = hidden_states.broadcast_matmul(&self.qkv_weight.t()?)?;

        // Split into Q, K, V
        let q_size = num_heads * head_dim;
        let kv_size = self.config.num_kv_heads.unwrap_or(num_heads) * head_dim;

        let q = qkv.narrow(2, 0, q_size)?;
        let k = qkv.narrow(2, q_size, kv_size)?;
        let v = qkv.narrow(2, q_size + kv_size, kv_size)?;

        // Reshape for attention: [batch, num_heads, seq_len, head_dim]
        // Make contiguous after transpose for matmul compatibility
        let q = q
            .reshape((batch, seq_len, num_heads, head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let mut k = k
            .reshape((
                batch,
                seq_len,
                self.config.num_kv_heads.unwrap_or(num_heads),
                head_dim,
            ))?
            .transpose(1, 2)?
            .contiguous()?;
        let mut v = v
            .reshape((
                batch,
                seq_len,
                self.config.num_kv_heads.unwrap_or(num_heads),
                head_dim,
            ))?
            .transpose(1, 2)?
            .contiguous()?;

        // For GQA: repeat K and V heads to match Q
        let num_kv_heads = self.config.num_kv_heads.unwrap_or(num_heads);
        if num_kv_heads < num_heads {
            let repeat_factor = num_heads / num_kv_heads;
            // k shape: [batch, num_kv_heads, seq_len, head_dim]
            // Expand to: [batch, num_kv_heads, repeat_factor, seq_len, head_dim]
            // Then reshape to: [batch, num_heads, seq_len, head_dim]
            let k_shape = k.shape().dims();
            k = k
                .unsqueeze(2)? // [batch, num_kv_heads, 1, seq_len, head_dim]
                .expand((
                    k_shape[0],
                    k_shape[1],
                    repeat_factor,
                    k_shape[2],
                    k_shape[3],
                ))?
                .reshape((k_shape[0], num_heads, k_shape[2], k_shape[3]))?
                .contiguous()?;

            let v_shape = v.shape().dims();
            v = v
                .unsqueeze(2)?
                .expand((
                    v_shape[0],
                    v_shape[1],
                    repeat_factor,
                    v_shape[2],
                    v_shape[3],
                ))?
                .reshape((v_shape[0], num_heads, v_shape[2], v_shape[3]))?
                .contiguous()?;
        }

        // Scaled dot-product attention
        let scale = (head_dim as f64).sqrt();
        let scores = q.matmul(&k.transpose(2, 3)?.contiguous()?)?;
        let scores = (scores / scale)?;

        // Apply mask if provided
        let scores = match attention_mask {
            Some(mask) => scores.broadcast_add(mask)?,
            None => scores,
        };

        // Softmax
        let attn_weights = candle_nn::ops::softmax(&scores, 3)?;

        // Attention output
        let attn_output = attn_weights.matmul(&v)?;

        // Reshape back: [batch, seq_len, hidden]
        let attn_output = attn_output.transpose(1, 2)?.contiguous()?.reshape((
            batch,
            seq_len,
            num_heads * head_dim,
        ))?;

        // Output projection - use broadcast_matmul for 3D tensor with 2D weight
        let output = attn_output.broadcast_matmul(&self.o_weight.t()?)?;

        Ok(output)
    }

    /// CUDA implementation.
    ///
    /// Uses `CubeCL` fused Flash Attention kernel when available, otherwise
    /// falls back to Candle's CUDA backend.
    fn forward_cuda(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        tracing::debug!(
            "Using CUDA attention path for input shape {:?}",
            hidden_states.shape()
        );

        // Try CubeCL Flash Attention if enabled and available
        if self.config.use_flash && crate::kernels::attention_cubecl::has_cubecl_support() {
            return self.forward_flash_attention(hidden_states, attention_mask);
        }

        // Fallback to Candle's CUDA backend
        self.forward_cpu(hidden_states, attention_mask)
    }

    /// Flash Attention implementation using `CubeCL`.
    fn forward_flash_attention(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (batch, seq_len, _hidden) = hidden_states.dims3()?;
        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim;
        let num_kv_heads = self.config.num_kv_heads.unwrap_or(num_heads);

        // QKV projection
        let qkv = hidden_states.broadcast_matmul(&self.qkv_weight.t()?)?;

        // Split into Q, K, V
        let q_size = num_heads * head_dim;
        let kv_size = num_kv_heads * head_dim;

        let q = qkv.narrow(2, 0, q_size)?;
        let k = qkv.narrow(2, q_size, kv_size)?;
        let v = qkv.narrow(2, q_size + kv_size, kv_size)?;

        // Reshape for attention: [batch, num_heads, seq_len, head_dim]
        let q = q
            .reshape((batch, seq_len, num_heads, head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((batch, seq_len, num_kv_heads, head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((batch, seq_len, num_kv_heads, head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Scale factor
        let scale = (head_dim as f64).sqrt();

        // Call Flash Attention CubeCL kernel
        let attn_output = crate::kernels::attention_cubecl::flash_attention_cubecl(
            &q,
            &k,
            &v,
            scale,
            attention_mask,
        )?;

        // Reshape back: [batch, seq_len, hidden]
        let attn_output = attn_output.transpose(1, 2)?.contiguous()?.reshape((
            batch,
            seq_len,
            num_heads * head_dim,
        ))?;

        // Output projection
        let output = attn_output.broadcast_matmul(&self.o_weight.t()?)?;

        Ok(output)
    }

    /// Estimate VRAM usage in bytes.
    #[must_use]
    pub fn vram_estimate(&self, batch_size: usize, seq_len: usize) -> usize {
        let hidden = self.config.hidden_size;
        let bytes_per_elem = 4; // f32

        // QKV projection output
        let qkv_size = batch_size * seq_len * 3 * hidden * bytes_per_elem;
        // Attention scores
        let scores_size = batch_size * self.config.num_heads * seq_len * seq_len * bytes_per_elem;
        // Output
        let output_size = batch_size * seq_len * hidden * bytes_per_elem;

        qkv_size + scores_size + output_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    #[test]
    fn test_attention_creation() {
        let config = FusedAttentionConfig::default();
        let device = Device::Cpu;
        let attn = FusedAttention::new(config, &device);
        assert!(attn.is_ok());
    }

    #[test]
    fn test_attention_forward_shape() {
        let config = FusedAttentionConfig {
            hidden_size: 768,
            num_heads: 12,
            head_dim: 64,
            ..Default::default()
        };
        let device = Device::Cpu;
        let attn = FusedAttention::new(config, &device).unwrap();

        let input = Tensor::zeros(&[2, 10, 768], DType::F32, &device).unwrap();
        let output = attn.forward(&input, None, None).unwrap();

        assert_eq!(output.shape().dims(), &[2, 10, 768]);
    }

    #[test]
    fn test_attention_with_random_input() {
        let config = FusedAttentionConfig {
            hidden_size: 256,
            num_heads: 4,
            head_dim: 64,
            ..Default::default()
        };
        let device = Device::Cpu;
        let attn = FusedAttention::new(config, &device).unwrap();

        // Random input
        let input = Tensor::randn(0.0f32, 1.0, (1, 8, 256), &device).unwrap();
        let output = attn.forward(&input, None, None);

        assert!(output.is_ok());
        let output = output.unwrap();
        assert_eq!(output.shape().dims(), &[1, 8, 256]);

        // Output should not have NaN values
        let sum = output.sum_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(!sum.is_nan(), "Output contains NaN values");
    }

    #[test]
    fn test_attention_numerical_stability() {
        let config = FusedAttentionConfig {
            hidden_size: 128,
            num_heads: 2,
            head_dim: 64,
            ..Default::default()
        };
        let device = Device::Cpu;
        let attn = FusedAttention::new(config, &device).unwrap();

        // Test with larger values that could cause overflow
        let input = Tensor::randn(0.0f32, 10.0, (1, 4, 128), &device).unwrap();
        let output = attn.forward(&input, None, None);

        assert!(output.is_ok());
        let output = output.unwrap();

        // Check for NaN and Inf
        let values: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();
        for v in values {
            assert!(!v.is_nan(), "Output contains NaN");
            assert!(!v.is_infinite(), "Output contains Inf");
        }
    }

    #[test]
    fn test_attention_vram_estimate() {
        let config = FusedAttentionConfig {
            hidden_size: 4096,
            num_heads: 32,
            head_dim: 128,
            ..Default::default()
        };
        let device = Device::Cpu;
        let attn = FusedAttention::new(config, &device).unwrap();

        let vram = attn.vram_estimate(4, 2048);

        // Should be substantial (several GB for this config)
        assert!(vram > 100 * 1024 * 1024); // > 100 MB
        assert!(vram < 100 * 1024 * 1024 * 1024); // < 100 GB (sanity check)
    }
}
