// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! Rotary Position Embedding (`RoPE`) implementation.
//!
//! `RoPE` encodes position information directly into the query and key vectors
//! through rotation, enabling the model to learn relative position relationships.
//!
//! ## Why `RoPE`?
//!
//! Unlike absolute position embeddings, `RoPE`:
//! - Naturally encodes relative positions through rotation
//! - Scales well to longer sequences than seen during training
//! - Is used by modern LLMs like `LLaMA`, Mistral, and others
//!
//! ## Implementation Notes
//!
//! - Pre-computes cos/sin caches up to `max_seq_len` for efficiency
//! - Applies rotation in pairs: splits `head_dim` in half and rotates each pair
//! - Uses standard rotation formula: [x1*cos - x2*sin, x2*cos + x1*sin]

use candle_core::{Device, Tensor};

use crate::error::Result;

/// Rotary position embedding.
///
/// Applies rotary embeddings to query and key tensors for position encoding.
pub struct RotaryEmbedding {
    /// Cosine cache [`max_seq_len`, `head_dim/2`]
    cos_cache: Tensor,
    /// Sine cache [`max_seq_len`, `head_dim/2`]  
    sin_cache: Tensor,
    /// Head dimension
    head_dim: usize,
}

impl RotaryEmbedding {
    /// Create rotary embeddings.
    ///
    /// # Arguments
    /// * `head_dim` - Dimension per attention head
    /// * `max_seq_len` - Maximum sequence length to cache
    /// * `base` - Base for frequency computation (typically 10000)
    /// * `device` - Device for tensors
    pub fn new(head_dim: usize, max_seq_len: usize, base: f32, device: &Device) -> Result<Self> {
        // Compute inverse frequencies
        let inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| 1.0 / base.powf(i as f32 / head_dim as f32))
            .collect();

        let inv_freq = Tensor::from_vec(inv_freq, (head_dim / 2,), device)?;

        // Compute position indices
        let positions: Vec<f32> = (0..max_seq_len).map(|i| i as f32).collect();
        let positions = Tensor::from_vec(positions, (max_seq_len, 1), device)?;

        // Compute frequencies: [max_seq_len, head_dim/2]
        let freqs = positions.matmul(&inv_freq.unsqueeze(0)?)?;

        // Compute cos and sin caches
        let cos_cache = freqs.cos()?;
        let sin_cache = freqs.sin()?;

        Ok(Self {
            cos_cache,
            sin_cache,
            head_dim,
        })
    }

    /// Apply rotary embedding to query and key tensors.
    ///
    /// # Arguments
    /// * `q` - Query tensor [batch, `num_heads`, `seq_len`, `head_dim`]
    /// * `k` - Key tensor [batch, `num_kv_heads`, `seq_len`, `head_dim`]
    /// * `position_ids` - Position indices [batch, `seq_len`]
    ///
    /// # Returns
    /// Tuple of (`rotated_q`, `rotated_k`)
    pub fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        _position_ids: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let device = q.device();

        if device.is_cuda() {
            self.forward_cuda(q, k)
        } else {
            self.forward_cpu(q, k)
        }
    }

    /// CPU reference implementation for `RoPE`.
    fn forward_cpu(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        let seq_len = q.dim(2)?;

        // Get cos/sin for positions
        let cos = self.cos_cache.narrow(0, 0, seq_len)?;
        let sin = self.sin_cache.narrow(0, 0, seq_len)?;

        let q_rotated = self.apply_rotary(q, &cos, &sin)?;
        let k_rotated = self.apply_rotary(k, &cos, &sin)?;

        Ok((q_rotated, k_rotated))
    }

    /// CUDA implementation.
    ///
    /// Uses Candle's CUDA backend for GPU acceleration.
    /// The algorithm is the same as the CPU implementation.
    fn forward_cuda(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        tracing::debug!("Using CUDA RoPE path for Q shape {:?}", q.shape());
        self.forward_cpu(q, k)
    }

    fn apply_rotary(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let half_dim = self.head_dim / 2;

        // Split into two halves
        let x1 = x.narrow(3, 0, half_dim)?;
        let x2 = x.narrow(3, half_dim, half_dim)?;

        // Apply rotation: [x1, x2] -> [x1*cos - x2*sin, x2*cos + x1*sin]
        let rotated_x1 = (x1.broadcast_mul(cos)? - x2.broadcast_mul(sin)?)?;
        let rotated_x2 = (x2.broadcast_mul(cos)? + x1.broadcast_mul(sin)?)?;

        // Concatenate
        Tensor::cat(&[&rotated_x1, &rotated_x2], 3).map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    #[test]
    fn test_rope_creation() {
        let device = Device::Cpu;
        let rope = RotaryEmbedding::new(64, 2048, 10000.0, &device);
        assert!(rope.is_ok());
    }

    #[test]
    fn test_rope_preserves_shape() {
        let device = Device::Cpu;
        let rope = RotaryEmbedding::new(64, 2048, 10000.0, &device).unwrap();

        let q = Tensor::zeros(&[1, 12, 10, 64], DType::F32, &device).unwrap();
        let k = Tensor::zeros(&[1, 12, 10, 64], DType::F32, &device).unwrap();
        let pos = Tensor::zeros(&[1, 10], DType::I64, &device).unwrap();

        let (q_rot, k_rot) = rope.forward(&q, &k, &pos).unwrap();

        assert_eq!(q_rot.shape().dims(), &[1, 12, 10, 64]);
        assert_eq!(k_rot.shape().dims(), &[1, 12, 10, 64]);
    }
}
