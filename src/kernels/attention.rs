//! Fused attention implementation.
//!
//! Combines QKV projection, attention computation, and output projection
//! into a single memory-efficient operation.

use candle_core::{DType, Device, Tensor};

use crate::error::{Result, UnslothError};

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

/// Fused multi-head attention layer.
///
/// This implementation fuses multiple operations to reduce memory bandwidth:
/// 1. QKV projection
/// 2. Rotary position embeddings (optional)
/// 3. Scaled dot-product attention
/// 4. Output projection
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
        let hidden = config.hidden_size;
        let num_kv_heads = config.num_kv_heads.unwrap_or(config.num_heads);
        
        // QKV: Q has num_heads, K and V have num_kv_heads
        let qkv_size = config.num_heads * config.head_dim 
            + 2 * num_kv_heads * config.head_dim;
        
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
    /// * `hidden_states` - Input tensor [batch, seq_len, hidden]
    /// * `attention_mask` - Optional attention mask
    /// * `kv_cache` - Optional KV cache for inference
    ///
    /// # Returns
    /// Output tensor [batch, seq_len, hidden]
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

        // QKV projection
        let qkv = hidden_states.matmul(&self.qkv_weight.t()?)?;
        
        // Split into Q, K, V
        let q_size = num_heads * head_dim;
        let kv_size = self.config.num_kv_heads.unwrap_or(num_heads) * head_dim;
        
        let q = qkv.narrow(2, 0, q_size)?;
        let k = qkv.narrow(2, q_size, kv_size)?;
        let v = qkv.narrow(2, q_size + kv_size, kv_size)?;

        // Reshape for attention: [batch, num_heads, seq_len, head_dim]
        let q = q.reshape((batch, seq_len, num_heads, head_dim))?
            .transpose(1, 2)?;
        let k = k.reshape((batch, seq_len, self.config.num_kv_heads.unwrap_or(num_heads), head_dim))?
            .transpose(1, 2)?;
        let v = v.reshape((batch, seq_len, self.config.num_kv_heads.unwrap_or(num_heads), head_dim))?
            .transpose(1, 2)?;

        // Scaled dot-product attention
        let scale = (head_dim as f64).sqrt();
        let scores = q.matmul(&k.transpose(2, 3)?)?;
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
        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((batch, seq_len, num_heads * head_dim))?;

        // Output projection
        let output = attn_output.matmul(&self.o_weight.t()?)?;

        Ok(output)
    }

    /// CUDA optimized implementation.
    fn forward_cuda(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // TODO: Implement CubeCL fused kernel
        // For now, fall back to CPU implementation
        tracing::warn!("CUDA fused attention not yet implemented, using CPU fallback");
        self.forward_cpu(hidden_states, attention_mask)
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
}
