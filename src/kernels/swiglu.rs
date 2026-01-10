// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! `SwiGLU` activation implementation.
//!
//! `SwiGLU` (Swish-Gated Linear Unit) is a gated activation function that
//! combines the Swish activation with a linear gating mechanism.
//!
//! ## Why `SwiGLU`?
//!
//! `SwiGLU` has been shown to outperform other activations (`ReLU`, GELU) in
//! transformer MLPs, and is used in modern LLMs like `LLaMA`, `PaLM`, and others.
//!
//! ## Implementation Notes
//!
//! - Formula: SwiGLU(x) = Swish(x @ `gate_weight`) ⊙ (x @ `up_weight`)
//! - Swish(x) = x * sigmoid(x), also known as `SiLU`
//! - The ⊙ symbol denotes element-wise multiplication
//! - Down projection maps back to `hidden_size`: output = hidden @ `down_weight`

use candle_core::{Device, Tensor};

use crate::error::Result;

/// `SwiGLU` (Swish-Gated Linear Unit) activation.
///
/// Commonly used in LLaMA-style models for MLP layers.
/// `SwiGLU(x) = Swish(xW) ⊙ (xV)`
pub struct SwiGLU {
    /// Gate projection weight
    gate_weight: Tensor,
    /// Up projection weight
    up_weight: Tensor,
    /// Down projection weight
    down_weight: Tensor,
}

impl SwiGLU {
    /// Create a new `SwiGLU` layer.
    ///
    /// # Arguments
    /// * `hidden_size` - Input/output dimension
    /// * `intermediate_size` - Hidden dimension (typically 4 * `hidden_size` * 2/3)
    /// * `device` - Device for tensors
    pub fn new(hidden_size: usize, intermediate_size: usize, device: &Device) -> Result<Self> {
        let std = (1.0 / hidden_size as f64).sqrt() as f32;

        let gate_weight = Tensor::randn(0.0, std, (intermediate_size, hidden_size), device)?;
        let up_weight = Tensor::randn(0.0, std, (intermediate_size, hidden_size), device)?;
        let down_weight = Tensor::randn(0.0, std, (hidden_size, intermediate_size), device)?;

        Ok(Self {
            gate_weight,
            up_weight,
            down_weight,
        })
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `x` - Input tensor [..., `hidden_size`]
    ///
    /// # Returns
    /// Output tensor [..., `hidden_size`]
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let device = x.device();

        if device.is_cuda() {
            self.forward_cuda(x)
        } else {
            self.forward_cpu(x)
        }
    }

    fn forward_cpu(&self, x: &Tensor) -> Result<Tensor> {
        // Gate: Swish(x @ gate_weight^T) - use broadcast_matmul for 3D tensor with 2D weight
        let gate = x.broadcast_matmul(&self.gate_weight.t()?)?;
        let gate = candle_nn::ops::silu(&gate)?;

        // Up: x @ up_weight^T
        let up = x.broadcast_matmul(&self.up_weight.t()?)?;

        // Element-wise multiply
        let hidden = (gate * up)?;

        // Down projection
        let output = hidden.broadcast_matmul(&self.down_weight.t()?)?;

        Ok(output)
    }

    /// CUDA implementation.
    ///
    /// Uses Candle's CUDA backend for GPU acceleration.
    /// The algorithm is the same as the CPU implementation.
    fn forward_cuda(&self, x: &Tensor) -> Result<Tensor> {
        tracing::debug!("Using CUDA SwiGLU path for input shape {:?}", x.shape());
        self.forward_cpu(x)
    }

    /// Estimate VRAM usage in bytes.
    #[must_use]
    pub fn vram_estimate(&self, batch_size: usize, seq_len: usize) -> usize {
        let intermediate = self.gate_weight.dim(0).unwrap_or(0);
        let bytes_per_elem = 4;

        // gate + up + hidden activations
        3 * batch_size * seq_len * intermediate * bytes_per_elem
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    #[test]
    fn test_swiglu_creation() {
        let device = Device::Cpu;
        let swiglu = SwiGLU::new(768, 2048, &device);
        assert!(swiglu.is_ok());
    }

    #[test]
    fn test_swiglu_forward() {
        let device = Device::Cpu;
        let swiglu = SwiGLU::new(768, 2048, &device).unwrap();

        let input = Tensor::zeros(&[2, 10, 768], DType::F32, &device).unwrap();
        let output = swiglu.forward(&input).unwrap();

        assert_eq!(output.shape().dims(), &[2, 10, 768]);
    }
}
