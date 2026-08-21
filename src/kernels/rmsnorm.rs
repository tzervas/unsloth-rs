// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! RMS Normalization (LLaMA-style) via the G0 CustomOp path.
//!
//! Formula: `output = (x / sqrt(mean(x^2) + eps)) * weight` over the last dim.
//!
//! `forward` always goes through [`crate::kernels::custom_op::rmsnorm_custom_op`]:
//! Candle `CustomOp2` on `CpuStorage` / `CudaStorage`. It does **not**
//! `to_vec1` into CubeCL. The old CUDA path was unfused Candle tensor ops
//! (still on device, not a fused kernel); that is no longer the default.
//!
//! CubeCL interop (`kernels::cubecl`) remains a separate, host-roundtrip path.

use candle_core::{DType, Device, Tensor};

use crate::error::Result;
use crate::kernels::custom_op::rmsnorm_custom_op;

/// Root Mean Square Layer Normalization.
///
/// f32 CustomOp forward on whatever device `x` lives on.
pub struct RmsNorm {
    /// Learned scale parameter `[hidden_size]`.
    weight: Tensor,
    /// Epsilon for numerical stability.
    eps: f64,
}

impl RmsNorm {
    /// Create a new RMS normalization layer (ones init, f32).
    ///
    /// # Arguments
    /// * `hidden_size` - Dimension to normalize over
    /// * `eps` - Epsilon for numerical stability
    /// * `device` - Device for the weight tensor
    pub fn new(hidden_size: usize, eps: f64, device: &Device) -> Result<Self> {
        let weight = Tensor::ones((hidden_size,), DType::F32, device)?;
        Ok(Self { weight, eps })
    }

    /// Forward: CustomOp RMSNorm (device-resident, f32).
    ///
    /// # Arguments
    /// * `x` - Input tensor `[..., hidden_size]`
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(rmsnorm_custom_op(x, &self.weight, self.eps as f32)?)
    }

    /// Borrow the scale weights.
    #[must_use]
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    /// Epsilon used by this layer.
    #[must_use]
    pub fn eps(&self) -> f64 {
        self.eps
    }

    /// Activation bytes for `(batch, seq)` at f32. Weight is not included.
    ///
    /// Offline planning helper, same contract as [`crate::kernels::swiglu::SwiGLU::vram_estimate`].
    /// Not a measured allocation and not a 2× / 70% VRAM claim.
    #[must_use]
    pub fn vram_estimate(&self, batch_size: usize, seq_len: usize) -> usize {
        let hidden_size = self.weight.dim(0).unwrap_or(0);
        let bytes_per_elem = 4; // f32
        batch_size * seq_len * hidden_size * bytes_per_elem
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rmsnorm_creation() {
        let device = Device::Cpu;
        let norm = RmsNorm::new(768, 1e-5, &device);
        assert!(norm.is_ok());
    }

    #[test]
    fn test_rmsnorm_forward() {
        let device = Device::Cpu;
        let norm = RmsNorm::new(768, 1e-5, &device).unwrap();

        let input = Tensor::randn(0.0f32, 1.0, (2, 10, 768), &device).unwrap();
        let output = norm.forward(&input).unwrap();

        assert_eq!(output.shape().dims(), &[2, 10, 768]);
    }

    #[test]
    fn test_rmsnorm_normalizes_values() {
        let device = Device::Cpu;
        let norm = RmsNorm::new(64, 1e-5, &device).unwrap();

        let input = Tensor::randn(0.0f32, 5.0, (1, 1, 64), &device).unwrap();
        let output = norm.forward(&input).unwrap();

        let output_sq = output.sqr().unwrap();
        let mean_sq = output_sq.mean_all().unwrap().to_scalar::<f32>().unwrap();

        assert!(
            (mean_sq.sqrt() - 1.0).abs() < 0.5,
            "RMS should be approximately 1, got {}",
            mean_sq.sqrt()
        );
    }

    #[test]
    fn test_rmsnorm_numerical_stability() {
        let device = Device::Cpu;
        let norm = RmsNorm::new(128, 1e-5, &device).unwrap();

        let small_input = Tensor::full(1e-6f32, (1, 1, 128), &device).unwrap();
        let output = norm.forward(&small_input);
        assert!(output.is_ok());

        let large_input = Tensor::randn(0.0f32, 100.0, (1, 1, 128), &device).unwrap();
        let output = norm.forward(&large_input).unwrap();

        let values: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();
        for v in values {
            assert!(!v.is_nan(), "Output contains NaN");
            assert!(!v.is_infinite(), "Output contains Inf");
        }
    }

    #[test]
    fn test_rmsnorm_matches_broadcast_reference() {
        let device = Device::Cpu;
        let hidden = 48;
        let norm = RmsNorm::new(hidden, 1e-6, &device).unwrap();
        let x = Tensor::randn(0.0f32, 1.0, (4, 7, hidden), &device).unwrap();
        let y = norm.forward(&x).unwrap();
        let mean_sq = x.sqr().unwrap().mean_keepdim(2).unwrap();
        let ref_y = x
            .broadcast_div(&(mean_sq + 1e-6).unwrap().sqrt().unwrap())
            .unwrap()
            .broadcast_mul(norm.weight())
            .unwrap();
        let mae = (y - ref_y)
            .unwrap()
            .abs()
            .unwrap()
            .mean_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(mae < 1e-6, "mae={mae}");
    }

    #[test]
    fn test_rmsnorm_vram_estimate() {
        let device = Device::Cpu;
        let norm = RmsNorm::new(768, 1e-5, &device).unwrap();
        let vram = norm.vram_estimate(4, 2048);
        assert_eq!(vram, 4 * 2048 * 768 * 4);
    }
}
