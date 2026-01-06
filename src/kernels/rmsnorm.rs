//! RMS Normalization implementation.

use candle_core::{DType, Device, Tensor};

use crate::error::Result;

/// Root Mean Square Layer Normalization.
///
/// Normalizes inputs using RMS, commonly used in LLaMA-style models.
pub struct RmsNorm {
    /// Learned scale parameter
    weight: Tensor,
    /// Epsilon for numerical stability
    eps: f64,
}

impl RmsNorm {
    /// Create a new RMS normalization layer.
    ///
    /// # Arguments
    /// * `hidden_size` - Dimension to normalize over
    /// * `eps` - Epsilon for numerical stability
    /// * `device` - Device for tensors
    pub fn new(hidden_size: usize, eps: f64, device: &Device) -> Result<Self> {
        let weight = Tensor::ones((hidden_size,), DType::F32, device)?;
        Ok(Self { weight, eps })
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `x` - Input tensor [..., hidden_size]
    ///
    /// # Returns
    /// Normalized tensor with same shape
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let device = x.device();
        
        if device.is_cuda() {
            self.forward_cuda(x)
        } else {
            self.forward_cpu(x)
        }
    }

    fn forward_cpu(&self, x: &Tensor) -> Result<Tensor> {
        // RMS = sqrt(mean(x^2))
        let x_sq = x.sqr()?;
        let mean_sq = x_sq.mean_keepdim(x.rank() - 1)?;
        let rms = (mean_sq + self.eps)?.sqrt()?;
        
        // Normalize and scale
        let normalized = x.broadcast_div(&rms)?;
        let output = normalized.broadcast_mul(&self.weight)?;
        
        Ok(output)
    }

    fn forward_cuda(&self, x: &Tensor) -> Result<Tensor> {
        // TODO: Implement fused CubeCL kernel
        self.forward_cpu(x)
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
}
