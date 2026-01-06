//! RMS Normalization implementation.
//!
//! Root Mean Square Layer Normalization normalizes inputs using only the RMS
//! (root mean square) statistic, without centering (no mean subtraction).
//!
//! ## Why RMSNorm?
//!
//! Compared to LayerNorm:
//! - Simpler computation (no mean calculation needed)
//! - Empirically performs similarly in practice
//! - Used in modern LLMs like LLaMA for efficiency
//!
//! ## Implementation Notes
//!
//! - Formula: output = (x / RMS(x)) * weight
//! - RMS(x) = sqrt(mean(x^2) + eps)
//! - Epsilon (eps) prevents division by zero for numerical stability

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

    /// CUDA implementation.
    ///
    /// Uses Candle's CUDA backend for GPU acceleration.
    /// The algorithm is the same as the CPU implementation.
    fn forward_cuda(&self, x: &Tensor) -> Result<Tensor> {
        tracing::debug!("Using CUDA RMSNorm path for input shape {:?}", x.shape());
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

    #[test]
    fn test_rmsnorm_normalizes_values() {
        let device = Device::Cpu;
        let norm = RmsNorm::new(64, 1e-5, &device).unwrap();

        // Create input with known variance
        let input = Tensor::randn(0.0f32, 5.0, (1, 1, 64), &device).unwrap();
        let output = norm.forward(&input).unwrap();

        // Output should have approximately unit RMS
        let output_sq = output.sqr().unwrap();
        let mean_sq = output_sq.mean_all().unwrap().to_scalar::<f32>().unwrap();
        
        // RMS should be close to 1 (within tolerance)
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

        // Test with very small values
        let small_input = Tensor::full(1e-6f32, (1, 1, 128), &device).unwrap();
        let output = norm.forward(&small_input);
        assert!(output.is_ok());
        
        // Test with larger values
        let large_input = Tensor::randn(0.0f32, 100.0, (1, 1, 128), &device).unwrap();
        let output = norm.forward(&large_input).unwrap();
        
        // Check no NaN/Inf
        let values: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();
        for v in values {
            assert!(!v.is_nan(), "Output contains NaN");
            assert!(!v.is_infinite(), "Output contains Inf");
        }
    }
}
