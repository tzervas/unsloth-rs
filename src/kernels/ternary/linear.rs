// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! Drop-in `TernaryLinear` layer for ternary weight inference.
//!
//! This module provides a `TernaryLinear` layer that can replace
//! `candle_nn::Linear` for memory-efficient inference with ternary weights.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use unsloth_rs::kernels::ternary::{TernaryLinear, quantize_linear_weights, TernaryConfig};
//!
//! // Convert existing Linear layer
//! let fp_linear: candle_nn::Linear = /* ... */;
//! let ternary_weights = quantize_linear_weights(fp_linear.weight(), &TernaryConfig::default())?;
//! let ternary_linear = TernaryLinear::new(ternary_weights, fp_linear.bias().cloned())?;
//!
//! // Forward pass (same API as nn::Linear)
//! let output = ternary_linear.forward(&input)?;
//! ```

use super::config::TernaryConfig;
use super::matmul::ternary_matmul;
use super::types::TernaryTensor;
use crate::error::{Result, UnslothError};
use candle_core::{Module, Tensor};

/// A linear layer with ternary quantized weights.
///
/// Provides significant memory reduction (10-30x) and speedup (5-20x)
/// on sparse pruned models compared to FP16/FP32 linear layers.
///
/// # Memory Layout
///
/// - Weights: `TernaryTensor` with +plane/-plane u32 arrays + f32 scales
/// - Bias: Optional f32 tensor [`out_features`]
///
/// # Forward Pass
///
/// ```text
/// output = input @ weights^T + bias
///        = ternary_matmul(input, ternary_weights) + bias
/// ```
#[derive(Debug, Clone)]
pub struct TernaryLinear {
    /// Ternary quantized weight matrix [`out_features`, `in_features`].
    weights: TernaryTensor,

    /// Optional bias vector [`out_features`].
    bias: Option<Tensor>,

    /// Configuration for ternary operations.
    config: TernaryConfig,
}

impl TernaryLinear {
    /// Create a new `TernaryLinear` layer.
    ///
    /// # Arguments
    ///
    /// * `weights` - Pre-quantized ternary weights
    /// * `bias` - Optional bias tensor [`out_features`]
    ///
    /// # Errors
    ///
    /// Returns error if bias shape doesn't match weight `out_features`.
    pub fn new(weights: TernaryTensor, bias: Option<Tensor>) -> Result<Self> {
        Self::with_config(weights, bias, TernaryConfig::default())
    }

    /// Create with custom configuration.
    ///
    /// # Arguments
    ///
    /// * `weights` - Pre-quantized ternary weights
    /// * `bias` - Optional bias tensor [`out_features`]
    /// * `config` - Ternary configuration
    ///
    /// # Errors
    ///
    /// Returns error if bias shape doesn't match weight `out_features`.
    pub fn with_config(
        weights: TernaryTensor,
        bias: Option<Tensor>,
        config: TernaryConfig,
    ) -> Result<Self> {
        // Validate bias shape if present
        if let Some(ref b) = bias {
            let bias_shape = b.shape().dims();
            if bias_shape.len() != 1 || bias_shape[0] != weights.dims().0 {
                return Err(UnslothError::ShapeMismatch {
                    expected: vec![weights.dims().0],
                    actual: bias_shape.to_vec(),
                });
            }
        }

        Ok(Self {
            weights,
            bias,
            config,
        })
    }

    /// Get the weight tensor dimensions.
    #[must_use]
    pub fn dims(&self) -> (usize, usize) {
        self.weights.dims()
    }

    /// Get the input features (`in_features`).
    #[must_use]
    pub fn in_features(&self) -> usize {
        self.weights.dims().1
    }

    /// Get the output features (`out_features`).
    #[must_use]
    pub fn out_features(&self) -> usize {
        self.weights.dims().0
    }

    /// Get weight sparsity.
    #[must_use]
    pub fn sparsity(&self) -> f32 {
        self.weights.sparsity()
    }

    /// Get compression ratio vs FP32.
    #[must_use]
    pub fn compression_ratio(&self) -> f32 {
        self.weights.compression_ratio()
    }

    /// Get memory usage in bytes.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        let weight_bytes = self.weights.memory_bytes();
        let bias_bytes = self.bias.as_ref().map_or(0, |b| b.elem_count() * 4);
        weight_bytes + bias_bytes
    }

    /// Check if weights are sparse enough for ternary acceleration.
    #[must_use]
    pub fn is_sparse_enough(&self) -> bool {
        self.weights.is_sparse_enough(&self.config)
    }

    /// Get reference to the ternary weights.
    #[must_use]
    pub fn weights(&self) -> &TernaryTensor {
        &self.weights
    }

    /// Get reference to bias if present.
    #[must_use]
    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }

    /// Forward pass.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor [..., `in_features`]
    ///
    /// # Returns
    ///
    /// Output tensor [..., `out_features`]
    ///
    /// # Errors
    ///
    /// Returns error if input shape doesn't match or computation fails.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Compute Y = X @ W^T via ternary matmul
        let mut output = ternary_matmul(input, &self.weights, &self.config)?;

        // Add bias if present
        if let Some(ref bias) = self.bias {
            output = output.broadcast_add(bias)?;
        }

        Ok(output)
    }
}

/// Implement Candle's Module trait for compatibility.
impl Module for TernaryLinear {
    fn forward(&self, input: &Tensor) -> candle_core::Result<Tensor> {
        Self::forward(self, input).map_err(|e| candle_core::Error::Msg(e.to_string()))
    }
}

/// Builder for creating `TernaryLinear` from FP weights.
#[derive(Debug, Clone)]
pub struct TernaryLinearBuilder {
    config: TernaryConfig,
    build_sparsity_metadata: bool,
}

impl Default for TernaryLinearBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl TernaryLinearBuilder {
    /// Create a new builder with default config.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: TernaryConfig::default(),
            build_sparsity_metadata: true,
        }
    }

    /// Set ternary configuration.
    #[must_use]
    pub fn config(mut self, config: TernaryConfig) -> Self {
        self.config = config;
        self
    }

    /// Enable/disable sparsity metadata generation.
    #[must_use]
    pub fn with_sparsity_metadata(mut self, enable: bool) -> Self {
        self.build_sparsity_metadata = enable;
        self
    }

    /// Build from FP weight tensor and optional bias.
    ///
    /// # Arguments
    ///
    /// * `weights` - FP32 weight tensor [`out_features`, `in_features`]
    /// * `bias` - Optional bias tensor [`out_features`]
    ///
    /// # Errors
    ///
    /// Returns error if quantization or validation fails.
    pub fn build(self, weights: &Tensor, bias: Option<Tensor>) -> Result<TernaryLinear> {
        use super::quantize::quantize_tensor;

        let (mut ternary_weights, _stats) = quantize_tensor(weights, &self.config)?;

        if self.build_sparsity_metadata && self.config.enable_dim_metadata {
            ternary_weights.build_sparsity_metadata(self.config.metadata_chunk_size as usize);
        }

        TernaryLinear::with_config(ternary_weights, bias, self.config)
    }

    /// Build from an existing Candle Linear layer.
    ///
    /// # Arguments
    ///
    /// * `linear` - Candle `nn::Linear` layer to convert
    ///
    /// # Errors
    ///
    /// Returns error if quantization fails.
    pub fn build_from_linear(self, linear: &candle_nn::Linear) -> Result<TernaryLinear> {
        let bias = linear.bias().cloned();
        self.build(linear.weight(), bias)
    }
}

/// Convert a Candle `nn::Linear` to `TernaryLinear`.
///
/// Convenience function using default configuration.
///
/// # Arguments
///
/// * `linear` - Candle `nn::Linear` layer
///
/// # Returns
///
/// `TernaryLinear` with quantized weights.
///
/// # Errors
///
/// Returns error if quantization fails.
pub fn convert_linear(linear: &candle_nn::Linear) -> Result<TernaryLinear> {
    TernaryLinearBuilder::new().build_from_linear(linear)
}

/// Convert with custom configuration.
pub fn convert_linear_with_config(
    linear: &candle_nn::Linear,
    config: TernaryConfig,
) -> Result<TernaryLinear> {
    TernaryLinearBuilder::new()
        .config(config)
        .build_from_linear(linear)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_ternary_linear_basic() -> Result<()> {
        // Create simple ternary weights manually
        let shape = (4, 8);
        let k_words = 1; // 8 / 32 rounded up = 1

        // Simple pattern: alternating +1, -1, 0
        let plus = vec![0b00010001u32; 4]; // Bits 0, 4 are +1
        let minus = vec![0b00100010u32; 4]; // Bits 1, 5 are -1
        let scales = vec![1.0f32; 4];

        let ternary_weights = super::super::types::TernaryTensor::new(plus, minus, scales, shape);

        let layer = TernaryLinear::new(ternary_weights, None)?;

        assert_eq!(layer.in_features(), 8);
        assert_eq!(layer.out_features(), 4);

        // Test forward pass
        let input = Tensor::ones((2, 8), candle_core::DType::F32, &Device::Cpu)?;
        let output = layer.forward(&input)?;

        assert_eq!(output.shape().dims(), &[2, 4]);

        Ok(())
    }

    #[test]
    fn test_ternary_linear_with_bias() -> Result<()> {
        use super::super::quantize::quantize_tensor;

        let weight_data = vec![0.5f32; 16];
        let weights = Tensor::from_vec(weight_data, (4, 4), &Device::Cpu)?;

        let config = TernaryConfig::default();
        let (ternary_weights, _) = quantize_tensor(&weights, &config)?;

        let bias_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let bias = Tensor::from_vec(bias_data, 4, &Device::Cpu)?;

        let layer = TernaryLinear::new(ternary_weights, Some(bias))?;

        let input = Tensor::zeros((1, 4), candle_core::DType::F32, &Device::Cpu)?;
        let output = layer.forward(&input)?;

        // Output should be bias since input is zeros
        let output_data: Vec<f32> = output.flatten_all()?.to_vec1()?;
        assert!((output_data[0] - 1.0).abs() < 0.1);
        assert!((output_data[1] - 2.0).abs() < 0.1);

        Ok(())
    }

    #[test]
    fn test_builder_pattern() -> Result<()> {
        // Use larger tensor to get meaningful compression ratio
        let weight_data: Vec<f32> = (0..4096)
            .map(|i| {
                // Precision loss acceptable for test data generation
                #[allow(clippy::cast_precision_loss)]
                {
                    (i as f32 - 2048.0) / 2048.0
                }
            })
            .collect();
        let weights = Tensor::from_vec(weight_data, (64, 64), &Device::Cpu)?;

        let layer = TernaryLinearBuilder::new()
            .config(TernaryConfig::for_sparse_model())
            .with_sparsity_metadata(true)
            .build(&weights, None)?;

        assert_eq!(layer.dims(), (64, 64));
        // Compression should be meaningful for larger tensors
        assert!(
            layer.compression_ratio() > 5.0,
            "Got ratio: {}",
            layer.compression_ratio()
        );

        Ok(())
    }

    #[test]
    fn test_module_trait() -> Result<()> {
        use candle_core::Module;

        let weight_data = vec![1.0f32; 16];
        let weights = Tensor::from_vec(weight_data, (4, 4), &Device::Cpu)?;

        let layer = TernaryLinearBuilder::new().build(&weights, None)?;

        // Use Module::forward trait method
        let input = Tensor::ones((2, 4), candle_core::DType::F32, &Device::Cpu)?;
        let output = Module::forward(&layer, &input)?;

        assert_eq!(output.shape().dims(), &[2, 4]);

        Ok(())
    }

    #[test]
    fn test_memory_efficiency() -> Result<()> {
        // Large layer: 4096 x 4096
        let weight_data = vec![0.1f32; 4096 * 4096];
        let weights = Tensor::from_vec(weight_data, (4096, 4096), &Device::Cpu)?;

        let layer = TernaryLinearBuilder::new()
            .with_sparsity_metadata(false) // Skip metadata for this test
            .build(&weights, None)?;

        // FP32 would be: 4096 * 4096 * 4 = 64MB
        // Ternary should be: ~4MB (2 planes of u32 + scales)
        let ratio = layer.compression_ratio();
        assert!(
            ratio > 10.0,
            "Compression ratio should be >10x, got {ratio}"
        );

        Ok(())
    }
}
