// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! FP → Ternary quantization with scale calibration.
//!
//! This module implements TWN-style (Ternary Weight Networks) quantization
//! to convert floating-point weights to ternary {-1, 0, +1} representation.
//!
//! ## Quantization Formula
//!
//! For a weight tensor W with threshold Δ:
//!
//! ```text
//! W_ternary[i] = +1  if W[i] > Δ
//!              =  0  if |W[i]| ≤ Δ
//!              = -1  if W[i] < -Δ
//!
//! scale = mean(|W[i]|) for i where |W[i]| > Δ
//! ```
//!
//! ## Calibration Methods
//!
//! - **`AbsMax`**: Δ = α × max(|W|), where α ∈ [0, 1] (typically 0.7)
//! - **Percentile**: Δ = percentile(|W|, p), e.g., p=99.5
//! - **`MeanStd`**: Δ = mean(|W|) + k × std(|W|)
//!
//! ## References
//!
//! - Li et al., "Ternary Weight Networks" (2016)
//! - Mellempudi et al., "Ternary Neural Networks with Fine-Grained Quantization" (2017)

use super::config::{CalibrationMethodConfig, TernaryConfig};
use super::types::TernaryTensor;
use crate::error::{Result, UnslothError};
use candle_core::{DType, Tensor};

/// Calibration method for determining quantization threshold.
#[derive(Debug, Clone, Copy)]
pub enum CalibrationMethod {
    /// Threshold = factor × max(|W|).
    /// Factor typically 0.7 for TWN.
    AbsMax {
        /// Scaling factor applied to max absolute value (typically 0.7).
        factor: f32,
    },

    /// Threshold = percentile of |W|.
    /// Percentile typically 99.5 to exclude outliers.
    Percentile {
        /// Percentile value (0-100) for threshold selection.
        percentile: f32,
    },

    /// Threshold = mean(|W|) + k × std(|W|).
    /// k typically 1.0-2.0.
    MeanStd {
        /// Standard deviation multiplier.
        k: f32,
    },

    /// Fixed threshold value.
    Manual {
        /// Fixed threshold for ternary quantization.
        threshold: f32,
    },
}

impl Default for CalibrationMethod {
    fn default() -> Self {
        Self::AbsMax { factor: 0.7 }
    }
}

impl From<CalibrationMethodConfig> for CalibrationMethod {
    fn from(config: CalibrationMethodConfig) -> Self {
        match config {
            CalibrationMethodConfig::AbsMax => Self::AbsMax { factor: 0.7 },
            CalibrationMethodConfig::Percentile(p) => Self::Percentile { percentile: p },
            CalibrationMethodConfig::MeanStd(k) => Self::MeanStd { k },
            CalibrationMethodConfig::Manual(t) => Self::Manual { threshold: t },
        }
    }
}

/// Statistics computed during quantization.
#[derive(Debug, Clone)]
pub struct QuantizationStats {
    /// Fraction of weights quantized to 0.
    pub sparsity: f32,

    /// Fraction of weights quantized to +1.
    pub positive_ratio: f32,

    /// Fraction of weights quantized to -1.
    pub negative_ratio: f32,

    /// Per-channel thresholds used.
    pub thresholds: Vec<f32>,

    /// Per-channel scales computed.
    pub scales: Vec<f32>,

    /// Mean absolute quantization error.
    pub mean_error: f32,

    /// Max absolute quantization error.
    pub max_error: f32,
}

/// Quantize a 2D tensor to ternary representation.
///
/// # Arguments
///
/// * `tensor` - Input tensor [`out_features`, `in_features`] (must be 2D, f32)
/// * `config` - Ternary configuration with calibration settings
///
/// # Returns
///
/// Tuple of (`TernaryTensor`, `QuantizationStats`)
///
/// # Errors
///
/// Returns error if tensor is not 2D or not f32.
///
/// # Example
///
/// ```rust,ignore
/// use unsloth_rs::kernels::ternary::{quantize_tensor, TernaryConfig};
///
/// let weights = Tensor::randn(0.0f32, 1.0, (1024, 4096), &Device::Cpu)?;
/// let (ternary, stats) = quantize_tensor(&weights, &TernaryConfig::default())?;
///
/// println!("Sparsity: {:.1}%", stats.sparsity * 100.0);
/// println!("Compression: {:.1}x", ternary.compression_ratio());
/// ```
pub fn quantize_tensor(
    tensor: &Tensor,
    config: &TernaryConfig,
) -> Result<(TernaryTensor, QuantizationStats)> {
    // Validate input
    let shape = tensor.shape();
    if shape.dims().len() != 2 {
        return Err(UnslothError::ShapeMismatch {
            // Expected a 2D tensor (rank 2) for weight matrix
            expected: vec![2],
            actual: shape.dims().to_vec(),
        });
    }

    if tensor.dtype() != DType::F32 {
        return Err(UnslothError::InvalidConfig(format!(
            "quantize_tensor requires f32, got {:?}",
            tensor.dtype()
        )));
    }

    let (out_features, in_features) = (shape.dims()[0], shape.dims()[1]);

    // Get data as f32 vec (move to CPU if needed)
    let data: Vec<f32> = tensor.flatten_all()?.to_vec1()?;

    // Determine calibration method
    let calibration = CalibrationMethod::from(config.calibration_method);

    // Quantize per row (per output channel)
    let k_words = in_features.div_ceil(32);
    let mut plus_plane = vec![0u32; out_features * k_words];
    let mut minus_plane = vec![0u32; out_features * k_words];
    let mut scales = vec![0.0f32; out_features];
    let mut thresholds = vec![0.0f32; out_features];

    let mut total_positive = 0usize;
    let mut total_negative = 0usize;
    let mut total_zero = 0usize;
    let mut total_error = 0.0f64;
    let mut max_error = 0.0f32;

    for row in 0..out_features {
        let row_start = row * in_features;
        let row_data = &data[row_start..row_start + in_features];

        // Compute threshold for this row
        let threshold = compute_threshold(row_data, calibration);
        thresholds[row] = threshold;

        // Quantize and compute scale
        let (row_plus, row_minus, scale, pos, neg, zero) =
            quantize_row(row_data, threshold, k_words);

        // Copy to output planes
        let plane_offset = row * k_words;
        plus_plane[plane_offset..plane_offset + k_words].copy_from_slice(&row_plus);
        minus_plane[plane_offset..plane_offset + k_words].copy_from_slice(&row_minus);
        scales[row] = scale;

        total_positive += pos;
        total_negative += neg;
        total_zero += zero;

        // Compute reconstruction error
        for (i, &val) in row_data.iter().enumerate() {
            let word_idx = i / 32;
            let bit_idx = i % 32;
            let mask = 1u32 << bit_idx;

            let is_plus = (row_plus[word_idx] & mask) != 0;
            let is_minus = (row_minus[word_idx] & mask) != 0;

            let reconstructed = if is_plus {
                scale
            } else if is_minus {
                -scale
            } else {
                0.0
            };

            let error = (val - reconstructed).abs();
            total_error += f64::from(error);
            max_error = max_error.max(error);
        }
    }

    let total_elements = out_features * in_features;
    #[allow(clippy::cast_precision_loss)] // Sparsity calculations for statistics only
    let stats = QuantizationStats {
        sparsity: total_zero as f32 / total_elements as f32,
        positive_ratio: total_positive as f32 / total_elements as f32,
        negative_ratio: total_negative as f32 / total_elements as f32,
        thresholds,
        scales: scales.clone(),
        #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)] // Error statistics approximation
        mean_error: (total_error / total_elements as f64) as f32,
        max_error,
    };

    let ternary = TernaryTensor::new(plus_plane, minus_plane, scales, (out_features, in_features));

    Ok((ternary, stats))
}

/// Compute quantization threshold using specified calibration method.
fn compute_threshold(data: &[f32], method: CalibrationMethod) -> f32 {
    match method {
        CalibrationMethod::AbsMax { factor } => {
            let max_abs = data.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            factor * max_abs
        }

        CalibrationMethod::Percentile { percentile } => {
            let mut abs_values: Vec<f32> = data.iter().map(|x| x.abs()).collect();
            abs_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                clippy::cast_precision_loss
            )]
            // Percentile calculation: precision loss acceptable for threshold approximation
            let idx = ((percentile / 100.0) * (abs_values.len() - 1) as f32) as usize;
            abs_values[idx.min(abs_values.len() - 1)]
        }

        CalibrationMethod::MeanStd { k } => {
            #[allow(clippy::cast_precision_loss)]
            // Precision loss acceptable for statistical calculations
            let n = data.len() as f64;
            let abs_values: Vec<f64> = data.iter().map(|x| f64::from(x.abs())).collect();

            let mean = abs_values.iter().sum::<f64>() / n;
            let variance = abs_values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
            let std = variance.sqrt();

            // Truncation acceptable for threshold calculation
            #[allow(clippy::cast_possible_truncation)]
            let threshold_value = (mean + f64::from(k) * std) as f32;
            threshold_value
        }

        CalibrationMethod::Manual { threshold } => threshold,
    }
}

/// Quantize a single row to ternary planes.
///
/// Returns (`plus_plane`, `minus_plane`, `scale`, `positive_count`, `negative_count`, `zero_count`)
fn quantize_row(
    data: &[f32],
    threshold: f32,
    k_words: usize,
) -> (Vec<u32>, Vec<u32>, f32, usize, usize, usize) {
    let mut plus = vec![0u32; k_words];
    let mut minus = vec![0u32; k_words];

    let mut positive_sum = 0.0f64;
    let mut positive_count = 0usize;
    let mut negative_sum = 0.0f64;
    let mut negative_count = 0usize;
    let mut zero_count = 0usize;

    for (i, &val) in data.iter().enumerate() {
        let word_idx = i / 32;
        let bit_idx = i % 32;
        let mask = 1u32 << bit_idx;

        if val > threshold {
            plus[word_idx] |= mask;
            positive_sum += f64::from(val.abs());
            positive_count += 1;
        } else if val < -threshold {
            minus[word_idx] |= mask;
            negative_sum += f64::from(val.abs());
            negative_count += 1;
        } else {
            zero_count += 1;
        }
    }

    // Scale is mean of non-zero absolute values
    let nonzero_count = positive_count + negative_count;
    let scale = if nonzero_count > 0 {
        // Truncation/precision loss acceptable for scale approximation
        #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
        let scale = ((positive_sum + negative_sum) / nonzero_count as f64) as f32;
        scale
    } else {
        1.0 // Fallback for all-zero rows
    };

    (
        plus,
        minus,
        scale,
        positive_count,
        negative_count,
        zero_count,
    )
}

/// Dequantize a ternary tensor back to f32 (for validation).
///
/// # Arguments
///
/// * `ternary` - Ternary tensor to dequantize
///
/// # Returns
///
/// Candle tensor [`out_features`, `in_features`] with reconstructed f32 values.
///
/// # Errors
///
/// Returns error if tensor creation fails.
pub fn dequantize_tensor(ternary: &TernaryTensor) -> Result<Tensor> {
    let (out_features, in_features) = ternary.dims();
    let mut data = vec![0.0f32; out_features * in_features];

    for row in 0..out_features {
        let scale = ternary.scales[row];
        let planes = ternary.get_row_planes(row);

        for col in 0..in_features {
            let val = planes.get(col);
            data[row * in_features + col] = f32::from(val) * scale;
        }
    }

    let tensor = Tensor::from_vec(data, (out_features, in_features), &candle_core::Device::Cpu)?;
    Ok(tensor)
}

/// Quantize weights directly from Candle Linear layer.
///
/// # Arguments
///
/// * `weights` - Weight tensor from `nn::Linear` [`out_features`, `in_features`]
/// * `config` - Ternary configuration
///
/// # Returns
///
/// Ternary tensor ready for use in `TernaryLinear`.
///
/// # Errors
///
/// Returns an error if the weight tensor cannot be accessed, has invalid dimensions,
/// or if quantization fails due to numerical issues.
pub fn quantize_linear_weights(weights: &Tensor, config: &TernaryConfig) -> Result<TernaryTensor> {
    let (ternary, _stats) = quantize_tensor(weights, config)?;
    Ok(ternary)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_quantize_simple() -> Result<()> {
        // Create simple test tensor
        let data: Vec<f32> = vec![
            0.5, -0.5, 0.1, -0.1, 0.8, -0.8, 0.0, 0.3, // Row 0
            1.0, -1.0, 0.2, -0.2, 0.0, 0.0, 0.9, -0.9, // Row 1
        ];
        let tensor = Tensor::from_vec(data, (2, 8), &Device::Cpu)?;

        let config = TernaryConfig {
            calibration_method: CalibrationMethodConfig::Manual(0.3),
            ..Default::default()
        };

        let (ternary, stats) = quantize_tensor(&tensor, &config)?;

        assert_eq!(ternary.dims(), (2, 8));
        assert!(stats.sparsity > 0.0); // Some zeros
        assert!(stats.positive_ratio > 0.0);
        assert!(stats.negative_ratio > 0.0);

        Ok(())
    }

    #[test]
    fn test_quantize_dequantize_roundtrip() -> Result<()> {
        // Test that dequantization produces reasonable values
        let data: Vec<f32> = (0..256)
            .map(|i| {
                // Precision loss acceptable for test data generation
                #[allow(clippy::cast_precision_loss)]
                {
                    (i as f32 - 128.0) / 128.0
                }
            })
            .collect();
        let tensor = Tensor::from_vec(data.clone(), (4, 64), &Device::Cpu)?;

        let config = TernaryConfig::default();
        let (ternary, _stats) = quantize_tensor(&tensor, &config)?;

        let reconstructed = dequantize_tensor(&ternary)?;
        let recon_data: Vec<f32> = reconstructed.flatten_all()?.to_vec1()?;

        // Check that reconstruction is reasonable (not exact due to quantization)
        let mse: f32 = data
            .iter()
            .zip(recon_data.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / {
                // Precision loss acceptable for test metric calculation
                #[allow(clippy::cast_precision_loss)]
                {
                    data.len() as f32
                }
            };

        // MSE should be reasonable (< 0.5 for well-calibrated ternary)
        assert!(mse < 0.5, "MSE too high: {mse}");

        Ok(())
    }

    #[test]
    fn test_calibration_methods() {
        let data: Vec<f32> = vec![0.1, 0.5, 1.0, -0.3, -0.8, 2.0, -1.5, 0.0];

        // AbsMax: factor * max(|x|) = 0.7 * 2.0 = 1.4
        let t1 = compute_threshold(&data, CalibrationMethod::AbsMax { factor: 0.7 });
        assert!((t1 - 1.4).abs() < 0.01);

        // Manual
        let t2 = compute_threshold(&data, CalibrationMethod::Manual { threshold: 0.5 });
        assert!((t2 - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_sparsity_detection() -> Result<()> {
        // Create sparse tensor (90% zeros)
        let mut data = vec![0.0f32; 1000];
        for i in 0..100 {
            data[i * 10] = if i % 2 == 0 { 1.0 } else { -1.0 };
        }
        let tensor = Tensor::from_vec(data, (10, 100), &Device::Cpu)?;

        let config = TernaryConfig {
            calibration_method: CalibrationMethodConfig::Manual(0.1),
            ..Default::default()
        };

        let (ternary, stats) = quantize_tensor(&tensor, &config)?;

        // Should have ~90% sparsity after quantization
        assert!(stats.sparsity > 0.85, "Sparsity: {}", stats.sparsity);
        assert!(ternary.sparsity() > 0.85);

        Ok(())
    }

    #[test]
    fn test_compression_ratio() -> Result<()> {
        let data = vec![0.0f32; 4096 * 4096];
        let tensor = Tensor::from_vec(data, (4096, 4096), &Device::Cpu)?;

        let config = TernaryConfig::default();
        let (ternary, _) = quantize_tensor(&tensor, &config)?;

        // Compression should be ~16x for 2 bits vs 32 bits
        let ratio = ternary.compression_ratio();
        assert!(ratio > 10.0, "Compression ratio too low: {ratio}");

        Ok(())
    }
}
