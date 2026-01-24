// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! Training utilities.
//!
//! This module provides training utilities including:
//! - Mixed precision training support (FP32, FP16, BF16)
//! - Gradient scaling for numerical stability
//! - Gradient checkpointing configuration

use candle_core::{DType, Tensor};

use crate::error::{Result, UnslothError};
use crate::memory::CheckpointConfig;

/// Precision mode for training.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrecisionMode {
    /// Full precision (FP32)
    Full,
    /// Half precision (FP16)
    Half,
    /// Brain float 16 (BF16)
    BFloat16,
}

impl PrecisionMode {
    /// Convert precision mode to Candle `DType`.
    #[must_use]
    pub fn to_dtype(&self) -> DType {
        match self {
            Self::Full => DType::F32,
            Self::Half => DType::F16,
            Self::BFloat16 => DType::BF16,
        }
    }

    /// Get the precision mode from a Candle `DType`.
    ///
    /// # Errors
    /// Returns error if dtype is not a supported floating point type.
    pub fn from_dtype(dtype: DType) -> Result<Self> {
        match dtype {
            DType::F32 => Ok(Self::Full),
            DType::F16 => Ok(Self::Half),
            DType::BF16 => Ok(Self::BFloat16),
            _ => Err(UnslothError::InvalidConfig(format!(
                "Unsupported dtype for mixed precision: {dtype:?}"
            ))),
        }
    }
}

/// Mixed precision training configuration.
#[derive(Debug, Clone)]
pub struct MixedPrecisionConfig {
    /// Precision mode for computation
    pub compute_precision: PrecisionMode,
    /// Precision mode for master weights (usually FP32)
    pub master_precision: PrecisionMode,
    /// Loss scale factor to prevent gradient underflow
    pub loss_scale: f32,
    /// Enable dynamic loss scaling
    pub dynamic_loss_scale: bool,
    /// Minimum loss scale for dynamic scaling
    pub min_loss_scale: f32,
    /// Maximum loss scale for dynamic scaling
    pub max_loss_scale: f32,
    /// Growth factor for dynamic loss scaling
    pub scale_growth_factor: f32,
    /// Backoff factor for dynamic loss scaling
    pub scale_backoff_factor: f32,
    /// Number of consecutive non-overflow steps before increasing scale
    pub scale_growth_interval: usize,
}

impl Default for MixedPrecisionConfig {
    fn default() -> Self {
        Self {
            compute_precision: PrecisionMode::Half,
            master_precision: PrecisionMode::Full,
            loss_scale: 65536.0, // 2^16
            dynamic_loss_scale: true,
            min_loss_scale: 1.0,
            max_loss_scale: 2_147_483_648.0, // 2^31
            scale_growth_factor: 2.0,
            scale_backoff_factor: 0.5,
            scale_growth_interval: 2000,
        }
    }
}

impl MixedPrecisionConfig {
    /// Create a new mixed precision configuration.
    #[must_use]
    pub fn new(compute_precision: PrecisionMode) -> Self {
        Self {
            compute_precision,
            ..Default::default()
        }
    }

    /// Create a configuration for FP16 training.
    #[must_use]
    pub fn fp16() -> Self {
        Self::new(PrecisionMode::Half)
    }

    /// Create a configuration for BF16 training.
    #[must_use]
    pub fn bf16() -> Self {
        Self::new(PrecisionMode::BFloat16)
    }

    /// Create a configuration for FP32 training (no mixed precision).
    #[must_use]
    pub fn fp32() -> Self {
        Self {
            compute_precision: PrecisionMode::Full,
            master_precision: PrecisionMode::Full,
            dynamic_loss_scale: false,
            loss_scale: 1.0,
            ..Default::default()
        }
    }
}

/// Training configuration.
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Batch size
    pub batch_size: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Gradient accumulation steps
    pub gradient_accumulation_steps: usize,
    /// Mixed precision configuration (None = FP32)
    pub mixed_precision: Option<MixedPrecisionConfig>,
    /// Gradient checkpointing
    pub checkpoint_config: CheckpointConfig,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            batch_size: 4,
            max_seq_len: 2048,
            gradient_accumulation_steps: 4,
            mixed_precision: Some(MixedPrecisionConfig::default()),
            checkpoint_config: CheckpointConfig::default(),
        }
    }
}

/// Convert tensor to specified precision.
///
/// # Arguments
/// * `tensor` - Input tensor to convert
/// * `precision` - Target precision mode (FP32, FP16, or BF16)
///
/// # Returns
/// Tensor converted to target precision
///
/// # Errors
/// Returns an error if the dtype conversion fails.
pub fn convert_precision(tensor: &Tensor, precision: PrecisionMode) -> Result<Tensor> {
    let target_dtype = precision.to_dtype();
    if tensor.dtype() == target_dtype {
        Ok(tensor.clone())
    } else {
        Ok(tensor.to_dtype(target_dtype)?)
    }
}

/// Scale loss for mixed precision training.
///
/// Scales the loss by the loss scale factor to prevent gradient underflow
/// in lower precision formats.
///
/// # Arguments
/// * `loss` - Original loss tensor to scale
/// * `config` - Mixed precision configuration containing the loss scale factor
///
/// # Returns
/// Scaled loss tensor
///
/// # Errors
/// Returns an error if tensor multiplication fails.
pub fn scale_loss(loss: &Tensor, config: &MixedPrecisionConfig) -> Result<Tensor> {
    if (config.loss_scale - 1.0).abs() < f32::EPSILON {
        Ok(loss.clone())
    } else {
        Ok((loss * f64::from(config.loss_scale))?)
    }
}

/// Unscale gradients after backward pass.
///
/// Divides gradients by the loss scale factor to get the true gradient values.
///
/// # Arguments
/// * `gradients` - Scaled gradients from backward pass
/// * `config` - Mixed precision configuration
///
/// # Returns
/// Unscaled gradients
///
/// # Errors
///
/// Returns an error if tensor operations fail.
pub fn unscale_gradients(
    gradients: &[Tensor],
    config: &MixedPrecisionConfig,
) -> Result<Vec<Tensor>> {
    if (config.loss_scale - 1.0).abs() < f32::EPSILON {
        Ok(gradients.to_vec())
    } else {
        let scale = 1.0 / f64::from(config.loss_scale);
        gradients
            .iter()
            .map(|g| (g * scale).map_err(Into::into))
            .collect()
    }
}

/// Check if gradients contain NaN or Inf values.
///
/// Used to detect gradient overflow in mixed precision training.
///
/// # Arguments
/// * `gradients` - Slice of gradient tensors to check for numerical instability
///
/// # Returns
/// `true` if any gradient contains NaN or Inf, `false` otherwise
///
/// # Errors
/// Returns an error if tensor dtype conversion or flattening fails.
pub fn has_inf_or_nan(gradients: &[Tensor]) -> Result<bool> {
    for grad in gradients {
        let grad_f32 = grad.to_dtype(DType::F32)?;
        let values: Vec<f32> = grad_f32.flatten_all()?.to_vec1()?;

        for &val in &values {
            if val.is_nan() || val.is_infinite() {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

/// Update loss scale based on gradient overflow status.
///
/// Implements dynamic loss scaling to automatically adjust the loss scale
/// based on whether gradients overflow.
///
/// # Arguments
/// * `config` - Mixed precision configuration (will be modified)
/// * `has_overflow` - Whether gradients overflowed in this step
/// * `steps_since_overflow` - Number of steps since last overflow
///
/// # Returns
/// New loss scale value
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
pub fn update_loss_scale(
    config: &mut MixedPrecisionConfig,
    has_overflow: bool,
    steps_since_overflow: usize,
) -> f32 {
    if !config.dynamic_loss_scale {
        return config.loss_scale;
    }

    if has_overflow {
        // Reduce loss scale on overflow
        config.loss_scale =
            (config.loss_scale * config.scale_backoff_factor).max(config.min_loss_scale);
    } else if steps_since_overflow >= config.scale_growth_interval {
        // Increase loss scale after many successful steps
        config.loss_scale =
            (config.loss_scale * config.scale_growth_factor).min(config.max_loss_scale);
    }

    config.loss_scale
}

/// Compute gradient with optional checkpointing.
///
/// This function performs gradient computation with activation checkpointing,
/// which trades compute for memory by recomputing activations during the backward pass
/// instead of storing them in memory.
///
/// # Arguments
/// * `_input` - Input tensor for the forward pass
/// * `_forward_fn` - Function that computes the forward pass
/// * `_config` - Checkpoint configuration specifying checkpointing strategy
///
/// # Returns
/// Computed gradient tensor
///
/// # Errors
/// Returns an error if gradient computation fails.
///
/// # Note
/// This is currently unimplemented and will return an error.
/// Gradient checkpointing is planned for a future release.
pub fn compute_gradient_checkpointed<F>(
    _input: &Tensor,
    _forward_fn: F,
    _config: &CheckpointConfig,
) -> Result<Tensor>
where
    F: Fn(&Tensor) -> Result<Tensor>,
{
    // TODO: Implement gradient checkpointing
    // This would recompute forward pass during backward instead of storing activations
    Err(UnslothError::InvalidConfig(
        "Gradient checkpointing is not yet implemented. This feature is planned for a future release.".to_string()
    ))
}

/// Scale gradients for mixed precision training.
pub fn scale_gradients(gradients: &[Tensor], scale: f32) -> Result<Vec<Tensor>> {
    gradients
        .iter()
        .map(|g| (g * f64::from(scale)).map_err(Into::into))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_training_config_default() {
        let config = TrainingConfig::default();
        assert_eq!(config.batch_size, 4);
        assert!(config.mixed_precision.is_some());
    }

    #[test]
    fn test_precision_mode_to_dtype() {
        assert_eq!(PrecisionMode::Full.to_dtype(), DType::F32);
        assert_eq!(PrecisionMode::Half.to_dtype(), DType::F16);
        assert_eq!(PrecisionMode::BFloat16.to_dtype(), DType::BF16);
    }

    #[test]
    fn test_precision_mode_from_dtype() {
        assert_eq!(
            PrecisionMode::from_dtype(DType::F32).unwrap(),
            PrecisionMode::Full
        );
        assert_eq!(
            PrecisionMode::from_dtype(DType::F16).unwrap(),
            PrecisionMode::Half
        );
        assert_eq!(
            PrecisionMode::from_dtype(DType::BF16).unwrap(),
            PrecisionMode::BFloat16
        );

        // Test unsupported dtype
        assert!(PrecisionMode::from_dtype(DType::U8).is_err());
    }

    #[test]
    fn test_mixed_precision_config_defaults() {
        let config = MixedPrecisionConfig::default();
        assert_eq!(config.compute_precision, PrecisionMode::Half);
        assert_eq!(config.master_precision, PrecisionMode::Full);
        assert_eq!(config.loss_scale, 65536.0);
        assert!(config.dynamic_loss_scale);
    }

    #[test]
    fn test_mixed_precision_config_fp16() {
        let config = MixedPrecisionConfig::fp16();
        assert_eq!(config.compute_precision, PrecisionMode::Half);
        assert_eq!(config.master_precision, PrecisionMode::Full);
    }

    #[test]
    fn test_mixed_precision_config_bf16() {
        let config = MixedPrecisionConfig::bf16();
        assert_eq!(config.compute_precision, PrecisionMode::BFloat16);
    }

    #[test]
    fn test_mixed_precision_config_fp32() {
        let config = MixedPrecisionConfig::fp32();
        assert_eq!(config.compute_precision, PrecisionMode::Full);
        assert_eq!(config.master_precision, PrecisionMode::Full);
        assert!(!config.dynamic_loss_scale);
        assert_eq!(config.loss_scale, 1.0);
    }

    #[test]
    fn test_convert_precision() {
        let device = Device::Cpu;
        let tensor = Tensor::ones((2, 3), DType::F32, &device).unwrap();

        // Convert to FP16
        let fp16 = convert_precision(&tensor, PrecisionMode::Half).unwrap();
        assert_eq!(fp16.dtype(), DType::F16);

        // Convert to BF16
        let bf16 = convert_precision(&tensor, PrecisionMode::BFloat16).unwrap();
        assert_eq!(bf16.dtype(), DType::BF16);

        // Convert to same precision should work
        let same = convert_precision(&tensor, PrecisionMode::Full).unwrap();
        assert_eq!(same.dtype(), DType::F32);
    }

    #[test]
    fn test_scale_loss() {
        let device = Device::Cpu;
        let loss = Tensor::full(2.0f32, (), &device).unwrap(); // scalar tensor

        let mut config = MixedPrecisionConfig::default();
        config.loss_scale = 4.0;

        let scaled = scale_loss(&loss, &config).unwrap();
        let value: f32 = scaled.to_scalar().unwrap();

        assert!((value - 8.0).abs() < 1e-5);
    }

    #[test]
    fn test_unscale_gradients() {
        let device = Device::Cpu;
        let grad1 = Tensor::full(8.0f32, (2, 2), &device).unwrap();
        let grad2 = Tensor::full(16.0f32, (2, 2), &device).unwrap();

        let gradients = vec![grad1, grad2];

        let mut config = MixedPrecisionConfig::default();
        config.loss_scale = 4.0;

        let unscaled = unscale_gradients(&gradients, &config).unwrap();

        // Check first gradient: 8.0 / 4.0 = 2.0
        let vals1: Vec<f32> = unscaled[0].flatten_all().unwrap().to_vec1().unwrap();
        for val in vals1 {
            assert!((val - 2.0).abs() < 1e-5);
        }

        // Check second gradient: 16.0 / 4.0 = 4.0
        let vals2: Vec<f32> = unscaled[1].flatten_all().unwrap().to_vec1().unwrap();
        for val in vals2 {
            assert!((val - 4.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_has_inf_or_nan() {
        let device = Device::Cpu;

        // Test normal gradients
        let grad1 = Tensor::ones((2, 2), DType::F32, &device).unwrap();
        let grad2 = Tensor::full(2.0f32, (2, 2), &device).unwrap();
        assert!(!has_inf_or_nan(&[grad1, grad2]).unwrap());

        // Test with NaN
        let nan_grad = Tensor::full(f32::NAN, (2, 2), &device).unwrap();
        assert!(has_inf_or_nan(&[nan_grad]).unwrap());

        // Test with Inf
        let inf_grad = Tensor::full(f32::INFINITY, (2, 2), &device).unwrap();
        assert!(has_inf_or_nan(&[inf_grad]).unwrap());
    }

    #[test]
    fn test_update_loss_scale_on_overflow() {
        let mut config = MixedPrecisionConfig {
            loss_scale: 1000.0,
            scale_backoff_factor: 0.5,
            ..Default::default()
        };

        // Test backoff on overflow
        let new_scale = update_loss_scale(&mut config, true, 0);
        assert_eq!(new_scale, 500.0);
        assert_eq!(config.loss_scale, 500.0);
    }

    #[test]
    fn test_update_loss_scale_growth() {
        let mut config = MixedPrecisionConfig {
            loss_scale: 100.0,
            scale_growth_factor: 2.0,
            scale_growth_interval: 100,
            ..Default::default()
        };

        // Test growth after many successful steps
        let new_scale = update_loss_scale(&mut config, false, 100);
        assert_eq!(new_scale, 200.0);
        assert_eq!(config.loss_scale, 200.0);
    }

    #[test]
    fn test_update_loss_scale_no_change() {
        let mut config = MixedPrecisionConfig::default();
        config.loss_scale = 100.0;

        // No change if not enough steps and no overflow
        let new_scale = update_loss_scale(&mut config, false, 10);
        assert_eq!(new_scale, 100.0);
    }

    #[test]
    fn test_update_loss_scale_bounds() {
        let mut config = MixedPrecisionConfig {
            min_loss_scale: 1.0,
            max_loss_scale: 1000.0,
            loss_scale: 2.0,
            scale_backoff_factor: 0.5,
            ..Default::default()
        };

        // Test min bound
        update_loss_scale(&mut config, true, 0);
        assert!((config.loss_scale - 1.0).abs() < f32::EPSILON); // Should hit min

        // Test max bound
        config.loss_scale = 600.0;
        config.scale_growth_factor = 2.0;
        config.scale_growth_interval = 10;
        update_loss_scale(&mut config, false, 10);
        assert!((config.loss_scale - 1000.0).abs() < f32::EPSILON); // Should hit max
    }

    #[test]
    fn test_scale_gradients() {
        let device = Device::Cpu;
        let grad1 = Tensor::ones((2, 3), DType::F32, &device).unwrap();
        let grad2 = Tensor::full(2.0f32, (2, 3), &device).unwrap();

        let gradients = vec![grad1, grad2];
        let scale = 0.5;

        let scaled = scale_gradients(&gradients, scale).unwrap();

        // Check first gradient: 1.0 * 0.5 = 0.5
        let vals1: Vec<f32> = scaled[0].flatten_all().unwrap().to_vec1().unwrap();
        for val in vals1 {
            assert!((val - 0.5).abs() < 1e-5);
        }

        // Check second gradient: 2.0 * 0.5 = 1.0
        let vals2: Vec<f32> = scaled[1].flatten_all().unwrap().to_vec1().unwrap();
        for val in vals2 {
            assert!((val - 1.0).abs() < 1e-5);
        }
    }
}
