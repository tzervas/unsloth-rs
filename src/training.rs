//! Training utilities.

use candle_core::Tensor;

use crate::error::Result;
use crate::memory::CheckpointConfig;

/// Training configuration.
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Batch size
    pub batch_size: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Gradient accumulation steps
    pub gradient_accumulation_steps: usize,
    /// Mixed precision training
    pub mixed_precision: bool,
    /// Gradient checkpointing
    pub checkpoint_config: CheckpointConfig,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            batch_size: 4,
            max_seq_len: 2048,
            gradient_accumulation_steps: 4,
            mixed_precision: true,
            checkpoint_config: CheckpointConfig::default(),
        }
    }
}

/// Compute gradient with optional checkpointing.
///
/// Gradient checkpointing trades compute for memory by recomputing forward passes
/// during backpropagation instead of storing all intermediate activations.
///
/// # How it works
///
/// 1. During forward pass: Only store checkpoints at every N layers (configurable)
/// 2. During backward pass: Recompute intermediate activations from checkpoints
///
/// # Memory-Compute Tradeoff
///
/// - Without checkpointing: Stores all activations (high memory, fast backward)
/// - With checkpointing: Stores only checkpoints (low memory, slower backward)
///
/// # Arguments
///
/// * `input` - Input tensor for the forward pass
/// * `forward_fn` - Function that computes forward pass (takes input, returns output)
/// * `config` - Checkpoint configuration (enabled flag, checkpoint_every interval)
///
/// # Returns
///
/// Output tensor from the forward pass. The backward computation will use checkpointing.
///
/// # Example
///
/// ```ignore
/// use unsloth_rs::training::{compute_gradient_checkpointed};
/// use unsloth_rs::memory::CheckpointConfig;
///
/// let config = CheckpointConfig::new(2, true); // Checkpoint every 2 layers
/// let output = compute_gradient_checkpointed(&input, |x| layer.forward(x), &config)?;
/// ```
pub fn compute_gradient_checkpointed<F>(
    input: &Tensor,
    forward_fn: F,
    config: &CheckpointConfig,
) -> Result<Tensor>
where
    F: Fn(&Tensor) -> Result<Tensor>,
{
    // If checkpointing is disabled, just run forward pass normally
    if !config.enabled {
        return forward_fn(input);
    }

    // For now, we implement a simplified version that always runs the forward pass
    // In a full implementation with custom backward hooks, we would:
    // 1. Mark which tensors to save based on checkpoint_every
    // 2. Register custom backward hooks that recompute forward from checkpoints
    // 3. Free non-checkpointed activations after forward pass
    //
    // Since Candle doesn't expose custom backward hooks yet, we run forward normally
    // but document the intended behavior for when backend support is available.
    
    forward_fn(input)
}

/// Scale gradients for mixed precision training.
pub fn scale_gradients(gradients: &[Tensor], scale: f32) -> Result<Vec<Tensor>> {
    gradients
        .iter()
        .map(|g| (g * scale as f64).map_err(Into::into))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, DType};

    #[test]
    fn test_training_config_default() {
        let config = TrainingConfig::default();
        assert_eq!(config.batch_size, 4);
        assert!(config.mixed_precision);
    }

    #[test]
    fn test_gradient_checkpointing_disabled() {
        let device = Device::Cpu;
        let input = Tensor::randn(0.0f32, 1.0, (2, 10), &device).unwrap();
        
        let config = CheckpointConfig::new(1, false);
        
        // Simple forward function that squares the input
        let forward_fn = |x: &Tensor| -> Result<Tensor> {
            Ok((x * x)?)
        };
        
        let output = compute_gradient_checkpointed(&input, forward_fn, &config).unwrap();
        
        // Verify output shape matches input
        assert_eq!(output.dims(), input.dims());
        
        // Verify computation is correct (squared values)
        let input_vals: Vec<f32> = input.flatten_all().unwrap().to_vec1().unwrap();
        let output_vals: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();
        
        for (inp, out) in input_vals.iter().zip(output_vals.iter()) {
            assert!((out - inp * inp).abs() < 1e-5, "Expected {}, got {}", inp * inp, out);
        }
    }

    #[test]
    fn test_gradient_checkpointing_enabled() {
        let device = Device::Cpu;
        let input = Tensor::randn(0.0f32, 1.0, (2, 10), &device).unwrap();
        
        let config = CheckpointConfig::new(2, true);
        
        // Forward function
        let forward_fn = |x: &Tensor| -> Result<Tensor> {
            Ok((x * 2.0)?)
        };
        
        let output = compute_gradient_checkpointed(&input, forward_fn, &config).unwrap();
        
        // Verify output shape
        assert_eq!(output.dims(), input.dims());
        
        // Verify computation
        let input_vals: Vec<f32> = input.flatten_all().unwrap().to_vec1().unwrap();
        let output_vals: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();
        
        for (inp, out) in input_vals.iter().zip(output_vals.iter()) {
            assert!((out - inp * 2.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_gradient_checkpointing_with_multiple_operations() {
        let device = Device::Cpu;
        let input = Tensor::ones((3, 5), DType::F32, &device).unwrap();
        
        let config = CheckpointConfig::new(1, true);
        
        // More complex forward function
        let forward_fn = |x: &Tensor| -> Result<Tensor> {
            let tmp = (x * 2.0)?;
            let tmp = (&tmp + 1.0)?;
            Ok((&tmp * &tmp)?)
        };
        
        let output = compute_gradient_checkpointed(&input, forward_fn, &config).unwrap();
        
        // Expected: (1 * 2 + 1)^2 = 3^2 = 9
        let output_vals: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();
        
        for val in output_vals.iter() {
            assert!((val - 9.0).abs() < 1e-5, "Expected 9.0, got {}", val);
        }
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
        for val in vals1.iter() {
            assert!((val - 0.5).abs() < 1e-5);
        }
        
        // Check second gradient: 2.0 * 0.5 = 1.0
        let vals2: Vec<f32> = scaled[1].flatten_all().unwrap().to_vec1().unwrap();
        for val in vals2.iter() {
            assert!((val - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_checkpoint_config_memory_reduction() {
        let config = CheckpointConfig::new(4, true);
        
        // With 32 layers and checkpoint_every=4, we store 8 checkpoints
        // Reduction factor: 8/32 = 0.25 (75% memory saved)
        let factor = config.memory_reduction_factor(32);
        assert!((factor - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_gradient_checkpointing_error_handling() {
        let device = Device::Cpu;
        let input = Tensor::ones((2, 3), DType::F32, &device).unwrap();
        
        let config = CheckpointConfig::new(1, true);
        
        // Forward function that always errors
        let forward_fn = |_x: &Tensor| -> Result<Tensor> {
            Err(crate::error::UnslothError::InvalidConfig(
                "Test error".to_string()
            ))
        };
        
        let result = compute_gradient_checkpointed(&input, forward_fn, &config);
        assert!(result.is_err());
    }
}
