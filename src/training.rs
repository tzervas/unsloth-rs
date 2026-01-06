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
    unimplemented!("Gradient checkpointing not yet implemented")
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

    #[test]
    fn test_training_config_default() {
        let config = TrainingConfig::default();
        assert_eq!(config.batch_size, 4);
        assert!(config.mixed_precision);
    }
}
