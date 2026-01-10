// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! Memory management utilities for tracking allocations.
//!
//! This module provides tools for memory tracking and estimation, which are
//! essential for managing GPU memory when training large language models.
//!
//! ## Why Memory Management?
//!
//! Large language models can easily exhaust GPU memory. These utilities help:
//! - Estimate memory requirements before running operations
//! - Track actual allocations during execution
//! - Configure gradient checkpointing to trade compute for memory
//!
//! ## Provided Utilities
//!
//! - `MemoryPool`: Tracks allocations with optional limit enforcement
//! - `CheckpointConfig`: Configuration for gradient checkpointing
//! - `estimate_forward_memory`: Estimates memory for forward passes
//! - `estimate_attention_vram`: Estimates memory for attention operations
//! - `format_bytes`: Human-readable byte formatting

use crate::error::{Result, UnslothError};

/// Memory pool for efficient GPU allocation.
///
/// Tracks memory allocations and provides limit enforcement.
/// Future versions will integrate with `CubeCL` for device-aware allocation.
pub struct MemoryPool {
    /// Total allocated bytes
    allocated: usize,
    /// Peak memory usage
    peak: usize,
    /// Memory limit (if set)
    limit: Option<usize>,
    /// Device type for allocation tracking
    device_type: DeviceType,
}

/// Device type for memory tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeviceType {
    /// CPU memory
    #[default]
    Cpu,
    /// CUDA GPU memory
    Cuda,
    /// Metal GPU memory (Apple)
    Metal,
    /// Vulkan GPU memory
    Vulkan,
}

impl MemoryPool {
    /// Create a new memory pool.
    #[must_use]
    pub fn new(limit: Option<usize>) -> Self {
        Self {
            allocated: 0,
            peak: 0,
            limit,
            device_type: DeviceType::default(),
        }
    }

    /// Create a new memory pool for a specific device.
    #[must_use]
    pub fn with_device(limit: Option<usize>, device_type: DeviceType) -> Self {
        Self {
            allocated: 0,
            peak: 0,
            limit,
            device_type,
        }
    }

    /// Request memory allocation.
    ///
    /// # Errors
    /// Returns `OutOfMemory` if limit would be exceeded.
    pub fn allocate(&mut self, bytes: usize) -> Result<()> {
        let new_total = self.allocated + bytes;

        if let Some(limit) = self.limit {
            if new_total > limit {
                return Err(UnslothError::OutOfMemory {
                    required: new_total,
                    available: limit.saturating_sub(self.allocated),
                });
            }
        }

        self.allocated = new_total;
        self.peak = self.peak.max(self.allocated);
        Ok(())
    }

    /// Free memory.
    pub fn free(&mut self, bytes: usize) {
        self.allocated = self.allocated.saturating_sub(bytes);
    }

    /// Get current allocation.
    #[must_use]
    pub fn allocated(&self) -> usize {
        self.allocated
    }

    /// Get peak allocation.
    #[must_use]
    pub fn peak(&self) -> usize {
        self.peak
    }

    /// Get the device type.
    #[must_use]
    pub fn device_type(&self) -> DeviceType {
        self.device_type
    }

    /// Reset peak tracking.
    pub fn reset_peak(&mut self) {
        self.peak = self.allocated;
    }

    /// Calculate memory efficiency (allocated vs peak).
    #[must_use]
    pub fn efficiency(&self) -> f64 {
        if self.peak == 0 {
            1.0
        } else {
            // Precision loss acceptable for efficiency metric
            #[allow(clippy::cast_precision_loss)]
            {
                self.allocated as f64 / self.peak as f64
            }
        }
    }
}

/// Gradient checkpointing configuration.
///
/// Controls how activations are stored during forward pass.
/// Higher `checkpoint_every` values reduce memory but increase compute.
#[derive(Debug, Clone)]
pub struct CheckpointConfig {
    /// Checkpoint every N layers
    pub checkpoint_every: usize,
    /// Enable checkpointing
    pub enabled: bool,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            checkpoint_every: 1,
            enabled: true,
        }
    }
}

impl CheckpointConfig {
    /// Create a new checkpoint config.
    #[must_use]
    pub fn new(checkpoint_every: usize, enabled: bool) -> Self {
        Self {
            checkpoint_every,
            enabled,
        }
    }

    /// Calculate expected memory reduction factor.
    ///
    /// Returns a value between 0 and 1, where lower is better.
    #[must_use]
    pub fn memory_reduction_factor(&self, num_layers: usize) -> f64 {
        if !self.enabled || num_layers == 0 {
            1.0
        } else {
            let checkpointed = num_layers.div_ceil(self.checkpoint_every);
            // Precision loss acceptable for memory reduction factor metric
            #[allow(clippy::cast_precision_loss)]
            {
                checkpointed as f64 / num_layers as f64
            }
        }
    }
}

/// Calculate memory requirements for a forward pass.
///
/// # Arguments
/// * `batch_size` - Batch size
/// * `seq_len` - Sequence length
/// * `hidden_size` - Hidden dimension size
/// * `num_layers` - Number of transformer layers
/// * `checkpoint_config` - Gradient checkpointing configuration
///
/// # Returns
/// Estimated memory usage in bytes
#[must_use]
pub fn estimate_forward_memory(
    batch_size: usize,
    seq_len: usize,
    hidden_size: usize,
    num_layers: usize,
    checkpoint_config: &CheckpointConfig,
) -> usize {
    let bytes_per_elem = 4; // f32

    // Per-layer activation memory
    let activation_per_layer = batch_size * seq_len * hidden_size * bytes_per_elem;

    // With checkpointing, only store every N layers
    let stored_layers = if checkpoint_config.enabled {
        num_layers.div_ceil(checkpoint_config.checkpoint_every)
    } else {
        num_layers
    };

    stored_layers * activation_per_layer
}

/// Estimate VRAM for attention operation.
///
/// # Arguments
/// * `batch_size` - Batch size
/// * `seq_len` - Sequence length
/// * `hidden_size` - Hidden dimension
/// * `num_heads` - Number of attention heads
///
/// # Returns
/// Estimated VRAM in bytes
#[must_use]
pub fn estimate_attention_vram(
    batch_size: usize,
    seq_len: usize,
    hidden_size: usize,
    num_heads: usize,
) -> usize {
    let bytes_per_elem = 4; // f32

    // QKV projection output
    let qkv_size = batch_size * seq_len * 3 * hidden_size * bytes_per_elem;
    // Attention scores: [batch, num_heads, seq_len, seq_len]
    let scores_size = batch_size * num_heads * seq_len * seq_len * bytes_per_elem;
    // Output
    let output_size = batch_size * seq_len * hidden_size * bytes_per_elem;

    qkv_size + scores_size + output_size
}

/// Format bytes as human-readable string.
#[must_use]
pub fn format_bytes(bytes: usize) -> String {
    const KB: usize = 1024;
    const MB: usize = KB * 1024;
    const GB: usize = MB * 1024;

    // Precision loss acceptable for human-readable byte formatting
    #[allow(clippy::cast_precision_loss)]
    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} bytes")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool_allocation() {
        let mut pool = MemoryPool::new(Some(1000));

        assert!(pool.allocate(500).is_ok());
        assert_eq!(pool.allocated(), 500);

        assert!(pool.allocate(400).is_ok());
        assert_eq!(pool.allocated(), 900);

        // Should fail - would exceed limit
        assert!(pool.allocate(200).is_err());

        pool.free(300);
        assert_eq!(pool.allocated(), 600);
    }

    #[test]
    fn test_memory_pool_with_device() {
        let pool = MemoryPool::with_device(Some(1024 * 1024), DeviceType::Cuda);
        assert_eq!(pool.device_type(), DeviceType::Cuda);
        assert_eq!(pool.allocated(), 0);
    }

    #[test]
    fn test_checkpoint_memory_reduction() {
        let batch = 4;
        let seq = 2048;
        let hidden = 4096;
        let layers = 32;

        let no_checkpoint = CheckpointConfig {
            enabled: false,
            ..Default::default()
        };
        let with_checkpoint = CheckpointConfig {
            enabled: true,
            checkpoint_every: 4,
        };

        let mem_full = estimate_forward_memory(batch, seq, hidden, layers, &no_checkpoint);
        let mem_checkpoint = estimate_forward_memory(batch, seq, hidden, layers, &with_checkpoint);

        // Checkpointing should reduce memory significantly
        assert!(mem_checkpoint < mem_full / 2);
    }

    #[test]
    fn test_checkpoint_reduction_factor() {
        let config = CheckpointConfig::new(4, true);
        let factor = config.memory_reduction_factor(32);
        // 32 layers, checkpoint every 4 = 8 checkpoints = 8/32 = 0.25
        assert!((factor - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(500), "500 bytes");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.00 MB");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.00 GB");
    }

    #[test]
    fn test_attention_vram_estimate() {
        let vram = estimate_attention_vram(4, 2048, 4096, 32);
        // Should be substantial but not unreasonable
        assert!(vram > 100 * 1024 * 1024); // > 100 MB
        assert!(vram < 10 * 1024 * 1024 * 1024); // < 10 GB
    }
}
