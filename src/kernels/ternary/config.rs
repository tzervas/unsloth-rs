// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! Configuration for ternary bitsliced operations.
//!
//! This module provides compile-time and runtime configuration for
//! ternary kernels, including sparsity thresholds, tile sizes, and
//! calibration parameters.

/// Configuration for ternary bitsliced operations.
///
/// # Example
///
/// ```rust
/// use unsloth_rs::kernels::ternary::TernaryConfig;
///
/// // Default configuration (good for most cases)
/// let config = TernaryConfig::default();
///
/// // Tuned for extremely sparse weights (99%+ zeros)
/// let config = TernaryConfig::for_sparse_model();
/// ```
#[derive(Debug, Clone, Copy)]
pub struct TernaryConfig {
    /// Minimum sparsity (fraction of zeros) to enable ternary path.
    /// Below this threshold, falls back to FP computation.
    /// Range: 0.0 to 1.0 (default: 0.8 = 80% zeros)
    pub sparsity_threshold: f32,

    /// Tile size for matmul kernel (must be power of 2).
    /// Larger tiles reduce memory transactions but use more shared mem.
    pub tile_size: u32,

    /// Block size (threads per block) for CUDA kernels.
    /// Must be warp-aligned (multiple of 32), max 1024.
    pub block_size: u32,

    /// Enable plane skipping optimization.
    /// When true, kernel skips computation for all-zero planes.
    pub enable_plane_skipping: bool,

    /// Enable dimension-level sparsity metadata.
    /// Adds bitmap tracking which dimensions are active.
    pub enable_dim_metadata: bool,

    /// Chunk size for sparsity metadata (dimensions per bitmap).
    /// Must be power of 2. Each chunk gets a 64-bit activity bitmap.
    pub metadata_chunk_size: u32,

    /// Quantization threshold for ternary conversion.
    /// Values with |x| < threshold become 0.
    /// None means auto-calibrate via abs-max or percentile.
    pub quantization_threshold: Option<f32>,

    /// Scale calibration method for quantization.
    /// See [`CalibrationMethod`] for options.
    pub calibration_method: CalibrationMethodConfig,
}

/// Scale calibration method for ternary quantization.
#[derive(Debug, Clone, Copy, Default)]
pub enum CalibrationMethodConfig {
    /// Use max absolute value per channel (fast, may lose precision).
    #[default]
    AbsMax,

    /// Use percentile of absolute values (more robust to outliers).
    /// Parameter is percentile (e.g., 99.9).
    Percentile(f32),

    /// Use mean + k*std for threshold (statistical approach).
    /// Parameter is k (typically 1.0-3.0).
    MeanStd(f32),

    /// Manual threshold (fixed value across all channels).
    Manual(f32),
}

impl Default for TernaryConfig {
    fn default() -> Self {
        Self {
            sparsity_threshold: 0.8,
            tile_size: 64,
            block_size: 256,
            enable_plane_skipping: true,
            enable_dim_metadata: true,
            metadata_chunk_size: 2048,
            quantization_threshold: None,
            calibration_method: CalibrationMethodConfig::default(),
        }
    }
}

impl TernaryConfig {
    /// Configuration optimized for sparse pruned models (95%+ zeros).
    ///
    /// Uses aggressive plane skipping and larger metadata chunks.
    #[must_use]
    pub fn for_sparse_model() -> Self {
        Self {
            sparsity_threshold: 0.90,
            tile_size: 128,
            block_size: 256,
            enable_plane_skipping: true,
            enable_dim_metadata: true,
            metadata_chunk_size: 4096,
            quantization_threshold: None,
            calibration_method: CalibrationMethodConfig::Percentile(99.5),
        }
    }

    /// Configuration for dense models (moderate sparsity).
    ///
    /// Disables aggressive optimizations that have overhead on denser weights.
    #[must_use]
    pub fn for_dense_model() -> Self {
        Self {
            sparsity_threshold: 0.5,
            tile_size: 64,
            block_size: 256,
            enable_plane_skipping: false,
            enable_dim_metadata: false,
            metadata_chunk_size: 2048,
            quantization_threshold: None,
            calibration_method: CalibrationMethodConfig::AbsMax,
        }
    }

    /// Configuration for RTX 5080 (larger shared memory).
    #[must_use]
    pub fn for_rtx_5080() -> Self {
        Self {
            tile_size: 128,
            block_size: 256,
            ..Self::default()
        }
    }

    /// Configuration for A100/H100 datacenter GPUs.
    #[must_use]
    pub fn for_datacenter() -> Self {
        Self {
            tile_size: 256,
            block_size: 256,
            ..Self::default()
        }
    }

    /// Validate configuration parameters.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - `tile_size` is not a power of 2
    /// - `block_size` is not warp-aligned or exceeds 1024
    /// - `metadata_chunk_size` is not a power of 2
    pub fn validate(&self) -> Result<(), ConfigError> {
        if !self.tile_size.is_power_of_two() {
            return Err(ConfigError::InvalidTileSize(self.tile_size));
        }
        if !self.block_size.is_multiple_of(32) || self.block_size > 1024 {
            return Err(ConfigError::InvalidBlockSize(self.block_size));
        }
        if !self.metadata_chunk_size.is_power_of_two() {
            return Err(ConfigError::InvalidChunkSize(self.metadata_chunk_size));
        }
        if !(0.0..=1.0).contains(&self.sparsity_threshold) {
            return Err(ConfigError::InvalidSparsityThreshold(
                self.sparsity_threshold,
            ));
        }
        Ok(())
    }

    /// Calculate shared memory required for matmul kernel (bytes).
    #[must_use]
    pub const fn shared_memory_bytes(&self) -> u32 {
        // Two planes (+ and -) × tile_size × sizeof(u32)
        2 * self.tile_size * 4
    }

    /// Calculate number of u32 words for K dimension.
    #[must_use]
    pub const fn k_words(k_dim: u32) -> u32 {
        k_dim.div_ceil(32)
    }
}

/// Configuration validation errors.
#[derive(Debug, Clone)]
pub enum ConfigError {
    /// Tile size must be power of 2.
    InvalidTileSize(u32),
    /// Block size must be warp-aligned (×32) and ≤1024.
    InvalidBlockSize(u32),
    /// Metadata chunk size must be power of 2.
    InvalidChunkSize(u32),
    /// Sparsity threshold must be in [0.0, 1.0].
    InvalidSparsityThreshold(f32),
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidTileSize(v) => write!(f, "tile_size {v} must be power of 2"),
            Self::InvalidBlockSize(v) => {
                write!(f, "block_size {v} must be multiple of 32 and ≤1024")
            }
            Self::InvalidChunkSize(v) => write!(f, "metadata_chunk_size {v} must be power of 2"),
            Self::InvalidSparsityThreshold(v) => {
                write!(f, "sparsity_threshold {v} must be in [0.0, 1.0]")
            }
        }
    }
}

impl std::error::Error for ConfigError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_valid() {
        let config = TernaryConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_sparse_config_valid() {
        let config = TernaryConfig::for_sparse_model();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_invalid_tile_size() {
        let config = TernaryConfig {
            tile_size: 65, // Not power of 2
            ..Default::default()
        };
        assert!(matches!(
            config.validate(),
            Err(ConfigError::InvalidTileSize(65))
        ));
    }

    #[test]
    fn test_invalid_block_size() {
        let config = TernaryConfig {
            block_size: 100, // Not warp-aligned
            ..Default::default()
        };
        assert!(matches!(
            config.validate(),
            Err(ConfigError::InvalidBlockSize(100))
        ));
    }

    #[test]
    fn test_k_words_calculation() {
        assert_eq!(TernaryConfig::k_words(32), 1);
        assert_eq!(TernaryConfig::k_words(33), 2);
        assert_eq!(TernaryConfig::k_words(64), 2);
        assert_eq!(TernaryConfig::k_words(128), 4);
    }
}
