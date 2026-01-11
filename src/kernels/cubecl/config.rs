// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! Kernel configuration for Flash Attention.
//!
//! This module contains compile-time and runtime configuration for the
//! Flash Attention `CubeCL` kernel, including tile sizes and launch parameters.

/// Configuration for Flash Attention kernel launch.
///
/// Tile sizes are tuned per-GPU for optimal performance:
/// - RTX 5080: `tile_size=256` (primary target)
/// - RTX 3090 Ti: `tile_size=128` (validation target)
/// - A100/H100: `tile_size=256` or larger
///
/// # Example
///
/// ```rust
/// use unsloth_rs::kernels::cubecl::FlashAttentionConfig;
///
/// // Use defaults (tile_size=128, good for most GPUs)
/// let config = FlashAttentionConfig::default();
///
/// // Tune for specific GPU
/// let config = FlashAttentionConfig::for_rtx_5080();
/// ```
#[derive(Debug, Clone, Copy)]
pub struct FlashAttentionConfig {
    /// Tile size for Q/K/V blocks (typically 128 or 256).
    /// Must be a power of 2. Larger tiles increase shared memory usage
    /// but reduce global memory accesses.
    pub tile_size: u32,

    /// Head dimension (e.g., 64, 128).
    /// Used to calculate shared memory requirements.
    pub head_dim: u32,

    /// Number of threads per block (warp-aligned, max 1024).
    /// Typically 128-256 for attention kernels.
    pub block_size: u32,

    /// Whether to use vectorized loads (4-element `Line<F>`).
    /// Enables 128-bit coalesced memory transactions.
    pub use_vectorized_loads: bool,

    /// Enable causal masking (upper triangular mask).
    pub causal_mask: bool,
}

impl Default for FlashAttentionConfig {
    fn default() -> Self {
        Self {
            tile_size: 128,
            head_dim: 64,
            block_size: 256,
            use_vectorized_loads: true,
            causal_mask: false,
        }
    }
}

impl FlashAttentionConfig {
    /// Configuration optimized for RTX 5080 (primary development target).
    ///
    /// RTX 5080 has larger shared memory, allowing for bigger tiles.
    #[must_use]
    pub fn for_rtx_5080() -> Self {
        Self {
            tile_size: 256,
            head_dim: 64,
            block_size: 256,
            use_vectorized_loads: true,
            causal_mask: false,
        }
    }

    /// Configuration optimized for RTX 3090 Ti (validation target).
    #[must_use]
    pub fn for_rtx_3090_ti() -> Self {
        Self {
            tile_size: 128,
            head_dim: 64,
            block_size: 256,
            use_vectorized_loads: true,
            causal_mask: false,
        }
    }

    /// Configuration for A100/H100 datacenter GPUs.
    #[must_use]
    pub fn for_datacenter() -> Self {
        Self {
            tile_size: 256,
            head_dim: 128,
            block_size: 256,
            use_vectorized_loads: true,
            causal_mask: false,
        }
    }

    /// Enable causal (autoregressive) masking.
    #[must_use]
    pub const fn with_causal_mask(mut self) -> Self {
        self.causal_mask = true;
        self
    }

    /// Set custom tile size.
    ///
    /// # Panics
    ///
    /// Panics if `tile_size` is not a power of 2.
    #[must_use]
    pub fn with_tile_size(mut self, tile_size: u32) -> Self {
        assert!(tile_size.is_power_of_two(), "tile_size must be power of 2");
        self.tile_size = tile_size;
        self
    }

    /// Set head dimension.
    #[must_use]
    pub const fn with_head_dim(mut self, head_dim: u32) -> Self {
        self.head_dim = head_dim;
        self
    }

    /// Calculate shared memory required per block in bytes.
    ///
    /// Layout:
    /// - Q tile: `tile_size × head_dim`
    /// - K tile: `tile_size × head_dim`
    /// - V tile: `tile_size × head_dim`
    /// - Scores tile: `tile_size × tile_size`
    /// - Statistics: `tile_size × 2` (running max and sum)
    #[must_use]
    pub const fn shared_memory_bytes(&self, bytes_per_elem: usize) -> usize {
        let tile = self.tile_size as usize;
        let dim = self.head_dim as usize;

        let qkv_tiles = 3 * tile * dim;
        let scores = tile * tile;
        let stats = 2 * tile;

        (qkv_tiles + scores + stats) * bytes_per_elem
    }

    /// Calculate number of Q tiles for a given sequence length.
    #[must_use]
    pub const fn num_q_tiles(&self, seq_len: u32) -> u32 {
        seq_len.div_ceil(self.tile_size)
    }

    /// Calculate number of KV tiles for a given sequence length.
    #[must_use]
    pub const fn num_kv_tiles(&self, seq_len: u32) -> u32 {
        seq_len.div_ceil(self.tile_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = FlashAttentionConfig::default();
        assert_eq!(config.tile_size, 128);
        assert_eq!(config.block_size, 256);
        assert!(config.use_vectorized_loads);
        assert!(!config.causal_mask);
    }

    #[test]
    fn test_shared_memory_calculation() {
        let config = FlashAttentionConfig::default();
        let bytes = config.shared_memory_bytes(4); // f32

        // Expected: (3 * 128 * 64 + 128 * 128 + 2 * 128) * 4
        // = (24576 + 16384 + 256) * 4 = 164864 bytes
        assert_eq!(bytes, 164_864);
    }

    #[test]
    fn test_num_tiles() {
        let config = FlashAttentionConfig::default(); // tile_size=128

        assert_eq!(config.num_q_tiles(512), 4);
        assert_eq!(config.num_q_tiles(1024), 8);
        assert_eq!(config.num_q_tiles(1025), 9); // Ceiling division
    }

    #[test]
    fn test_builder_pattern() {
        let config = FlashAttentionConfig::default()
            .with_tile_size(256)
            .with_head_dim(128)
            .with_causal_mask();

        assert_eq!(config.tile_size, 256);
        assert_eq!(config.head_dim, 128);
        assert!(config.causal_mask);
    }

    #[test]
    #[should_panic(expected = "tile_size must be power of 2")]
    fn test_invalid_tile_size() {
        let _ = FlashAttentionConfig::default().with_tile_size(100);
    }
}
