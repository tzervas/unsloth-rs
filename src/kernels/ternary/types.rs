//! Core types for ternary bitsliced representation.
//!
//! This module defines the fundamental data structures for storing
//! ternary {-1, 0, +1} weights in a bitsliced format.
//!
//! ## Representation
//!
//! Each ternary value is stored as two bits across two planes:
//!
//! ```text
//! Value | +plane | -plane
//! ------+--------+-------
//!   +1  |   1    |   0
//!    0  |   0    |   0
//!   -1  |   0    |   1
//! ```
//!
//! This representation ensures mutual exclusion (never both set) and
//! enables efficient popcount-based operations.

use super::config::TernaryConfig;
use candle_core::{Device, Shape, Tensor};

/// Bitsliced planes for ternary representation.
///
/// Each plane is a `Vec<u32>` where each bit represents one dimension.
/// A dimension is packed as `dim_index / 32` for array index and
/// `dim_index % 32` for bit position.
#[derive(Debug, Clone)]
pub struct TernaryPlanes {
    /// Positive plane: bit set if value is +1.
    pub plus: Vec<u32>,

    /// Negative plane: bit set if value is -1.
    pub minus: Vec<u32>,

    /// Number of logical dimensions (may be padded to multiple of 32).
    pub num_dims: usize,
}

impl TernaryPlanes {
    /// Create new ternary planes with given capacity.
    ///
    /// # Arguments
    ///
    /// * `num_dims` - Number of logical dimensions
    ///
    /// # Returns
    ///
    /// Zeroed planes (all values implicitly 0).
    #[must_use]
    pub fn new(num_dims: usize) -> Self {
        let num_words = num_dims.div_ceil(32);
        Self {
            plus: vec![0u32; num_words],
            minus: vec![0u32; num_words],
            num_dims,
        }
    }

    /// Get the number of u32 words used.
    #[must_use]
    pub fn num_words(&self) -> usize {
        self.plus.len()
    }

    /// Set a dimension to a ternary value.
    ///
    /// # Arguments
    ///
    /// * `dim` - Dimension index (0-based)
    /// * `value` - Ternary value: -1, 0, or +1
    ///
    /// # Panics
    ///
    /// Panics if `dim >= num_dims` or `value` is not in {-1, 0, +1}.
    pub fn set(&mut self, dim: usize, value: i8) {
        assert!(dim < self.num_dims, "dimension out of bounds");
        assert!(
            (-1..=1).contains(&value),
            "value must be -1, 0, or +1, got {value}"
        );

        let word_idx = dim / 32;
        let bit_idx = dim % 32;
        let mask = 1u32 << bit_idx;

        // Clear both planes first
        self.plus[word_idx] &= !mask;
        self.minus[word_idx] &= !mask;

        // Set appropriate plane
        match value {
            1 => self.plus[word_idx] |= mask,
            -1 => self.minus[word_idx] |= mask,
            0 => {} // Already cleared
            _ => unreachable!(),
        }
    }

    /// Get the ternary value at a dimension.
    ///
    /// # Arguments
    ///
    /// * `dim` - Dimension index (0-based)
    ///
    /// # Returns
    ///
    /// The ternary value: -1, 0, or +1.
    ///
    /// # Panics
    ///
    /// Panics if `dim >= num_dims`.
    #[must_use]
    pub fn get(&self, dim: usize) -> i8 {
        assert!(dim < self.num_dims, "dimension out of bounds");

        let word_idx = dim / 32;
        let bit_idx = dim % 32;
        let mask = 1u32 << bit_idx;

        let is_plus = (self.plus[word_idx] & mask) != 0;
        let is_minus = (self.minus[word_idx] & mask) != 0;

        debug_assert!(
            !(is_plus && is_minus),
            "invalid state: both planes set at dim {dim}"
        );

        if is_plus {
            1
        } else if is_minus {
            -1
        } else {
            0
        }
    }

    /// Count non-zero elements (sparsity check).
    #[must_use]
    pub fn count_nonzero(&self) -> usize {
        let plus_count: u32 = self.plus.iter().map(|w| w.count_ones()).sum();
        let minus_count: u32 = self.minus.iter().map(|w| w.count_ones()).sum();
        (plus_count + minus_count) as usize
    }

    /// Calculate sparsity (fraction of zeros).
    #[must_use]
    pub fn sparsity(&self) -> f32 {
        // Precision loss acceptable for sparsity metric calculation
        #[allow(clippy::cast_precision_loss)]
        {
            1.0 - (self.count_nonzero() as f32 / self.num_dims as f32)
        }
    }

    /// Compute dot product with another `TernaryPlanes` via popcount.
    ///
    /// Uses the formula:
    /// ```text
    /// dot = popcount(a+ & b+) + popcount(a- & b-)
    ///     - popcount(a+ & b-) - popcount(a- & b+)
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if the planes have different sizes.
    #[must_use]
    pub fn dot(&self, other: &TernaryPlanes) -> i32 {
        assert_eq!(
            self.num_words(),
            other.num_words(),
            "planes must have same size"
        );

        let mut result: i32 = 0;

        for i in 0..self.num_words() {
            let pp = (self.plus[i] & other.plus[i]).count_ones().cast_signed();
            let mm = (self.minus[i] & other.minus[i]).count_ones().cast_signed();
            let pm = (self.plus[i] & other.minus[i]).count_ones().cast_signed();
            let mp = (self.minus[i] & other.plus[i]).count_ones().cast_signed();

            result += pp + mm - pm - mp;
        }

        result
    }
}

/// Sparsity metadata for efficient plane skipping.
///
/// Tracks which chunks of dimensions are entirely zero, enabling
/// the kernel to skip computation for those regions.
#[derive(Debug, Clone)]
pub struct SparsityMetadata {
    /// Bitmap of active chunks (bit set if chunk has non-zero elements).
    /// Each bit represents `chunk_size` consecutive dimensions.
    pub active_chunks: Vec<u64>,

    /// Size of each chunk in dimensions.
    pub chunk_size: usize,

    /// Total number of chunks.
    pub num_chunks: usize,
}

impl SparsityMetadata {
    /// Create metadata from ternary planes.
    #[must_use]
    pub fn from_planes(planes: &TernaryPlanes, chunk_size: usize) -> Self {
        let num_chunks = planes.num_dims.div_ceil(chunk_size);
        let num_words = num_chunks.div_ceil(64);
        let mut active_chunks = vec![0u64; num_words];

        // Check each chunk for activity
        for chunk_idx in 0..num_chunks {
            let start_dim = chunk_idx * chunk_size;
            let end_dim = (start_dim + chunk_size).min(planes.num_dims);
            let start_word = start_dim / 32;
            let end_word = end_dim.div_ceil(32);

            let mut is_active = false;
            for word_idx in start_word..end_word {
                if planes.plus[word_idx] != 0 || planes.minus[word_idx] != 0 {
                    is_active = true;
                    break;
                }
            }

            if is_active {
                let bitmap_idx = chunk_idx / 64;
                let bit_idx = chunk_idx % 64;
                active_chunks[bitmap_idx] |= 1u64 << bit_idx;
            }
        }

        Self {
            active_chunks,
            chunk_size,
            num_chunks,
        }
    }

    /// Check if a specific chunk is active.
    #[must_use]
    pub fn is_chunk_active(&self, chunk_idx: usize) -> bool {
        if chunk_idx >= self.num_chunks {
            return false;
        }
        let bitmap_idx = chunk_idx / 64;
        let bit_idx = chunk_idx % 64;
        (self.active_chunks[bitmap_idx] & (1u64 << bit_idx)) != 0
    }

    /// Count active chunks.
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.active_chunks
            .iter()
            .map(|w| w.count_ones() as usize)
            .sum()
    }

    /// Effective sparsity (fraction of inactive chunks).
    #[must_use]
    pub fn chunk_sparsity(&self) -> f32 {
        // Precision loss acceptable for sparsity metric calculation
        #[allow(clippy::cast_precision_loss)]
        {
            1.0 - (self.active_count() as f32 / self.num_chunks as f32)
        }
    }
}

/// A ternary tensor with bitsliced storage.
///
/// Supports 2D weight matrices [`out_features`, `in_features`] packed into
/// [`out_features`, `num_words`] u32 arrays for each plane.
#[derive(Debug, Clone)]
pub struct TernaryTensor {
    /// Positive plane: [rows, `k_words`] packed u32
    pub plus_plane: Vec<u32>,

    /// Negative plane: [rows, `k_words`] packed u32
    pub minus_plane: Vec<u32>,

    /// Per-channel (row) scale factors for dequantization.
    pub scales: Vec<f32>,

    /// Original shape [`out_features`, `in_features`].
    pub shape: (usize, usize),

    /// Number of u32 words per row (`in_features` / 32, rounded up).
    pub k_words: usize,

    /// Sparsity metadata (optional, for plane skipping).
    pub sparsity_meta: Option<Vec<SparsityMetadata>>,

    /// Cached sparsity ratio.
    sparsity: f32,
}

impl TernaryTensor {
    /// Create a new ternary tensor from planes and scales.
    ///
    /// # Arguments
    ///
    /// * `plus_plane` - Flattened [rows × `k_words`] positive plane
    /// * `minus_plane` - Flattened [rows × `k_words`] negative plane
    /// * `scales` - Per-row scale factors [rows]
    /// * `shape` - Original (`out_features`, `in_features`)
    ///
    /// # Panics
    ///
    /// Panics if plane sizes don't match expected dimensions.
    #[must_use]
    pub fn new(
        plus_plane: Vec<u32>,
        minus_plane: Vec<u32>,
        scales: Vec<f32>,
        shape: (usize, usize),
    ) -> Self {
        let k_words = shape.1.div_ceil(32);
        let expected_len = shape.0 * k_words;

        assert_eq!(
            plus_plane.len(),
            expected_len,
            "plus_plane size mismatch: expected {expected_len}, got {}",
            plus_plane.len()
        );
        assert_eq!(
            minus_plane.len(),
            expected_len,
            "minus_plane size mismatch: expected {expected_len}, got {}",
            minus_plane.len()
        );
        assert_eq!(
            scales.len(),
            shape.0,
            "scales size mismatch: expected {}, got {}",
            shape.0,
            scales.len()
        );

        // Calculate sparsity
        let plus_ones: u32 = plus_plane.iter().map(|w| w.count_ones()).sum();
        let minus_ones: u32 = minus_plane.iter().map(|w| w.count_ones()).sum();
        let total_elements = shape.0 * shape.1;
        let nonzero = plus_ones + minus_ones;
        // Precision loss acceptable for sparsity metric calculation
        #[allow(clippy::cast_precision_loss)]
        let sparsity = 1.0 - (nonzero as f32 / total_elements as f32);

        Self {
            plus_plane,
            minus_plane,
            scales,
            shape,
            k_words,
            sparsity_meta: None,
            sparsity,
        }
    }

    /// Get the row and column dimensions.
    #[must_use]
    pub const fn dims(&self) -> (usize, usize) {
        self.shape
    }

    /// Get sparsity (fraction of zeros).
    #[must_use]
    pub const fn sparsity(&self) -> f32 {
        self.sparsity
    }

    /// Check if sparse enough for ternary kernel.
    #[must_use]
    pub fn is_sparse_enough(&self, config: &TernaryConfig) -> bool {
        self.sparsity >= config.sparsity_threshold
    }

    /// Build sparsity metadata for plane skipping.
    ///
    /// Call this after construction if using plane-skipping optimization.
    pub fn build_sparsity_metadata(&mut self, chunk_size: usize) {
        let mut metadata = Vec::with_capacity(self.shape.0);

        for row in 0..self.shape.0 {
            let row_offset = row * self.k_words;
            let plus_row: Vec<u32> =
                self.plus_plane[row_offset..row_offset + self.k_words].to_vec();
            let minus_row: Vec<u32> =
                self.minus_plane[row_offset..row_offset + self.k_words].to_vec();

            let planes = TernaryPlanes {
                plus: plus_row,
                minus: minus_row,
                num_dims: self.shape.1,
            };

            metadata.push(SparsityMetadata::from_planes(&planes, chunk_size));
        }

        self.sparsity_meta = Some(metadata);
    }

    /// Get row planes for CPU computation.
    ///
    /// # Panics
    ///
    /// Panics if `row` is out of bounds (>= number of rows).
    #[must_use]
    pub fn get_row_planes(&self, row: usize) -> TernaryPlanes {
        assert!(row < self.shape.0, "row out of bounds");

        let row_offset = row * self.k_words;
        TernaryPlanes {
            plus: self.plus_plane[row_offset..row_offset + self.k_words].to_vec(),
            minus: self.minus_plane[row_offset..row_offset + self.k_words].to_vec(),
            num_dims: self.shape.1,
        }
    }

    /// Memory size in bytes (for both planes + scales).
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        let plane_bytes = self.plus_plane.len() * 4 * 2; // Two planes, 4 bytes per u32
        let scale_bytes = self.scales.len() * 4; // f32 scales
        let meta_bytes = self
            .sparsity_meta
            .as_ref()
            .map_or(0, |m| m.iter().map(|s| s.active_chunks.len() * 8).sum());
        plane_bytes + scale_bytes + meta_bytes
    }

    /// Compression ratio vs FP32 weights.
    #[must_use]
    pub fn compression_ratio(&self) -> f32 {
        let fp32_bytes = self.shape.0 * self.shape.1 * 4;
        // Precision loss acceptable for compression ratio metric
        #[allow(clippy::cast_precision_loss)]
        {
            fp32_bytes as f32 / self.memory_bytes() as f32
        }
    }

    /// Convert plus plane to Candle tensor (for GPU upload).
    ///
    /// # Errors
    ///
    /// Returns error if tensor creation fails.
    pub fn plus_plane_tensor(&self, device: &Device) -> candle_core::Result<Tensor> {
        // Reinterpret u32 as bytes, then create tensor
        let shape = Shape::from_dims(&[self.shape.0, self.k_words]);
        // Candle doesn't have u32 dtype directly, use f32 reinterpret
        // Actually safer to just upload as bytes and handle in kernel
        let data: Vec<f32> = self.plus_plane.iter().map(|&x| f32::from_bits(x)).collect();
        Tensor::from_vec(data, shape, device)
    }

    /// Convert minus plane to Candle tensor (for GPU upload).
    ///
    /// # Errors
    ///
    /// Returns error if tensor creation fails.
    pub fn minus_plane_tensor(&self, device: &Device) -> candle_core::Result<Tensor> {
        let shape = Shape::from_dims(&[self.shape.0, self.k_words]);
        let data: Vec<f32> = self
            .minus_plane
            .iter()
            .map(|&x| f32::from_bits(x))
            .collect();
        Tensor::from_vec(data, shape, device)
    }

    /// Convert scales to Candle tensor.
    ///
    /// # Errors
    ///
    /// Returns error if tensor creation fails.
    pub fn scales_tensor(&self, device: &Device) -> candle_core::Result<Tensor> {
        Tensor::from_vec(self.scales.clone(), self.shape.0, device)
    }

    /// Modify a single ternary value in O(1) time via bit manipulation.
    ///
    /// This enables efficient in-place edits without reconstructing the tensor.
    /// Useful for fine-tuning, pruning, or weight correction.
    ///
    /// # Arguments
    ///
    /// * `row` - Row index (0 to out_features-1)
    /// * `col` - Column index (0 to in_features-1)
    /// * `new_val` - New ternary value: -1, 0, or +1
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `row >= shape.0` (out of bounds)
    /// - `col >= shape.1` (out of bounds)
    /// - `new_val` is not in {-1, 0, +1}
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut tensor = TernaryTensor::new(...);
    /// tensor.modify_dim(0, 5, 1);  // Set weight[0, 5] = +1
    /// tensor.modify_dim(0, 5, 0);  // Set weight[0, 5] = 0 (prune)
    /// tensor.modify_dim(0, 5, -1); // Set weight[0, 5] = -1
    /// ```
    pub fn modify_dim(&mut self, row: usize, col: usize, new_val: i8) {
        assert!(
            row < self.shape.0,
            "row {} out of bounds (max {})",
            row,
            self.shape.0 - 1
        );
        assert!(
            col < self.shape.1,
            "col {} out of bounds (max {})",
            col,
            self.shape.1 - 1
        );
        assert!(
            (-1..=1).contains(&new_val),
            "new_val must be -1, 0, or +1, got {new_val}"
        );

        let word_idx = col / 32;
        let bit_idx = col % 32;
        let mask = 1u32 << bit_idx;
        let plane_idx = row * self.k_words + word_idx;

        // Clear both planes at this position
        self.plus_plane[plane_idx] &= !mask;
        self.minus_plane[plane_idx] &= !mask;

        // Set the appropriate plane based on new value
        match new_val {
            1 => self.plus_plane[plane_idx] |= mask,
            -1 => self.minus_plane[plane_idx] |= mask,
            0 => {} // Already cleared
            _ => unreachable!(),
        }

        // Invalidate cached sparsity (it may have changed)
        // Note: We don't update self.sparsity here for performance;
        // call recalculate_sparsity() if needed after batch edits
    }

    /// Get a single ternary value.
    ///
    /// # Arguments
    ///
    /// * `row` - Row index
    /// * `col` - Column index
    ///
    /// # Returns
    ///
    /// The ternary value at (row, col): -1, 0, or +1
    ///
    /// # Panics
    ///
    /// Panics if `row` or `col` are out of bounds for the tensor dimensions.
    #[must_use]
    pub fn get_dim(&self, row: usize, col: usize) -> i8 {
        assert!(
            row < self.shape.0,
            "row index {} out of bounds for number of rows {}",
            row,
            self.shape.0
        );
        assert!(
            col < self.shape.1,
            "column index {} out of bounds for number of columns {}",
            col,
            self.shape.1
        );

        let word_idx = col / 32;
        let bit_idx = col % 32;
        let mask = 1u32 << bit_idx;
        let plane_idx = row * self.k_words + word_idx;

        let is_plus = (self.plus_plane[plane_idx] & mask) != 0;
        let is_minus = (self.minus_plane[plane_idx] & mask) != 0;

        debug_assert!(!(is_plus && is_minus), "invalid state: both planes set");

        if is_plus {
            1
        } else if is_minus {
            -1
        } else {
            0
        }
    }

    /// Recalculate and update the cached sparsity value.
    ///
    /// Call this after batch modifications via `modify_dim()`.
    pub fn recalculate_sparsity(&mut self) {
        let plus_ones: u32 = self.plus_plane.iter().map(|w| w.count_ones()).sum();
        let minus_ones: u32 = self.minus_plane.iter().map(|w| w.count_ones()).sum();
        let total_elements = self.shape.0 * self.shape.1;
        let nonzero = plus_ones + minus_ones;
        // Precision loss acceptable for sparsity metric calculation
        #[allow(clippy::cast_precision_loss)]
        {
            self.sparsity = 1.0 - (nonzero as f32 / total_elements as f32);
        }
    }

    /// Prune weights below a threshold by setting them to zero.
    ///
    /// This is intended as a batch operation that sets all weights with
    /// absolute scale contribution below `threshold` to zero.
    ///
    /// Note: For ternary weights, all non-zero values have equal magnitude
    /// (±scale), so simple threshold-based pruning would typically prune
    /// all or none per row. As a result, this method is currently a
    /// no-op stub and does not modify any weights.
    ///
    /// TODO: Implement a meaningful pruning strategy (e.g., pruning entire
    /// rows based on aggregate contribution) or remove this method if it
    /// remains unused.
    ///
    /// # Arguments
    ///
    /// * `threshold` - Minimum absolute contribution to keep
    ///
    /// # Returns
    ///
    /// Number of weights pruned (currently always 0; no-op)
    pub fn prune_below_threshold(&mut self, _threshold: f32) -> usize {
        // Intentionally a no-op: see documentation above.
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ternary_planes_basic() {
        let mut planes = TernaryPlanes::new(100);

        // Set some values
        planes.set(0, 1);
        planes.set(1, -1);
        planes.set(2, 0);
        planes.set(50, 1);
        planes.set(99, -1);

        // Verify
        assert_eq!(planes.get(0), 1);
        assert_eq!(planes.get(1), -1);
        assert_eq!(planes.get(2), 0);
        assert_eq!(planes.get(50), 1);
        assert_eq!(planes.get(99), -1);
        assert_eq!(planes.get(10), 0); // Unset defaults to 0
    }

    #[test]
    fn test_ternary_planes_dot_product() {
        let mut a = TernaryPlanes::new(64);
        let mut b = TernaryPlanes::new(64);

        // a = [1, -1, 0, 1, ...]
        // b = [1, 1, -1, 0, ...]
        a.set(0, 1);
        a.set(1, -1);
        a.set(3, 1);

        b.set(0, 1);
        b.set(1, 1);
        b.set(2, -1);

        // dot = 1*1 + (-1)*1 + 0*(-1) + 1*0 = 1 - 1 + 0 + 0 = 0
        assert_eq!(a.dot(&b), 0);

        // Test with matching signs
        b.set(1, -1); // Now b[1] = -1
                      // dot = 1*1 + (-1)*(-1) + 0*(-1) + 1*0 = 1 + 1 = 2
        assert_eq!(a.dot(&b), 2);
    }

    #[test]
    fn test_sparsity_calculation() {
        let mut planes = TernaryPlanes::new(100);

        // Set 5 non-zero values
        planes.set(0, 1);
        planes.set(10, -1);
        planes.set(20, 1);
        planes.set(50, -1);
        planes.set(99, 1);

        assert_eq!(planes.count_nonzero(), 5);
        assert!((planes.sparsity() - 0.95).abs() < 0.001);
    }

    #[test]
    fn test_sparsity_metadata() {
        let mut planes = TernaryPlanes::new(1000);

        // Set values in first chunk only
        for i in 0..10 {
            planes.set(i, 1);
        }

        let meta = SparsityMetadata::from_planes(&planes, 100);

        assert!(meta.is_chunk_active(0)); // First chunk has values
        assert!(!meta.is_chunk_active(1)); // Second chunk empty
        assert!(!meta.is_chunk_active(9)); // Last chunk empty
    }

    #[test]
    fn test_ternary_tensor_creation() {
        let shape = (4, 64); // 4 rows, 64 cols
        let k_words = 2; // 64 / 32 = 2

        let plus = vec![0u32; 4 * k_words];
        let minus = vec![0u32; 4 * k_words];
        let scales = vec![1.0f32; 4];

        let tensor = TernaryTensor::new(plus, minus, scales, shape);

        assert_eq!(tensor.dims(), (4, 64));
        assert_eq!(tensor.k_words, 2);
        assert!((tensor.sparsity() - 1.0).abs() < 0.001); // All zeros
    }

    #[test]
    fn test_compression_ratio() {
        let shape = (1024, 4096);
        let k_words = 128; // 4096 / 32

        let plus = vec![0u32; 1024 * k_words];
        let minus = vec![0u32; 1024 * k_words];
        let scales = vec![1.0f32; 1024];

        let tensor = TernaryTensor::new(plus, minus, scales, shape);

        // FP32: 1024 * 4096 * 4 = 16MB
        // Ternary: 2 planes * 1024 * 128 * 4 + scales = ~1MB + 4KB
        // Compression ~16x
        let ratio = tensor.compression_ratio();
        assert!(ratio > 10.0 && ratio < 20.0);
    }

    #[test]
    fn test_modify_dim_basic() {
        let shape = (4, 64);
        let k_words = 2;

        let plus = vec![0u32; 4 * k_words];
        let minus = vec![0u32; 4 * k_words];
        let scales = vec![1.0f32; 4];

        let mut tensor = TernaryTensor::new(plus, minus, scales, shape);

        // Initially all zeros
        assert_eq!(tensor.get_dim(0, 0), 0);
        assert_eq!(tensor.get_dim(0, 31), 0);
        assert_eq!(tensor.get_dim(0, 32), 0);

        // Set to +1
        tensor.modify_dim(0, 0, 1);
        assert_eq!(tensor.get_dim(0, 0), 1);

        // Set to -1
        tensor.modify_dim(0, 0, -1);
        assert_eq!(tensor.get_dim(0, 0), -1);

        // Set back to 0
        tensor.modify_dim(0, 0, 0);
        assert_eq!(tensor.get_dim(0, 0), 0);

        // Test across word boundary
        tensor.modify_dim(0, 32, 1);
        assert_eq!(tensor.get_dim(0, 32), 1);
        tensor.modify_dim(0, 63, -1);
        assert_eq!(tensor.get_dim(0, 63), -1);
    }

    #[test]
    fn test_modify_dim_different_rows() {
        let shape = (4, 64);
        let k_words = 2;

        let plus = vec![0u32; 4 * k_words];
        let minus = vec![0u32; 4 * k_words];
        let scales = vec![1.0f32; 4];

        let mut tensor = TernaryTensor::new(plus, minus, scales, shape);

        // Set values in different rows
        tensor.modify_dim(0, 0, 1);
        tensor.modify_dim(1, 0, -1);
        tensor.modify_dim(2, 0, 1);
        tensor.modify_dim(3, 0, 0);

        assert_eq!(tensor.get_dim(0, 0), 1);
        assert_eq!(tensor.get_dim(1, 0), -1);
        assert_eq!(tensor.get_dim(2, 0), 1);
        assert_eq!(tensor.get_dim(3, 0), 0);
    }

    #[test]
    fn test_recalculate_sparsity() {
        let shape = (4, 64);
        let k_words = 2;

        let plus = vec![0u32; 4 * k_words];
        let minus = vec![0u32; 4 * k_words];
        let scales = vec![1.0f32; 4];

        let mut tensor = TernaryTensor::new(plus, minus, scales, shape);

        // Initially 100% sparse
        assert!((tensor.sparsity() - 1.0).abs() < 0.001);

        // Add some non-zero values
        for i in 0..10 {
            tensor.modify_dim(0, i, 1);
        }

        // Sparsity is not automatically updated by modify_dim; recompute it now
        tensor.recalculate_sparsity();

        // Now should be (256 - 10) / 256 = 0.9609...
        let expected = 1.0 - (10.0 / 256.0);
        assert!((tensor.sparsity() - expected).abs() < 0.001);
    }

    #[test]
    #[should_panic(expected = "row")]
    fn test_modify_dim_row_bounds() {
        let shape = (4, 64);
        let k_words = 2;
        let plus = vec![0u32; 4 * k_words];
        let minus = vec![0u32; 4 * k_words];
        let scales = vec![1.0f32; 4];
        let mut tensor = TernaryTensor::new(plus, minus, scales, shape);

        tensor.modify_dim(4, 0, 1); // Should panic
    }

    #[test]
    #[should_panic(expected = "col")]
    fn test_modify_dim_col_bounds() {
        let shape = (4, 64);
        let k_words = 2;
        let plus = vec![0u32; 4 * k_words];
        let minus = vec![0u32; 4 * k_words];
        let scales = vec![1.0f32; 4];
        let mut tensor = TernaryTensor::new(plus, minus, scales, shape);

        tensor.modify_dim(0, 64, 1); // Should panic
    }

    #[test]
    #[should_panic(expected = "new_val")]
    fn test_modify_dim_invalid_value() {
        let shape = (4, 64);
        let k_words = 2;
        let plus = vec![0u32; 4 * k_words];
        let minus = vec![0u32; 4 * k_words];
        let scales = vec![1.0f32; 4];
        let mut tensor = TernaryTensor::new(plus, minus, scales, shape);

        tensor.modify_dim(0, 0, 2); // Should panic - invalid value outside {-1, 0, +1}
    }
}
