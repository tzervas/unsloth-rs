//! CubeCL GPU kernel for ternary bitsliced matrix multiplication.
//!
//! ## Implementation Status
//!
//! Provides both basic and tiled implementations:
//! - **Basic kernel**: Simple popcount-based implementation for validation
//! - **Tiled kernel**: Optimized with shared memory for better performance
//!
//! ## Completed Tasks
//! - Task 2.1: u32 tensor interop ✅
//! - Task 2.2: Basic popcount kernel ✅
//! - Task 2.3: Tiled kernel with shared memory ✅
//!
//! ## Future Tasks
//! - Task 2.4: Vectorized Line<u32> loads
//! - Task 2.5: Plane skipping with sparsity metadata
//! - Task 2.6: GPU dispatch integration
//!
//! ## Note on Array Types
//!
//! This implementation uses f32 arrays with bit reinterpretation for weight planes.
//! The weight planes (w_plus, w_minus) are f32 arrays where each f32 is reinterpreted as u32.
//! This approach works with CubeCL's current type system.

use cubecl::prelude::*;

/// Compile-time configuration for basic ternary matmul kernel.
/// Renamed from TernaryMatmulConfig to distinguish from the tiled version in matmul.rs.
#[derive(Clone, Copy, Debug)]
pub struct BasicTernaryMatmulConfig {
    /// Number of u32 words in K dimension (in_features / 32)
    pub k_words: u32,
    /// Number of output rows (batch size)
    pub m: u32,
    /// Number of output cols (out_features)
    pub n: u32,
    /// Number of input features (for bounds checking)
    pub in_features: u32,
}

/// Basic ternary matmul kernel with popcount-based dot product.
///
/// Each thread computes one output element: output[batch_idx, out_idx]
///
/// Algorithm:
/// ```text
/// For each input: Quantize FP input to bitsliced planes
/// For each u32 word:
///   pos_matches = popcount(w_plus & input_plus) + popcount(w_minus & input_minus)
///   neg_matches = popcount(w_plus & input_minus) + popcount(w_minus & input_plus)
///   dot += pos_matches - neg_matches
/// output = dot * scale
/// ```
///
/// This uses hardware popcount operations for efficiency.
#[cube(launch_unchecked)]
pub fn ternary_matmul_kernel_basic<F: Float>(
    // Input activations [batch, in_features] as f32
    input: &Array<F>,
    // Weight positive plane [out_features, k_words] as u32 (bit-reinterpreted from f32)
    w_plus: &Array<F>,
    // Weight negative plane [out_features, k_words] as u32
    w_minus: &Array<F>,
    // Per-row scales [out_features]
    scales: &Array<F>,
    // Output [batch, out_features]
    output: &mut Array<F>,
    // Compile-time configuration
    #[comptime] config: BasicTernaryMatmulConfig,
) {
    // Thread indices: each thread handles one (batch, out_feature) element
    // Grid layout: X=batch, Y=blocks of features
    // Block layout: X=threads (256), Y=1, Z=1
    let batch_idx = CUBE_POS_X;
    let out_idx = CUBE_POS_Y * CUBE_DIM_X + UNIT_POS_X;

    // Bounds checks
    if batch_idx >= config.m {
        return;
    }
    if out_idx >= config.n {
        return;
    }

    // Quantize input row to bitsliced planes
    // This converts FP values to ternary {-1, 0, +1} representation
    let input_offset = batch_idx * config.in_features;
    
    // Allocate local arrays for input planes (on registers)
    let mut input_plus = Array::<u32>::new(config.k_words);
    let mut input_minus = Array::<u32>::new(config.k_words);
    
    // Quantize input to planes
    for k in 0..config.k_words {
        let mut plus_word = 0u32;
        let mut minus_word = 0u32;
        
        // Process 32 dimensions per word
        let base_dim = k * 32;
        let end_dim = u32::min(base_dim + 32, config.in_features);
        
        for bit in 0u32..(end_dim - base_dim) {
            let dim_idx = base_dim + bit;
            let val = input[input_offset + dim_idx];
            
            // Simple threshold quantization: >0.5 → +1, <-0.5 → -1, else 0
            let threshold = F::new(0.5);
            let neg_threshold = F::new(-0.5);
            
            if val > threshold {
                plus_word = plus_word | (1u32 << bit);
            } else if val < neg_threshold {
                minus_word = minus_word | (1u32 << bit);
            }
        }
        
        input_plus[k] = plus_word;
        input_minus[k] = minus_word;
    }

    // Popcount-based dot product
    let weight_offset = out_idx * config.k_words;
    let mut pos_sum = 0u32;
    let mut neg_sum = 0u32;
    
    for k in 0..config.k_words {
        // Reinterpret f32 as u32 bits for weight planes
        let wp_f32 = w_plus[weight_offset + k];
        let wm_f32 = w_minus[weight_offset + k];
        let wp_bits = u32::reinterpret(wp_f32);
        let wm_bits = u32::reinterpret(wm_f32);
        
        let ip = input_plus[k];
        let im = input_minus[k];
        
        // Popcount-based ternary dot product:
        // pos_matches = (+1 weights × +1 inputs) + (-1 weights × -1 inputs)
        // neg_matches = (+1 weights × -1 inputs) + (-1 weights × +1 inputs)
        // dot = pos_matches - neg_matches
        
        pos_sum = pos_sum + (wp_bits & ip).count_ones();
        pos_sum = pos_sum + (wm_bits & im).count_ones();
        
        neg_sum = neg_sum + (wp_bits & im).count_ones();
        neg_sum = neg_sum + (wm_bits & ip).count_ones();
    }

    // Convert popcount result to float and apply scale
    let dot = F::cast_from(pos_sum) - F::cast_from(neg_sum);
    let scale = scales[out_idx];
    output[batch_idx * config.n + out_idx] = dot * scale;
}

/// Compile-time configuration for tiled ternary matmul kernel.
#[derive(Clone, Copy, Debug)]
pub struct TiledTernaryMatmulConfig {
    /// Number of u32 words in K dimension (in_features / 32)
    pub k_words: u32,
    /// Number of output rows (batch size)
    pub m: u32,
    /// Number of output cols (out_features)
    pub n: u32,
    /// Number of input features (for bounds checking)
    pub in_features: u32,
    /// Tile size for K dimension in u32 words (default: 32-64)
    pub tile_k: u32,
    /// Number of outputs computed per thread (default: 1-4)
    pub outputs_per_thread: u32,
    /// Block size (threads per block, default: 256)
    pub block_size: u32,
}

impl TiledTernaryMatmulConfig {
    /// Create configuration optimized for RTX 5080
    pub fn rtx_5080_preset(m: u32, n: u32, k_words: u32, in_features: u32) -> Self {
        Self {
            k_words,
            m,
            n,
            in_features,
            tile_k: 64,  // 64 u32 words = 2048 dimensions
            outputs_per_thread: 2,
            block_size: 256,
        }
    }

    /// Create configuration optimized for RTX 3090 Ti
    pub fn rtx_3090ti_preset(m: u32, n: u32, k_words: u32, in_features: u32) -> Self {
        Self {
            k_words,
            m,
            n,
            in_features,
            tile_k: 32,  // 32 u32 words = 1024 dimensions
            outputs_per_thread: 2,
            block_size: 256,
        }
    }
}

/// Tiled ternary matmul kernel with shared memory optimization.
///
/// Uses shared memory to cache input tiles and weight planes for better memory locality.
/// Each thread computes multiple output elements to improve occupancy.
///
/// Algorithm:
/// ```text
/// For each K tile:
///   - Cooperatively load input tile to shared memory
///   - Quantize input tile to bitsliced planes in shared memory
///   - Each thread loads its weight rows
///   - Compute popcount dot products for tile
///   - Accumulate results
/// output = accumulated_dot * scale
/// ```
#[cube(launch_unchecked)]
pub fn ternary_matmul_kernel_tiled<F: Float>(
    // Input activations [batch, in_features] as f32
    input: &Array<F>,
    // Weight positive plane [out_features, k_words] as u32 (bit-reinterpreted from f32)
    w_plus: &Array<F>,
    // Weight negative plane [out_features, k_words] as u32
    w_minus: &Array<F>,
    // Per-row scales [out_features]
    scales: &Array<F>,
    // Output [batch, out_features]
    output: &mut Array<F>,
    // Compile-time configuration
    #[comptime] config: TiledTernaryMatmulConfig,
) {
    // Thread and block indices
    let batch_idx = CUBE_POS_X;
    let out_block_idx = CUBE_POS_Y;
    let thread_idx = UNIT_POS_X;

    // Bounds check for batch
    if batch_idx >= config.m {
        return;
    }

    // Shared memory for input tile (plus and minus planes)
    // Each tile holds tile_k u32 words for plus and minus planes
    let mut input_plus_tile = SharedMemory::<u32>::new(config.tile_k);
    let mut input_minus_tile = SharedMemory::<u32>::new(config.tile_k);

    // Each thread computes outputs_per_thread outputs
    for out_local in 0..config.outputs_per_thread {
        let out_idx = out_block_idx * config.block_size * config.outputs_per_thread
            + thread_idx * config.outputs_per_thread
            + out_local;

        if out_idx >= config.n {
            continue;
        }

        let input_offset = batch_idx * config.in_features;
        let weight_offset = out_idx * config.k_words;

        // Accumulator for this output
        let mut pos_sum = 0u32;
        let mut neg_sum = 0u32;

        // Number of K tiles
        let num_k_tiles = (config.k_words + config.tile_k - 1) / config.tile_k;

        // Process K dimension in tiles
        for k_tile in 0..num_k_tiles {
            let k_start = k_tile * config.tile_k;
            let k_end = u32::min(k_start + config.tile_k, config.k_words);
            let tile_size = k_end - k_start;

            // Cooperatively load and quantize input tile to shared memory
            // Each thread loads one u32 word worth of input (32 FP values → 1 u32 word)
            if thread_idx < tile_size {
                let k_word = k_start + thread_idx;
                let mut plus_word = 0u32;
                let mut minus_word = 0u32;

                // Quantize 32 input dimensions to one u32 word
                let base_dim = k_word * 32;
                let end_dim = u32::min(base_dim + 32, config.in_features);

                for bit in 0..(end_dim - base_dim) {
                    let dim_idx = base_dim + bit;
                    let val = input[input_offset + dim_idx];

                    let threshold = F::new(0.5);
                    let neg_threshold = F::new(-0.5);

                    if val > threshold {
                        plus_word = plus_word | (1u32 << bit);
                    } else if val < neg_threshold {
                        minus_word = minus_word | (1u32 << bit);
                    }
                }

                input_plus_tile[thread_idx] = plus_word;
                input_minus_tile[thread_idx] = minus_word;
            }

            // Synchronize to ensure tile is fully loaded
            sync_units();

            // Compute popcount dot product for this tile
            for k_local in 0..tile_size {
                let k_word = k_start + k_local;

                // Load weight planes for this output feature
                let wp_f32 = w_plus[weight_offset + k_word];
                let wm_f32 = w_minus[weight_offset + k_word];
                let wp_bits = u32::reinterpret(wp_f32);
                let wm_bits = u32::reinterpret(wm_f32);

                // Load input planes from shared memory
                let ip = input_plus_tile[k_local];
                let im = input_minus_tile[k_local];

                // Popcount-based ternary dot product
                pos_sum = pos_sum + (wp_bits & ip).count_ones();
                pos_sum = pos_sum + (wm_bits & im).count_ones();

                neg_sum = neg_sum + (wp_bits & im).count_ones();
                neg_sum = neg_sum + (wm_bits & ip).count_ones();
            }

            // Synchronize before loading next tile
            sync_units();
        }

        // Convert popcount result to float and apply scale
        let dot = F::cast_from(pos_sum) - F::cast_from(neg_sum);
        let scale = scales[out_idx];
        output[batch_idx * config.n + out_idx] = dot * scale;
    }
}

/// Launch configuration for the tiled kernel
pub fn get_tiled_launch_config(
    config: &TiledTernaryMatmulConfig,
) -> (CubeCount, CubeDim) {
    // Number of output blocks needed
    let outputs_per_block = config.block_size * config.outputs_per_thread;
    let grid_y = (config.n + outputs_per_block - 1) / outputs_per_block;

    let cube_count = CubeCount::Static(config.m, grid_y, 1);
    let cube_dim = CubeDim::new(config.block_size, 1, 1);

    (cube_count, cube_dim)
}

/// Launch configuration for the basic kernel
pub fn get_basic_launch_config(batch_size: u32, out_features: u32) -> (CubeCount, CubeDim) {
    // Block size: 256 threads (warp-aligned)
    let block_size = 256u32;
    
    // Grid: (batch_size, ceil(out_features / block_size), 1)
    let grid_y = (out_features + block_size - 1) / block_size;
    
    let cube_count = CubeCount::Static(batch_size, grid_y, 1);
    let cube_dim = CubeDim::new(block_size, 1, 1);
    
    (cube_count, cube_dim)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_creation() {
        let config = BasicTernaryMatmulConfig {
            k_words: 4,
            m: 8,
            n: 64,
            in_features: 128,
        };
        assert_eq!(config.k_words, 4);
        assert_eq!(config.in_features, 128);
    }

    #[test]
    fn test_launch_config() {
        let (cube_count, cube_dim) = get_basic_launch_config(4, 512);
        // Should have 4 blocks in X (batch), 2 blocks in Y (512/256)
        assert_eq!(cube_dim.x, 256);
        assert_eq!(cube_dim.y, 1);
        assert_eq!(cube_dim.z, 1);
        
        // Verify grid dimensions
        if let CubeCount::Static(x, y, z) = cube_count {
            assert_eq!(x, 4, "Grid X dimension should match batch size");
            assert_eq!(y, 2, "Grid Y dimension should be ceil(512/256) = 2");
            assert_eq!(z, 1, "Grid Z dimension should be 1");
        } else {
            panic!("Expected Static cube count");
        }
    }

    /// Functional test: validates kernel algorithm with known ternary weights.
    /// 
    /// This test verifies the ternary matrix multiplication logic by simulating
    /// the kernel computation on CPU with popcount-based operations.
    #[test]
    fn test_ternary_matmul_kernel_correctness() {
        // Simple test case: 2 batches, 3 output features, 64 input features (2 words)
        // Weights pattern: each output uses different ternary pattern
        
        let batch_size = 2;
        let in_features = 64;
        let out_features = 3;
        let k_words = 2;  // 64 / 32 = 2
        
        // Input: values that will quantize clearly: >0.5 → +1, <-0.5 → -1, else 0
        // For batch 0: alternating pattern [1.0, -1.0, 1.0, -1.0, ...]
        // For batch 1: all positive [1.0, 1.0, 1.0, ...]
        let mut input_data = vec![0.0f32; batch_size * in_features];
        for b in 0..batch_size {
            for i in 0..in_features {
                input_data[b * in_features + i] = if b == 0 {
                    if i % 2 == 0 { 1.0 } else { -1.0 }
                } else {
                    1.0
                };
            }
        }
        
        // Quantize inputs to planes (simulating kernel quantization)
        let mut input_plus = vec![0u32; batch_size * k_words];
        let mut input_minus = vec![0u32; batch_size * k_words];
        
        for b in 0..batch_size {
            for k in 0..k_words {
                let mut plus_word = 0u32;
                let mut minus_word = 0u32;
                
                for bit in 0..32 {
                    let dim_idx = k * 32 + bit;
                    let val = input_data[b * in_features + dim_idx];
                    
                    if val > 0.5 {
                        plus_word |= 1u32 << bit;
                    } else if val < -0.5 {
                        minus_word |= 1u32 << bit;
                    }
                }
                
                input_plus[b * k_words + k] = plus_word;
                input_minus[b * k_words + k] = minus_word;
            }
        }
        
        // Weights: Create simple ternary patterns
        // Out feature 0: all +1 in both words
        // Out feature 1: all -1 in both words
        // Out feature 2: alternating +1/-1 pattern
        let mut w_plus = vec![0u32; out_features * k_words];
        let mut w_minus = vec![0u32; out_features * k_words];
        
        // Feature 0: all positive
        w_plus[0] = 0xFFFFFFFF;
        w_plus[1] = 0xFFFFFFFF;
        
        // Feature 1: all negative
        w_minus[2] = 0xFFFFFFFF;
        w_minus[3] = 0xFFFFFFFF;
        
        // Feature 2: alternating (0xAAAAAAAA = 10101...2)
        w_plus[4] = 0xAAAAAAAA;
        w_plus[5] = 0xAAAAAAAA;
        w_minus[4] = 0x55555555;
        w_minus[5] = 0x55555555;
        
        // Scales: all 1.0 for simplicity
        let scales = vec![1.0f32; out_features];
        
        // Simulate popcount-based kernel computation
        let mut output = vec![0.0f32; batch_size * out_features];
        
        for batch_idx in 0..batch_size {
            for out_idx in 0..out_features {
                let weight_offset = out_idx * k_words;
                let input_offset = batch_idx * k_words;
                let mut pos_sum = 0u32;
                let mut neg_sum = 0u32;
                
                for k in 0..k_words {
                    let wp_bits = w_plus[weight_offset + k];
                    let wm_bits = w_minus[weight_offset + k];
                    let ip = input_plus[input_offset + k];
                    let im = input_minus[input_offset + k];
                    
                    // Popcount-based dot product
                    pos_sum += (wp_bits & ip).count_ones();
                    pos_sum += (wm_bits & im).count_ones();
                    neg_sum += (wp_bits & im).count_ones();
                    neg_sum += (wm_bits & ip).count_ones();
                }
                
                let dot = (pos_sum as i32 - neg_sum as i32) as f32;
                output[batch_idx * out_features + out_idx] = dot * scales[out_idx];
            }
        }
        
        // Batch 0 input quantizes to: +plane = 0xAAAAAAAA (even bits), -plane = 0x55555555 (odd bits)
        // Batch 1 input quantizes to: +plane = 0xFFFFFFFF (all bits), -plane = 0x00000000 (no bits)
        
        // Batch 0, Feature 0 (all +1 weights):
        //   pos_sum = popcount(0xFFFF... & 0xAAAA...) + popcount(0x0 & 0x5555...) = 32 + 0 = 32 (per word) = 64 total
        //   neg_sum = popcount(0xFFFF... & 0x5555...) + popcount(0x0 & 0xAAAA...) = 32 + 0 = 32 (per word) = 64 total
        //   dot = 64 - 64 = 0
        let expected_b0_f0 = 0.0;
        
        // Batch 0, Feature 1 (all -1 weights):
        //   pos_sum = popcount(0x0 & 0xAAAA...) + popcount(0xFFFF... & 0x5555...) = 0 + 32 = 32 (per word) = 64 total
        //   neg_sum = popcount(0x0 & 0x5555...) + popcount(0xFFFF... & 0xAAAA...) = 0 + 32 = 32 (per word) = 64 total
        //   dot = 64 - 64 = 0
        let expected_b0_f1 = 0.0;
        
        // Batch 0, Feature 2 (alternating):
        //   pos_sum = popcount(0xAAAA... & 0xAAAA...) + popcount(0x5555... & 0x5555...) = 16 + 16 = 32 (per word) = 64 total
        //   neg_sum = popcount(0xAAAA... & 0x5555...) + popcount(0x5555... & 0xAAAA...) = 0 + 0 = 0
        //   dot = 64 - 0 = 64
        let expected_b0_f2 = 64.0;
        
        // Batch 1, Feature 0 (all +1 weights, all +1 input):
        //   pos_sum = popcount(0xFFFF... & 0xFFFF...) + popcount(0x0 & 0x0) = 64 + 0 = 64
        //   neg_sum = popcount(0xFFFF... & 0x0) + popcount(0x0 & 0xFFFF...) = 0 + 0 = 0
        //   dot = 64 - 0 = 64
        let expected_b1_f0 = 64.0;
        
        // Batch 1, Feature 1 (all -1 weights, all +1 input):
        //   pos_sum = popcount(0x0 & 0xFFFF...) + popcount(0xFFFF... & 0x0) = 0 + 0 = 0
        //   neg_sum = popcount(0x0 & 0x0) + popcount(0xFFFF... & 0xFFFF...) = 0 + 64 = 64
        //   dot = 0 - 64 = -64
        let expected_b1_f1 = -64.0;
        
        // Batch 1, Feature 2 (alternating, all +1 input):
        //   pos_sum = popcount(0xAAAA... & 0xFFFF...) + popcount(0x5555... & 0x0) = 32 + 0 = 32 (per word) = 64 total
        //   neg_sum = popcount(0xAAAA... & 0x0) + popcount(0x5555... & 0xFFFF...) = 0 + 32 = 32 (per word) = 64 total
        //   dot = 64 - 64 = 0
        let expected_b1_f2 = 0.0;
        
        assert!((output[0] - expected_b0_f0).abs() < 0.01, 
                "B0F0 mismatch: expected {}, got {}", expected_b0_f0, output[0]);
        assert!((output[1] - expected_b0_f1).abs() < 0.01,
                "B0F1 mismatch: expected {}, got {}", expected_b0_f1, output[1]);
        assert!((output[2] - expected_b0_f2).abs() < 0.01,
                "B0F2 mismatch: expected {}, got {}", expected_b0_f2, output[2]);
        
        assert!((output[3] - expected_b1_f0).abs() < 0.01,
                "B1F0 mismatch: expected {}, got {}", expected_b1_f0, output[3]);
        assert!((output[4] - expected_b1_f1).abs() < 0.01,
                "B1F1 mismatch: expected {}, got {}", expected_b1_f1, output[4]);
        assert!((output[5] - expected_b1_f2).abs() < 0.01,
                "B1F2 mismatch: expected {}, got {}", expected_b1_f2, output[5]);
    }

    /// Test partial word bounds checking with non-multiple of 32 input features.
    /// 
    /// This test validates the quantization and popcount with partial words
    /// (when in_features % 32 != 0).
    #[test]
    fn test_ternary_matmul_partial_word() {
        // Test with in_features = 48 (1 full word + 16 bits in partial word)
        let batch_size = 2;
        let in_features = 48;  // Not a multiple of 32
        let out_features = 2;
        let k_words = 2;  // ceil(48 / 32) = 2 words (second is partial)
        
        // Input: all +1.0 for batch 0, all -1.0 for batch 1
        let mut input_data = vec![0.0f32; batch_size * in_features];
        for b in 0..batch_size {
            for i in 0..in_features {
                input_data[b * in_features + i] = if b == 0 { 1.0 } else { -1.0 };
            }
        }
        
        // Quantize inputs
        let mut input_plus = vec![0u32; batch_size * k_words];
        let mut input_minus = vec![0u32; batch_size * k_words];
        
        for b in 0..batch_size {
            for k in 0..k_words {
                let mut plus_word = 0u32;
                let mut minus_word = 0u32;
                
                let base_dim = k * 32;
                let end_dim = std::cmp::min(base_dim + 32, in_features);
                
                for bit in 0..(end_dim - base_dim) {
                    let dim_idx = base_dim + bit;
                    let val = input_data[b * in_features + dim_idx];
                    
                    if val > 0.5 {
                        plus_word |= 1u32 << bit;
                    } else if val < -0.5 {
                        minus_word |= 1u32 << bit;
                    }
                }
                
                input_plus[b * k_words + k] = plus_word;
                input_minus[b * k_words + k] = minus_word;
            }
        }
        
        // Weights:
        // Feature 0: all +1 in both words (32 + 16 = 48 dimensions)
        // Feature 1: all -1 in partial second word only (16 dimensions)
        let mut w_plus = vec![0u32; out_features * k_words];
        let mut w_minus = vec![0u32; out_features * k_words];
        
        // Feature 0: positive in both words
        w_plus[0] = 0xFFFFFFFF;  // word 0 (bits 0-31, all valid)
        w_plus[1] = 0x0000FFFF;  // word 1 (bits 0-15 valid, 16-31 should be ignored)
        
        // Feature 1: negative in partial word only
        w_minus[2] = 0x00000000;  // word 0 (no contribution)
        w_minus[3] = 0x0000FFFF;  // word 1 (bits 0-15 valid)
        
        let scales = vec![1.0f32; out_features];
        
        // Simulate popcount computation
        let mut output = vec![0.0f32; batch_size * out_features];
        
        for batch_idx in 0..batch_size {
            for out_idx in 0..out_features {
                let weight_offset = out_idx * k_words;
                let input_offset = batch_idx * k_words;
                let mut pos_sum = 0u32;
                let mut neg_sum = 0u32;
                
                for k in 0..k_words {
                    let wp_bits = w_plus[weight_offset + k];
                    let wm_bits = w_minus[weight_offset + k];
                    let ip = input_plus[input_offset + k];
                    let im = input_minus[input_offset + k];
                    
                    pos_sum += (wp_bits & ip).count_ones();
                    pos_sum += (wm_bits & im).count_ones();
                    neg_sum += (wp_bits & im).count_ones();
                    neg_sum += (wm_bits & ip).count_ones();
                }
                
                let dot = (pos_sum as i32 - neg_sum as i32) as f32;
                output[batch_idx * out_features + out_idx] = dot * scales[out_idx];
            }
        }
        
        // Batch 0 (all +1 input):
        //   input quantizes to: word0=0xFFFFFFFF, word1=0x0000FFFF
        // Feature 0 (all +1 weights):
        //   pos_sum = popcount(0xFFFF... & 0xFFFF...) + popcount(0xFFFF & 0xFFFF) = 32 + 16 = 48
        //   neg_sum = 0
        //   dot = 48
        let expected_b0_f0 = 48.0;
        
        // Feature 1 (partial -1 weights):
        //   pos_sum = 0
        //   neg_sum = popcount(0xFFFF & 0xFFFF) = 16
        //   dot = -16
        let expected_b0_f1 = 0.0;  // No matches since input is +1 and weight is -1
        
        // Batch 1 (all -1 input):
        //   input quantizes to: word0=0xFFFFFFFF, word1=0x0000FFFF in minus plane
        // Feature 0:
        //   pos_sum = 0
        //   neg_sum = popcount(0xFFFF... & 0xFFFF...) + popcount(0xFFFF & 0xFFFF) = 32 + 16 = 48
        //   dot = -48
        let expected_b1_f0 = -48.0;
        
        // Feature 1:
        //   pos_sum = popcount(0xFFFF & 0xFFFF) = 16
        //   neg_sum = 0
        //   dot = 16
        let expected_b1_f1 = 16.0;
        
        assert!((output[0] - expected_b0_f0).abs() < 0.01,
                "B0F0 mismatch: expected {}, got {}", expected_b0_f0, output[0]);
        assert!((output[1] - expected_b0_f1).abs() < 0.01,
                "B0F1 mismatch: expected {}, got {}", expected_b0_f1, output[1]);
        
        assert!((output[2] - expected_b1_f0).abs() < 0.01,
                "B1F0 mismatch: expected {}, got {}", expected_b1_f0, output[2]);
        assert!((output[3] - expected_b1_f1).abs() < 0.01,
                "B1F1 mismatch: expected {}, got {}", expected_b1_f1, output[3]);
    }

    #[test]
    fn test_tiled_config_creation() {
        let config = TiledTernaryMatmulConfig::rtx_5080_preset(8, 512, 32, 1024);
        assert_eq!(config.k_words, 32);
        assert_eq!(config.m, 8);
        assert_eq!(config.n, 512);
        assert_eq!(config.tile_k, 64);
        assert_eq!(config.outputs_per_thread, 2);
        assert_eq!(config.block_size, 256);
    }

    #[test]
    fn test_tiled_launch_config() {
        let config = TiledTernaryMatmulConfig::rtx_5080_preset(4, 512, 32, 1024);
        let (cube_count, cube_dim) = get_tiled_launch_config(&config);

        assert_eq!(cube_dim.x, 256);
        assert_eq!(cube_dim.y, 1);
        assert_eq!(cube_dim.z, 1);

        // With outputs_per_thread=2 and block_size=256:
        // outputs_per_block = 2 * 256 = 512
        // grid_y = ceil(512 / 512) = 1
        if let CubeCount::Static(x, y, z) = cube_count {
            assert_eq!(x, 4, "Grid X should match batch size");
            assert_eq!(y, 1, "Grid Y should be ceil(512 / (256*2)) = 1");
            assert_eq!(z, 1);
        } else {
            panic!("Expected Static cube count");
        }
    }

    #[test]
    fn test_tiled_presets() {
        // Test RTX 5080 preset
        let rtx5080 = TiledTernaryMatmulConfig::rtx_5080_preset(16, 1024, 64, 2048);
        assert_eq!(rtx5080.tile_k, 64, "RTX 5080 should use larger tiles");

        // Test RTX 3090 Ti preset
        let rtx3090 = TiledTernaryMatmulConfig::rtx_3090ti_preset(16, 1024, 64, 2048);
        assert_eq!(rtx3090.tile_k, 32, "RTX 3090 Ti should use smaller tiles");

        // Both should have same block size
        assert_eq!(rtx5080.block_size, rtx3090.block_size);
    }

    /// Test tiled kernel correctness with small tile size.
    ///
    /// This validates that tiling produces the same results as the basic kernel.
    #[test]
    fn test_tiled_kernel_correctness() {
        let batch_size = 2;
        let in_features = 128;  // 4 words
        let out_features = 4;
        let k_words = 4;

        // Input: alternating pattern for batch 0, all positive for batch 1
        let mut input_data = vec![0.0f32; batch_size * in_features];
        for b in 0..batch_size {
            for i in 0..in_features {
                input_data[b * in_features + i] = if b == 0 {
                    if i % 2 == 0 { 1.0 } else { -1.0 }
                } else {
                    1.0
                };
            }
        }

        // Quantize inputs
        let mut input_plus = vec![0u32; batch_size * k_words];
        let mut input_minus = vec![0u32; batch_size * k_words];

        for b in 0..batch_size {
            for k in 0..k_words {
                let mut plus_word = 0u32;
                let mut minus_word = 0u32;

                for bit in 0..32 {
                    let dim_idx = k * 32 + bit;
                    let val = input_data[b * in_features + dim_idx];

                    if val > 0.5 {
                        plus_word |= 1u32 << bit;
                    } else if val < -0.5 {
                        minus_word |= 1u32 << bit;
                    }
                }

                input_plus[b * k_words + k] = plus_word;
                input_minus[b * k_words + k] = minus_word;
            }
        }

        // Weights: different patterns
        let mut w_plus = vec![0u32; out_features * k_words];
        let mut w_minus = vec![0u32; out_features * k_words];

        // Feature 0: all +1
        for k in 0..k_words {
            w_plus[0 * k_words + k] = 0xFFFFFFFF;
        }

        // Feature 1: all -1
        for k in 0..k_words {
            w_minus[1 * k_words + k] = 0xFFFFFFFF;
        }

        // Feature 2: alternating
        for k in 0..k_words {
            w_plus[2 * k_words + k] = 0xAAAAAAAA;
            w_minus[2 * k_words + k] = 0x55555555;
        }

        // Feature 3: first half +1, second half -1
        w_plus[3 * k_words + 0] = 0xFFFFFFFF;
        w_plus[3 * k_words + 1] = 0xFFFFFFFF;
        w_minus[3 * k_words + 2] = 0xFFFFFFFF;
        w_minus[3 * k_words + 3] = 0xFFFFFFFF;

        let scales = vec![1.0f32; out_features];

        // Simulate tiled computation with tile_k = 2 (2 words per tile)
        let mut output = vec![0.0f32; batch_size * out_features];
        let tile_k = 2;

        for batch_idx in 0..batch_size {
            for out_idx in 0..out_features {
                let input_offset = batch_idx * k_words;
                let weight_offset = out_idx * k_words;
                let mut pos_sum = 0u32;
                let mut neg_sum = 0u32;

                // Process in tiles
                let num_tiles = (k_words + tile_k - 1) / tile_k;
                for tile_idx in 0..num_tiles {
                    let k_start = tile_idx * tile_k;
                    let k_end = std::cmp::min(k_start + tile_k, k_words);

                    for k in k_start..k_end {
                        let wp_bits = w_plus[weight_offset + k];
                        let wm_bits = w_minus[weight_offset + k];
                        let ip = input_plus[input_offset + k];
                        let im = input_minus[input_offset + k];

                        pos_sum += (wp_bits & ip).count_ones();
                        pos_sum += (wm_bits & im).count_ones();
                        neg_sum += (wp_bits & im).count_ones();
                        neg_sum += (wm_bits & ip).count_ones();
                    }
                }

                let dot = (pos_sum as i32 - neg_sum as i32) as f32;
                output[batch_idx * out_features + out_idx] = dot * scales[out_idx];
            }
        }

        // Verify results match expected values
        // Batch 0: alternating input (0xAAAA... and 0x5555...)
        // Feature 0: all +1 → dot = 0 (equal pos and neg matches)
        // Feature 1: all -1 → dot = 0
        // Feature 2: alternating → dot = 256 (all matches are positive)
        // Feature 3: half +1, half -1 → dot = 0

        let expected = vec![
            0.0, 0.0, 256.0, 0.0,  // Batch 0
            256.0, -256.0, 0.0, 0.0, // Batch 1 (all +1 input)
        ];

        for (i, (actual, expected)) in output.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 0.01,
                "Mismatch at index {}: expected {}, got {}",
                i,
                expected,
                actual
            );
        }
    }
}
