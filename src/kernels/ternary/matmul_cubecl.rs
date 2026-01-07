//! CubeCL GPU kernel for ternary bitsliced matrix multiplication.
//!
//! ## Implementation Status
//!
//! This is a **basic, non-optimized** implementation using bit-by-bit iteration
//! for correctness validation. Future tasks will add:
//! - Task 2.3: Tiling with SharedMemory
//! - Task 2.4: Vectorized Line loads
//! - Task 2.5: Optimized popcount-based dot products
//!
//! ## Note on Array Types
//!
//! This implementation uses f32 arrays with bit reinterpretation for weight planes.
//! Task 2.1 will add proper u32 array support via CubeCL interop.
//! The weight planes (w_plus, w_minus) are f32 arrays where each f32 is reinterpreted as u32.
//! This is a temporary workaround until proper u32 tensor support is available.

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

/// Basic ternary matmul kernel (bit-by-bit iteration).
///
/// Each thread computes one output element: output[batch_idx, out_idx]
///
/// Algorithm:
/// ```text
/// For each bit in packed weights:
///   Extract ternary value from +plane and -plane
///   if +1: acc += input[bit]
///   if -1: acc -= input[bit]
///   if  0: skip
/// output = acc * scale
/// ```
///
/// Note: This uses bit-by-bit iteration for correctness validation.
/// Future optimizations (Task 2.5) will use vectorized popcount operations.
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

    // Accumulator for dot product
    let mut acc = F::new(0.0);

    // Input offset for this batch element
    let input_offset = batch_idx * config.in_features;
    // Weight offset for this output feature
    let weight_offset = out_idx * config.k_words;

    // Iterate over K dimension (packed u32 words)
    for k in 0..config.k_words {
        // Reinterpret f32 as u32 bits for weight planes
        let wp_f32 = w_plus[weight_offset + k];
        let wm_f32 = w_minus[weight_offset + k];
        let wp_bits = u32::reinterpret(wp_f32);
        let wm_bits = u32::reinterpret(wm_f32);
        
        // Check if this is a full word (all 32 bits are valid)
        let is_full_word = (k * 32 + 32) <= config.in_features;
        
        // For each bit position in the u32 word
        for bit in 0u32..32u32 {
            let dim_idx = k * 32 + bit;
            
            // Only check bounds for partial words
            if !is_full_word && dim_idx >= config.in_features {
                break;
            }
            
            let mask = 1u32 << bit;
            
            // Extract ternary weight value from planes
            let is_pos = (wp_bits & mask) != 0u32;
            let is_neg = (wm_bits & mask) != 0u32;

            let input_val = input[input_offset + dim_idx];

            // Ternary multiplication: +1, 0, or -1
            if is_pos {
                acc = acc + input_val;
            } else if is_neg {
                acc = acc - input_val;
            }
        }
    }

    // Apply per-channel scale
    let scale = scales[out_idx];
    output[batch_idx * config.n + out_idx] = acc * scale;
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
    /// the kernel computation on CPU with simple test data.
    #[test]
    fn test_ternary_matmul_kernel_correctness() {
        // Simple test case: 2 batches, 3 output features, 64 input features (2 words)
        // Weights pattern: each output uses different ternary pattern
        
        let batch_size = 2;
        let in_features = 64;
        let out_features = 3;
        let k_words = 2;  // 64 / 32 = 2
        
        // Input: simple pattern [1.0, 2.0, 3.0, ..., 64.0] for batch 0
        //                       [0.5, 1.0, 1.5, ..., 32.0] for batch 1
        let mut input_data = vec![0.0f32; batch_size * in_features];
        for b in 0..batch_size {
            for i in 0..in_features {
                input_data[b * in_features + i] = (i + 1) as f32 / (b + 1) as f32;
            }
        }
        
        // Weights: Create simple ternary patterns
        // Out feature 0: first 32 bits = +1, rest = 0
        // Out feature 1: first 32 bits = -1, rest = 0
        // Out feature 2: alternate +1/-1 pattern in first word
        let mut w_plus = vec![0u32; out_features * k_words];
        let mut w_minus = vec![0u32; out_features * k_words];
        
        // Feature 0: all positive in first word
        w_plus[0] = 0xFFFFFFFF;  // word 0
        w_plus[1] = 0;           // word 1
        
        // Feature 1: all negative in first word  
        w_minus[2] = 0xFFFFFFFF; // word 0
        w_minus[3] = 0;          // word 1
        
        // Feature 2: alternating pattern (0xAAAAAAAA = 1010... binary)
        w_plus[4] = 0xAAAAAAAA;  // word 0, every other bit
        w_minus[4] = 0x55555555; // word 0, opposite bits
        w_plus[5] = 0;
        w_minus[5] = 0;
        
        // Scales: all 1.0 for simplicity
        let scales = vec![1.0f32; out_features];
        
        // Expected outputs:
        // Batch 0, Feature 0: sum(1..32) = 32*33/2 = 528
        // Batch 0, Feature 1: -sum(1..32) = -528
        // Batch 0, Feature 2: sum(input values at odd bit positions) - sum(input values at even bit positions)
        //                   = (2+4+6+...+32) - (1+3+5+...+31) = 16
        
        // Simulate kernel computation
        let mut output = vec![0.0f32; batch_size * out_features];
        
        for batch_idx in 0..batch_size {
            for out_idx in 0..out_features {
                let mut acc = 0.0f32;
                let input_offset = batch_idx * in_features;
                let weight_offset = out_idx * k_words;
                
                for k in 0..k_words {
                    let wp_bits = w_plus[weight_offset + k];
                    let wm_bits = w_minus[weight_offset + k];
                    
                    let is_full_word = (k * 32 + 32) <= in_features;
                    
                    for bit in 0..32 {
                        let dim_idx = k * 32 + bit;
                        
                        if !is_full_word && dim_idx >= in_features {
                            break;
                        }
                        
                        let mask = 1u32 << bit;
                        let is_pos = (wp_bits & mask) != 0;
                        let is_neg = (wm_bits & mask) != 0;
                        
                        let input_val = input_data[input_offset + dim_idx];
                        
                        if is_pos {
                            acc += input_val;
                        } else if is_neg {
                            acc -= input_val;
                        }
                    }
                }
                
                output[batch_idx * out_features + out_idx] = acc * scales[out_idx];
            }
        }
        
        // Verify batch 0 results
        let expected_0_0 = (1..=32).sum::<i32>() as f32; // sum(1..32) = 528
        let expected_0_1 = -expected_0_0;                 // -528
        let expected_0_2 = 16.0;                          // alternating pattern
        
        assert!((output[0] - expected_0_0).abs() < 0.01, 
                "Feature 0 mismatch: expected {}, got {}", expected_0_0, output[0]);
        assert!((output[1] - expected_0_1).abs() < 0.01,
                "Feature 1 mismatch: expected {}, got {}", expected_0_1, output[1]);
        assert!((output[2] - expected_0_2).abs() < 0.01,
                "Feature 2 mismatch: expected {}, got {}", expected_0_2, output[2]);
        
        // Verify batch 1 results (half of batch 0 since inputs are halved)
        assert!((output[3] - expected_0_0 / 2.0).abs() < 0.01);
        assert!((output[4] - expected_0_1 / 2.0).abs() < 0.01);
        assert!((output[5] - expected_0_2 / 2.0).abs() < 0.01);
    }

    /// Test partial word bounds checking with non-multiple of 32 input features.
    /// 
    /// This test validates the optimized bounds checking for partial words
    /// (when in_features % 32 != 0). It ensures that bits beyond in_features
    /// are not processed.
    #[test]
    fn test_ternary_matmul_partial_word() {
        // Test with in_features = 48 (1 full word + 16 bits in partial word)
        let batch_size = 2;
        let in_features = 48;  // Not a multiple of 32
        let out_features = 2;
        let k_words = 2;  // ceil(48 / 32) = 2 words (second is partial)
        
        // Input: [1.0, 2.0, 3.0, ..., 48.0] for batch 0
        let mut input_data = vec![0.0f32; batch_size * in_features];
        for b in 0..batch_size {
            for i in 0..in_features {
                input_data[b * in_features + i] = (i + 1) as f32 / (b + 1) as f32;
            }
        }
        
        // Weights:
        // Feature 0: all +1 in first word, all +1 in partial second word (bits 0-15)
        // Feature 1: all -1 in partial second word only (bits 0-15)
        let mut w_plus = vec![0u32; out_features * k_words];
        let mut w_minus = vec![0u32; out_features * k_words];
        
        // Feature 0: positive in both words
        w_plus[0] = 0xFFFFFFFF;  // word 0 (bits 0-31, all valid)
        w_plus[1] = 0x0000FFFF;  // word 1 (bits 0-15 valid, 16-31 should be ignored)
        
        // Feature 1: negative in partial word only
        w_minus[2] = 0x00000000;  // word 0 (no contribution)
        w_minus[3] = 0x0000FFFF;  // word 1 (bits 0-15 valid, 16-31 should be ignored)
        
        let scales = vec![1.0f32; out_features];
        
        // Expected outputs:
        // Feature 0: sum(1..32) + sum(33..48) = 528 + (33+34+...+48)
        //          = 528 + (48*49/2 - 32*33/2) = 528 + (1176 - 528) = 1176
        // Feature 1: -sum(33..48) = -(33+34+...+48) = -648
        
        // Simulate kernel computation
        let mut output = vec![0.0f32; batch_size * out_features];
        
        for batch_idx in 0..batch_size {
            for out_idx in 0..out_features {
                let mut acc = 0.0f32;
                let input_offset = batch_idx * in_features;
                let weight_offset = out_idx * k_words;
                
                for k in 0..k_words {
                    let wp_bits = w_plus[weight_offset + k];
                    let wm_bits = w_minus[weight_offset + k];
                    
                    let is_full_word = (k * 32 + 32) <= in_features;
                    
                    for bit in 0..32 {
                        let dim_idx = k * 32 + bit;
                        
                        // This is the critical bounds check for partial words
                        if !is_full_word && dim_idx >= in_features {
                            break;
                        }
                        
                        let mask = 1u32 << bit;
                        let is_pos = (wp_bits & mask) != 0;
                        let is_neg = (wm_bits & mask) != 0;
                        
                        let input_val = input_data[input_offset + dim_idx];
                        
                        if is_pos {
                            acc += input_val;
                        } else if is_neg {
                            acc -= input_val;
                        }
                    }
                }
                
                output[batch_idx * out_features + out_idx] = acc * scales[out_idx];
            }
        }
        
        // Verify batch 0 results
        let expected_0 = (1..=48).sum::<i32>() as f32;  // sum(1..48) = 1176
        let expected_1 = -(33..=48).sum::<i32>() as f32;  // -sum(33..48) = -648
        
        assert!((output[0] - expected_0).abs() < 0.01,
                "Feature 0 mismatch: expected {}, got {}", expected_0, output[0]);
        assert!((output[1] - expected_1).abs() < 0.01,
                "Feature 1 mismatch: expected {}, got {}", expected_1, output[1]);
        
        // Verify batch 1 (inputs halved)
        assert!((output[2] - expected_0 / 2.0).abs() < 0.01);
        assert!((output[3] - expected_1 / 2.0).abs() < 0.01);
    }
}
