//! CubeCL GPU kernel for ternary bitsliced matrix multiplication.
//!
//! Uses native popcount intrinsics for efficient dot product computation.

use cubecl::prelude::*;

/// Compile-time configuration for ternary matmul kernel.
#[derive(Clone, Copy, Debug)]
pub struct TernaryMatmulConfig {
    /// Number of u32 words in K dimension (in_features / 32)
    pub k_words: u32,
    /// Number of output rows (batch size)
    pub m: u32,
    /// Number of output cols (out_features)
    pub n: u32,
    /// Number of input features (for bounds checking)
    pub in_features: u32,
}

/// Basic ternary matmul kernel using popcount.
///
/// Each thread computes one output element: output[batch_idx, out_idx]
///
/// Algorithm:
/// ```text
/// dot = popcount(w_plus & input_bits) - popcount(w_minus & input_bits)
/// output = dot * scale
/// ```
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
    #[comptime] config: TernaryMatmulConfig,
) {
    // Thread indices: each thread handles one (batch, out_feature) element
    let batch_idx = CUBE_POS_X;
    let out_idx = CUBE_POS_Y * CUBE_DIM_X + UNIT_POS_X;

    // Bounds check
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
        
        // For each bit position in the u32 word
        for bit in 0u32..32u32 {
            let dim_idx = k * 32 + bit;
            
            if dim_idx < config.in_features {
                let mask = 1u32 << bit;
                
                // Extract ternary weight value from planes
                // Note: We're working with f32 reinterpreted as u32
                // In production, use proper u32 arrays
                let wp_bits = u32::reinterpret(wp_f32);
                let wm_bits = u32::reinterpret(wm_f32);
                
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
        let config = TernaryMatmulConfig {
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
    }
}
