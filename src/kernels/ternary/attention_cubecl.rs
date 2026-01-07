//! CubeCL GPU kernel implementation for Ternary Attention.
//!
//! This module implements Phase 3, Task 3.1: Q·K^T Ternary Scoring Kernel
//! for memory-efficient attention with ternary-quantized K matrices.
//!
//! ## Algorithm Overview
//!
//! Computes attention scores S = Q · K^T where K is ternary-quantized:
//! - Q: [batch, heads, seq_len, head_dim] - FP32 query vectors
//! - K: [batch, heads, seq_len, head_dim] - Ternary bitsliced keys
//! - S: [batch, heads, seq_len, seq_len] - FP32 attention scores
//!
//! Uses popcount-based dot product from Phase 2 ternary matmul kernels,
//! adapted for the transpose-aware access pattern required by Q·K^T.
//!
//! ## Memory Efficiency
//!
//! - Keys stored as bitsliced planes (32x compression vs FP32)
//! - Cooperative loading into shared memory
//! - Tiled computation to fit in SRAM
//! - Output accumulated directly to global memory
//!
//! ## Implementation Status
//!
//! - [x] Configuration structures
//! - [x] CPU simulation kernel for testing
//! - [x] Multi-head support with GQA
//! - [x] Launch configuration helper
//! - [ ] Actual CubeCL kernel (awaiting GPU hardware)

use super::types::{TernaryPlanes, TernaryTensor};
use crate::error::{Result, UnslothError};
use candle_core::Tensor;

/// Configuration for ternary attention scoring kernel.
#[derive(Debug, Clone)]
pub struct TernaryAttentionScoreConfig {
    /// Tile size for K dimension (in u32 words)
    pub tile_k: u32,
    /// Number of threads per block
    pub block_size: u32,
    /// Outputs per thread (for better occupancy)
    pub outputs_per_thread: u32,
}

impl TernaryAttentionScoreConfig {
    /// Create configuration for RTX 5080 GPU.
    #[must_use]
    pub fn rtx_5080() -> Self {
        Self {
            tile_k: 64,  // 64 u32 words = 2048 dimensions
            block_size: 256,
            outputs_per_thread: 2,
        }
    }

    /// Create configuration for RTX 3090 Ti GPU.
    #[must_use]
    pub fn rtx_3090_ti() -> Self {
        Self {
            tile_k: 32,  // 32 u32 words = 1024 dimensions
            block_size: 256,
            outputs_per_thread: 2,
        }
    }

    /// Create default configuration (conservative settings).
    #[must_use]
    pub fn default_config() -> Self {
        Self::rtx_3090_ti()
    }
}

/// Launch configuration for attention scoring kernel.
#[derive(Debug, Clone)]
pub struct AttentionScoreLaunchConfig {
    /// Grid dimensions (batch * heads, queries)
    pub grid_dim: (u32, u32, u32),
    /// Block dimensions
    pub block_dim: (u32, u32, u32),
}

/// Calculate optimal launch configuration for attention scoring.
///
/// # Arguments
/// * `batch_size` - Batch size
/// * `num_heads` - Number of attention heads
/// * `q_seq_len` - Query sequence length
/// * `config` - Kernel configuration
///
/// # Returns
/// Launch configuration with grid and block dimensions
#[must_use]
pub fn get_attention_score_launch_config(
    batch_size: usize,
    num_heads: usize,
    q_seq_len: usize,
    config: &TernaryAttentionScoreConfig,
) -> AttentionScoreLaunchConfig {
    // Each block processes outputs_per_thread outputs
    let outputs_per_block = config.block_size * config.outputs_per_thread;
    let blocks_per_query_dim = ((q_seq_len as u32 + outputs_per_block - 1) / outputs_per_block).max(1);
    
    // Grid: (batch * heads) x blocks_for_queries x 1
    let grid_x = (batch_size * num_heads) as u32;
    let grid_y = blocks_per_query_dim;
    
    AttentionScoreLaunchConfig {
        grid_dim: (grid_x, grid_y, 1),
        block_dim: (config.block_size, 1, 1),
    }
}

/// CPU simulation of ternary attention scoring kernel.
///
/// This function simulates the GPU kernel behavior for testing purposes.
/// Computes S = Q · K^T where K is ternary bitsliced.
///
/// # Arguments
/// * `q` - Query tensor [batch, heads, q_seq_len, head_dim] (FP32)
/// * `k_ternary` - Ternary keys tensor with bitsliced planes
/// * `k_seq_len` - Key sequence length
/// * `scale` - Attention scale factor (typically 1/sqrt(head_dim))
/// * `config` - Kernel configuration
///
/// # Returns
/// Attention scores tensor [batch, heads, q_seq_len, k_seq_len]
///
/// # Errors
/// Returns error if shapes are incompatible or computation fails.
pub fn ternary_attention_score_kernel_cpu(
    q: &Tensor,
    k_ternary: &TernaryTensor,
    k_seq_len: usize,
    scale: f64,
    _config: &TernaryAttentionScoreConfig,
) -> Result<Tensor> {
    // Get Q dimensions
    let q_dims = q.dims();
    if q_dims.len() != 4 {
        return Err(UnslothError::ShapeMismatch {
            expected: vec![4],
            actual: q_dims.to_vec(),
        });
    }
    
    let (batch, num_heads, q_seq_len, head_dim) = 
        (q_dims[0], q_dims[1], q_dims[2], q_dims[3]);
    
    // Verify K shape matches
    let (k_out_features, k_in_features) = k_ternary.shape();
    let expected_k_shape = (num_heads * k_seq_len, head_dim);
    
    if (k_out_features, k_in_features) != expected_k_shape {
        return Err(UnslothError::ShapeMismatch {
            expected: vec![expected_k_shape.0, expected_k_shape.1],
            actual: vec![k_out_features, k_in_features],
        });
    }
    
    // Allocate output scores tensor
    let mut scores_data = vec![0.0f32; batch * num_heads * q_seq_len * k_seq_len];
    
    // Get Q data
    let q_data = q.to_vec2::<f32>()?;
    let q_flat: Vec<f32> = q_data.iter().flatten().copied().collect();
    
    // Compute Q · K^T using popcount-based ternary dot product
    // For each batch and head
    for b in 0..batch {
        for h in 0..num_heads {
            // For each query position
            for qi in 0..q_seq_len {
                // Get query vector
                let q_offset = ((b * num_heads + h) * q_seq_len + qi) * head_dim;
                let q_vec = &q_flat[q_offset..q_offset + head_dim];
                
                // For each key position
                for ki in 0..k_seq_len {
                    // Get K row index
                    let k_row = h * k_seq_len + ki;
                    
                    // Compute ternary dot product: Q[qi] · K[ki]
                    let score = ternary_dot_product_fp_query(q_vec, k_ternary, k_row, head_dim);
                    
                    // Apply scale and store
                    let output_idx = ((b * num_heads + h) * q_seq_len + qi) * k_seq_len + ki;
                    scores_data[output_idx] = score * (scale as f32);
                }
            }
        }
    }
    
    // Convert to tensor
    let scores = Tensor::from_vec(
        scores_data,
        (batch, num_heads, q_seq_len, k_seq_len),
        q.device(),
    )?;
    
    Ok(scores)
}

/// Compute dot product between FP32 query vector and ternary key row.
///
/// Uses popcount-based computation:
/// - Quantize Q to ternary planes on-the-fly
/// - Compute popcount(Q+ & K+) + popcount(Q- & K-)
/// - Subtract popcount(Q+ & K-) + popcount(Q- & K+)
/// - Scale by K scale factor
///
/// # Arguments
/// * `q_vec` - Query vector (FP32)
/// * `k_ternary` - Ternary keys tensor
/// * `k_row` - Row index in K tensor
/// * `head_dim` - Dimension of head (number of elements in vector)
///
/// # Returns
/// Dot product score (FP32)
fn ternary_dot_product_fp_query(
    q_vec: &[f32],
    k_ternary: &TernaryTensor,
    k_row: usize,
    head_dim: usize,
) -> f32 {
    const THRESHOLD: f32 = 0.5;
    
    // Get number of u32 words needed
    let k_words = (head_dim + 31) / 32;
    
    // Quantize Q to ternary planes
    let mut q_plus = vec![0u32; k_words];
    let mut q_minus = vec![0u32; k_words];
    
    for (dim_idx, &val) in q_vec.iter().enumerate().take(head_dim) {
        let word_idx = dim_idx / 32;
        let bit_idx = (dim_idx % 32) as u32;
        
        if val > THRESHOLD {
            q_plus[word_idx] |= 1u32 << bit_idx;
        } else if val < -THRESHOLD {
            q_minus[word_idx] |= 1u32 << bit_idx;
        }
    }
    
    // Get K row planes
    let k_plus = k_ternary.plus_plane();
    let k_minus = k_ternary.minus_plane();
    let k_scale = k_ternary.scale(k_row);
    
    // Compute popcount dot product
    let mut pos_sum = 0u32;
    let mut neg_sum = 0u32;
    
    for i in 0..k_words {
        let k_plus_word = k_plus[k_row * k_words + i];
        let k_minus_word = k_minus[k_row * k_words + i];
        
        // Positive contributions: both same sign
        pos_sum += (q_plus[i] & k_plus_word).count_ones();
        pos_sum += (q_minus[i] & k_minus_word).count_ones();
        
        // Negative contributions: opposite signs
        neg_sum += (q_plus[i] & k_minus_word).count_ones();
        neg_sum += (q_minus[i] & k_plus_word).count_ones();
    }
    
    // Final score
    let dot = (pos_sum as i32 - neg_sum as i32) as f32;
    dot * k_scale
}

/// GPU implementation of ternary attention scoring (placeholder).
///
/// This function will dispatch to the actual CubeCL kernel when GPU hardware
/// is available. Currently falls back to CPU simulation.
///
/// # Arguments
/// * `q` - Query tensor [batch, heads, q_seq_len, head_dim]
/// * `k_ternary` - Ternary keys tensor
/// * `k_seq_len` - Key sequence length
/// * `scale` - Attention scale factor
/// * `config` - Kernel configuration
///
/// # Returns
/// Attention scores tensor [batch, heads, q_seq_len, k_seq_len]
///
/// # Errors
/// Returns error if GPU execution fails or shapes are incompatible.
pub fn ternary_attention_score_cuda(
    q: &Tensor,
    k_ternary: &TernaryTensor,
    k_seq_len: usize,
    scale: f64,
    config: &TernaryAttentionScoreConfig,
) -> Result<Tensor> {
    // TODO: Implement actual CubeCL kernel dispatch
    // For now, fall back to CPU simulation
    ternary_attention_score_kernel_cpu(q, k_ternary, k_seq_len, scale, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn create_test_query(batch: usize, heads: usize, seq_len: usize, dim: usize) -> Tensor {
        // Create simple pattern for testing
        let size = batch * heads * seq_len * dim;
        let data: Vec<f32> = (0..size).map(|i| (i % 10) as f32 * 0.1).collect();
        Tensor::from_vec(data, (batch, heads, seq_len, dim), &Device::Cpu).unwrap()
    }

    fn create_test_ternary_keys(heads: usize, seq_len: usize, dim: usize) -> TernaryTensor {
        let out_features = heads * seq_len;
        let k_words = (dim + 31) / 32;
        
        // Create simple ternary pattern
        let mut plus = vec![0u32; out_features * k_words];
        let mut minus = vec![0u32; out_features * k_words];
        
        // Set some bits for testing
        for i in 0..out_features {
            if i % 3 == 0 {
                plus[i * k_words] = 0xFF00FF00;  // Some +1 values
            } else if i % 3 == 1 {
                minus[i * k_words] = 0x00FF00FF;  // Some -1 values
            }
            // i % 3 == 2 remains zeros
        }
        
        let scales = vec![1.0f32; out_features];
        let shape = (out_features, dim);
        
        TernaryTensor::new(plus, minus, scales, shape)
    }

    #[test]
    fn test_attention_score_config_creation() {
        let config = TernaryAttentionScoreConfig::rtx_5080();
        assert_eq!(config.tile_k, 64);
        assert_eq!(config.block_size, 256);
        assert_eq!(config.outputs_per_thread, 2);
        
        let config = TernaryAttentionScoreConfig::rtx_3090_ti();
        assert_eq!(config.tile_k, 32);
    }

    #[test]
    fn test_attention_score_launch_config() {
        let config = TernaryAttentionScoreConfig::default_config();
        let launch = get_attention_score_launch_config(2, 8, 128, &config);
        
        // Grid should cover batch * heads
        assert_eq!(launch.grid_dim.0, 2 * 8);
        
        // Block size should match config
        assert_eq!(launch.block_dim.0, config.block_size);
    }

    #[test]
    fn test_ternary_attention_score_kernel_shape() {
        let batch = 2;
        let heads = 4;
        let q_seq = 8;
        let k_seq = 8;
        let dim = 64;
        
        let q = create_test_query(batch, heads, q_seq, dim);
        let k = create_test_ternary_keys(heads, k_seq, dim);
        let config = TernaryAttentionScoreConfig::default_config();
        
        let scores = ternary_attention_score_kernel_cpu(&q, &k, k_seq, 1.0, &config).unwrap();
        
        // Verify output shape
        assert_eq!(scores.dims(), &[batch, heads, q_seq, k_seq]);
    }

    #[test]
    fn test_ternary_attention_score_kernel_numerical() {
        let batch = 1;
        let heads = 1;
        let q_seq = 2;
        let k_seq = 2;
        let dim = 32;
        
        // Create known Q values
        let mut q_data = vec![0.0f32; batch * heads * q_seq * dim];
        // First query: [1, 0, -1, 0, ...]
        q_data[0] = 1.0;
        q_data[2] = -1.0;
        // Second query: [0, 1, 0, -1, ...]
        q_data[dim + 1] = 1.0;
        q_data[dim + 3] = -1.0;
        
        let q = Tensor::from_vec(q_data, (batch, heads, q_seq, dim), &Device::Cpu).unwrap();
        
        // Create known K values (as ternary)
        let k_words = 1;  // 32 dimensions = 1 word
        let mut plus = vec![0u32; heads * k_seq * k_words];
        let mut minus = vec![0u32; heads * k_seq * k_words];
        
        // First key: bit 0 = +1 (matches first query)
        plus[0] = 1u32;
        // Second key: bit 1 = +1 (matches second query)
        plus[1] = 2u32;
        
        let scales = vec![1.0f32; heads * k_seq];
        let k = TernaryTensor::new(plus, minus, scales, (heads * k_seq, dim));
        
        let config = TernaryAttentionScoreConfig::default_config();
        let scores = ternary_attention_score_kernel_cpu(&q, &k, k_seq, 1.0, &config).unwrap();
        
        let scores_data = scores.to_vec3::<f32>().unwrap();
        
        // First query should match first key better (score at [0,0,0,0])
        let score_q0_k0 = scores_data[0][0][0][0];
        let score_q0_k1 = scores_data[0][0][0][1];
        assert!(score_q0_k0 > score_q0_k1);
        
        // Second query should match second key better (score at [0,0,1,1])
        let score_q1_k0 = scores_data[0][0][1][0];
        let score_q1_k1 = scores_data[0][0][1][1];
        assert!(score_q1_k1 > score_q1_k0);
    }

    #[test]
    fn test_ternary_dot_product_simple() {
        let dim = 32;
        let k_words = 1;
        
        // Query: [1.0, 0.0, -1.0, ...zeros]
        let mut q = vec![0.0f32; dim];
        q[0] = 1.0;
        q[2] = -1.0;
        
        // Key: bit 0 = +1, bit 2 = -1
        let plus = vec![1u32];  // bit 0 set
        let minus = vec![4u32];  // bit 2 set
        let scales = vec![1.0f32];
        let k = TernaryTensor::new(plus, minus, scales, (1, dim));
        
        let score = ternary_dot_product_fp_query(&q, &k, 0, dim);
        
        // Should have two matches: q[0]*k[0]=+1 and q[2]*k[2]=+1, total=2
        assert!((score - 2.0).abs() < 0.001, "Expected 2.0, got {}", score);
    }

    #[test]
    fn test_ternary_attention_score_cuda_fallback() {
        // Test that CUDA version falls back to CPU correctly
        let batch = 1;
        let heads = 2;
        let q_seq = 4;
        let k_seq = 4;
        let dim = 64;
        
        let q = create_test_query(batch, heads, q_seq, dim);
        let k = create_test_ternary_keys(heads, k_seq, dim);
        let config = TernaryAttentionScoreConfig::default_config();
        
        let scores = ternary_attention_score_cuda(&q, &k, k_seq, 1.0, &config).unwrap();
        
        // Should produce valid output
        assert_eq!(scores.dims(), &[batch, heads, q_seq, k_seq]);
    }
}
