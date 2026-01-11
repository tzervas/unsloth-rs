// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! Ternary attention scoring for memory-efficient transformer inference.
//!
//! This module provides ternary bitsliced attention computation using
//! popcount-based Q·K^T scoring with online softmax for numerical stability.

use super::types::{TernaryPlanes, TernaryTensor};
use crate::error::{Result, UnslothError};
use candle_core::Tensor;

/// Configuration for ternary attention.
#[derive(Debug, Clone)]
pub struct TernaryAttentionConfig {
    /// Number of attention heads
    pub num_heads: usize,
    /// Dimension per head
    pub head_dim: usize,
    /// Whether to apply causal masking
    pub causal: bool,
    /// Sparsity threshold for hybrid dispatch (use FP if below this)
    pub sparsity_threshold: f32,
}

impl Default for TernaryAttentionConfig {
    fn default() -> Self {
        Self {
            num_heads: 12,
            head_dim: 64,
            causal: true,
            sparsity_threshold: 0.8,
        }
    }
}

/// Ternary attention weights (Q, K, V projections as ternary tensors).
///
/// # Note on Per-Head Scales
///
/// The `q_scales` and `k_scales` fields provide per-head scale factors for use
/// with the `ternary_attention_score()` function in future GPU implementations
/// using popcount-based scoring. The current `ternary_attention_cpu()` reference
/// implementation uses standard tensor operations and does not utilize these
/// scales directly, but they are included for API completeness and future GPU
/// kernel integration.
#[derive(Debug, Clone)]
pub struct TernaryAttentionWeights {
    /// Query projection weights
    pub q_proj: TernaryTensor,
    /// Key projection weights
    pub k_proj: TernaryTensor,
    /// Value projection weights
    pub v_proj: TernaryTensor,
    /// Output projection weights
    pub o_proj: TernaryTensor,
    /// Per-head Q scale factors (for future GPU popcount-based scoring)
    pub q_scales: Vec<f32>,
    /// Per-head K scale factors (for future GPU popcount-based scoring)
    pub k_scales: Vec<f32>,
}

/// Compute ternary attention score between Q and K planes via popcount.
///
/// Score = (popcount(Q+ & K+) + popcount(Q- & K-)
///        - popcount(Q+ & K-) - popcount(Q- & K+)) * `scale_q` * `scale_k`
///
/// # Arguments
///
/// * `q_planes` - Query ternary planes for one position
/// * `k_planes` - Key ternary planes for one position
/// * `scale_q` - Query scale factor
/// * `scale_k` - Key scale factor
///
/// # Returns
///
/// Scaled attention score as f32
#[must_use]
pub fn ternary_attention_score(
    q_planes: &TernaryPlanes,
    k_planes: &TernaryPlanes,
    scale_q: f32,
    scale_k: f32,
) -> f32 {
    let dot = q_planes.dot(k_planes);
    dot as f32 * scale_q * scale_k
}

/// Online softmax state for numerically stable attention.
///
/// Maintains running max and sum for the online softmax algorithm.
/// This implements the online softmax algorithm to avoid storing the full
/// `O(seq_len²)` attention matrix, enabling memory-efficient attention computation.
///
/// # Note on CPU Implementation
///
/// The current `ternary_attention_cpu()` reference implementation uses standard
/// tensor operations with `candle_nn::ops::softmax` for validation purposes.
/// This `OnlineSoftmaxState` type is designed for future GPU kernel integration
/// where memory-efficient online softmax becomes critical for long sequences.
#[derive(Debug, Clone)]
pub struct OnlineSoftmaxState {
    /// Running maximum score (for numerical stability)
    pub max: f32,
    /// Running sum of exp(score - max)
    pub sum: f32,
    /// Running weighted output accumulator
    pub output: Vec<f32>,
}

impl OnlineSoftmaxState {
    /// Create new state for given output dimension.
    #[must_use]
    pub fn new(dim: usize) -> Self {
        Self {
            max: f32::NEG_INFINITY,
            sum: 0.0,
            output: vec![0.0; dim],
        }
    }

    /// Update state with a new score and corresponding value.
    ///
    /// Implements online softmax: maintains running statistics
    /// to avoid storing the full attention matrix.
    ///
    /// # Arguments
    ///
    /// * `score` - New attention score
    /// * `value` - Corresponding value vector
    ///
    /// # Panics
    ///
    /// Panics if value length doesn't match output dimension.
    pub fn update(&mut self, score: f32, value: &[f32]) {
        assert_eq!(value.len(), self.output.len(), "value dimension mismatch");

        if score > self.max {
            // New maximum: rescale existing accumulator
            let correction = (self.max - score).exp();
            self.sum *= correction;
            for o in &mut self.output {
                *o *= correction;
            }
            self.max = score;
        }

        // Add contribution from this score
        let exp_score = (score - self.max).exp();
        self.sum += exp_score;
        for (o, &v) in self.output.iter_mut().zip(value.iter()) {
            *o += exp_score * v;
        }
    }

    /// Finalize and return normalized output.
    ///
    /// # Returns
    ///
    /// The normalized output vector. If `sum == 0.0` (which can occur if all scores
    /// were negative infinity due to causal masking), returns the unnormalized
    /// output (a vector of zeros) to avoid division by zero.
    #[must_use]
    pub fn finalize(self) -> Vec<f32> {
        if self.sum == 0.0 {
            // All scores were masked out (negative infinity), return zeros
            return self.output;
        }
        self.output.into_iter().map(|o| o / self.sum).collect()
    }
}

/// Apply causal masking to ternary planes by zeroing future positions.
///
/// For position `query_pos`, zeros out all key positions > `query_pos`
/// in both +plane and -plane. This implements Task 3.3 from the ternary GPU
/// implementation plan: causal masking via bitplane zeroing.
///
/// # Note on CPU Implementation
///
/// The current `ternary_attention_cpu()` reference implementation uses standard
/// tensor-based causal masking for validation purposes. This function provides
/// the bit-level masking operation for future GPU kernel integration where
/// bitplane operations are more efficient.
///
/// # Arguments
///
/// * `planes` - Ternary planes to mask (modified in place)
/// * `query_pos` - Current query position
/// * `seq_len` - Total sequence length
pub fn apply_causal_mask_to_planes(planes: &mut TernaryPlanes, query_pos: usize, seq_len: usize) {
    // Zero out positions after query_pos
    for pos in (query_pos + 1)..seq_len {
        if pos < planes.num_dims {
            let word_idx = pos / 32;
            let bit_idx = pos % 32;
            let mask = !(1u32 << bit_idx);

            if word_idx < planes.plus.len() {
                planes.plus[word_idx] &= mask;
                planes.minus[word_idx] &= mask;
            }
        }
    }
}

/// CPU reference implementation of ternary attention.
///
/// Computes attention using ternary Q, K, V projections. This is a validation
/// baseline for future GPU kernels and uses standard tensor operations.
///
/// # Implementation Notes
///
/// This CPU reference implementation uses:
/// - Standard `matmul` for Q·K^T scoring (not popcount-based `ternary_attention_score`)
/// - Standard `candle_nn::ops::softmax` (not memory-efficient `OnlineSoftmaxState`)
/// - Standard tensor-based causal mask (not bitplane-level `apply_causal_mask_to_planes`)
///
/// These design choices prioritize correctness validation over performance. The
/// corresponding GPU-optimized implementations will use the popcount-based scoring,
/// online softmax, and bitplane masking functions defined in this module.
///
/// # Arguments
///
/// * `hidden_states` - Input tensor [batch, `seq_len`, hidden]
/// * `weights` - Ternary attention weights
/// * `config` - Attention configuration
///
/// # Returns
///
/// Output tensor [batch, `seq_len`, hidden]
///
/// # Errors
///
/// Returns error if shapes don't match or computation fails.
pub fn ternary_attention_cpu(
    hidden_states: &Tensor,
    weights: &TernaryAttentionWeights,
    config: &TernaryAttentionConfig,
) -> Result<Tensor> {
    use super::matmul::ternary_matmul_cpu;

    let dims = hidden_states.dims();
    if dims.len() != 3 {
        return Err(UnslothError::ShapeMismatch {
            // Expected a 3D tensor (rank 3) for [batch, seq_len, hidden]
            expected: vec![3],
            actual: dims.to_vec(),
        });
    }

    let (batch, seq_len, _hidden) = (dims[0], dims[1], dims[2]);
    let num_heads = config.num_heads;
    let head_dim = config.head_dim;

    // Project to Q, K, V using ternary matmul
    let q = ternary_matmul_cpu(hidden_states, &weights.q_proj)?;
    let k = ternary_matmul_cpu(hidden_states, &weights.k_proj)?;
    let v = ternary_matmul_cpu(hidden_states, &weights.v_proj)?;

    // Reshape: [batch, seq, num_heads * head_dim] -> [batch, num_heads, seq, head_dim]
    let q = q
        .reshape((batch, seq_len, num_heads, head_dim))?
        .transpose(1, 2)?;
    let k = k
        .reshape((batch, seq_len, num_heads, head_dim))?
        .transpose(1, 2)?;
    let v = v
        .reshape((batch, seq_len, num_heads, head_dim))?
        .transpose(1, 2)?;

    // Compute attention scores: Q @ K^T / sqrt(head_dim)
    let scale = (head_dim as f64).sqrt();
    let scores = q.matmul(&k.transpose(2, 3)?)?;
    let scores = (scores / scale)?;

    // Apply causal mask if needed
    let scores = if config.causal {
        let mask = create_causal_mask(seq_len, hidden_states.device())?;
        // Reshape mask from [seq_len, seq_len] to [1, 1, seq_len, seq_len] for broadcasting
        let mask = mask.reshape((1, 1, seq_len, seq_len))?;
        scores.broadcast_add(&mask)?
    } else {
        scores
    };

    // Softmax
    let attn_weights = candle_nn::ops::softmax(&scores, 3)?;

    // Attention output: attn_weights @ V
    let attn_output = attn_weights.matmul(&v)?;

    // Reshape back: [batch, num_heads, seq, head_dim] -> [batch, seq, hidden]
    let attn_output =
        attn_output
            .transpose(1, 2)?
            .reshape((batch, seq_len, num_heads * head_dim))?;

    // Output projection
    let output = ternary_matmul_cpu(&attn_output, &weights.o_proj)?;

    Ok(output)
}

/// Create a causal attention mask.
fn create_causal_mask(seq_len: usize, device: &candle_core::Device) -> Result<Tensor> {
    // Create a mask where mask[i,j] = -inf if j > i (future positions)
    let mut mask_data = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            if j > i {
                mask_data[i * seq_len + j] = f32::NEG_INFINITY;
            }
        }
    }
    let mask = Tensor::from_vec(mask_data, (seq_len, seq_len), device)?;
    Ok(mask)
}

/// Check if ternary attention should be used based on sparsity.
#[must_use]
pub fn should_use_ternary_attention(
    weights: &TernaryAttentionWeights,
    config: &TernaryAttentionConfig,
) -> bool {
    let avg_sparsity = (weights.q_proj.sparsity()
        + weights.k_proj.sparsity()
        + weights.v_proj.sparsity()
        + weights.o_proj.sparsity())
        / 4.0;

    avg_sparsity >= config.sparsity_threshold
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ternary_attention_score() {
        let mut q = TernaryPlanes::new(64);
        let mut k = TernaryPlanes::new(64);

        // Set some matching values
        q.set(0, 1);
        q.set(1, -1);
        k.set(0, 1);
        k.set(1, -1);

        // Score should be 2 * scales
        let score = ternary_attention_score(&q, &k, 1.0, 1.0);
        assert!((score - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_online_softmax() {
        let mut state = OnlineSoftmaxState::new(4);

        // Add some scores and values
        state.update(1.0, &[1.0, 0.0, 0.0, 0.0]);
        state.update(2.0, &[0.0, 1.0, 0.0, 0.0]);
        state.update(1.0, &[0.0, 0.0, 1.0, 0.0]);

        let output = state.finalize();

        // Check output is normalized (sums to 1 per "head")
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_causal_mask_planes() {
        let mut planes = TernaryPlanes::new(8);

        // Set all to +1
        for i in 0..8 {
            planes.set(i, 1);
        }

        // Apply causal mask at position 3
        apply_causal_mask_to_planes(&mut planes, 3, 8);

        // Positions 0-3 should still be +1
        assert_eq!(planes.get(0), 1);
        assert_eq!(planes.get(3), 1);

        // Positions 4-7 should be 0
        assert_eq!(planes.get(4), 0);
        assert_eq!(planes.get(7), 0);
    }

    #[test]
    fn test_attention_config_default() {
        let config = TernaryAttentionConfig::default();
        assert_eq!(config.num_heads, 12);
        assert_eq!(config.head_dim, 64);
        assert!(config.causal);
    }

    #[test]
    fn test_should_use_ternary_attention_high_sparsity() {
        let shape = (64, 64);
        let k_words = 2;
        // Create tensors with high sparsity (all zeros = 100% sparsity)
        let plus = vec![0u32; 64 * k_words];
        let minus = vec![0u32; 64 * k_words];
        let scales = vec![1.0f32; 64];

        let weights = TernaryAttentionWeights {
            q_proj: TernaryTensor::new(plus.clone(), minus.clone(), scales.clone(), shape),
            k_proj: TernaryTensor::new(plus.clone(), minus.clone(), scales.clone(), shape),
            v_proj: TernaryTensor::new(plus.clone(), minus.clone(), scales.clone(), shape),
            o_proj: TernaryTensor::new(plus, minus, scales, shape),
            q_scales: vec![1.0; 12],
            k_scales: vec![1.0; 12],
        };

        let config = TernaryAttentionConfig {
            sparsity_threshold: 0.8,
            ..Default::default()
        };

        // 100% sparsity should exceed 0.8 threshold
        assert!(should_use_ternary_attention(&weights, &config));
    }

    #[test]
    fn test_should_use_ternary_attention_low_sparsity() {
        let shape = (64, 64);
        let k_words = 2;
        // Create tensors with low sparsity (all +1)
        let plus = vec![u32::MAX; 64 * k_words];
        let minus = vec![0u32; 64 * k_words];
        let scales = vec![1.0f32; 64];

        let weights = TernaryAttentionWeights {
            q_proj: TernaryTensor::new(plus.clone(), minus.clone(), scales.clone(), shape),
            k_proj: TernaryTensor::new(plus.clone(), minus.clone(), scales.clone(), shape),
            v_proj: TernaryTensor::new(plus.clone(), minus.clone(), scales.clone(), shape),
            o_proj: TernaryTensor::new(plus, minus, scales, shape),
            q_scales: vec![1.0; 12],
            k_scales: vec![1.0; 12],
        };

        let config = TernaryAttentionConfig {
            sparsity_threshold: 0.8,
            ..Default::default()
        };

        // Low sparsity should not exceed 0.8 threshold
        assert!(!should_use_ternary_attention(&weights, &config));
    }

    #[test]
    fn test_should_use_ternary_attention_at_threshold() {
        let shape = (64, 64);
        let k_words = 2;
        // Create tensors with exactly 80% sparsity
        let plus = vec![0u32; 64 * k_words];
        let minus = vec![0u32; 64 * k_words];
        let scales = vec![1.0f32; 64];

        let weights = TernaryAttentionWeights {
            q_proj: TernaryTensor::new(plus.clone(), minus.clone(), scales.clone(), shape),
            k_proj: TernaryTensor::new(plus.clone(), minus.clone(), scales.clone(), shape),
            v_proj: TernaryTensor::new(plus.clone(), minus.clone(), scales.clone(), shape),
            o_proj: TernaryTensor::new(plus, minus, scales, shape),
            q_scales: vec![1.0; 12],
            k_scales: vec![1.0; 12],
        };

        let config = TernaryAttentionConfig {
            sparsity_threshold: 1.0, // Exact threshold
            ..Default::default()
        };

        // At exactly 100% sparsity with 1.0 threshold, should use ternary
        assert!(should_use_ternary_attention(&weights, &config));
    }

    #[test]
    fn test_online_softmax_all_masked() {
        // Test behavior when sum == 0.0 (all masked)
        let state = OnlineSoftmaxState::new(4);

        // Don't update with any scores - simulates all masked case
        let output = state.finalize();

        // Should return zeros without division by zero
        assert_eq!(output, vec![0.0, 0.0, 0.0, 0.0]);
    }
}
