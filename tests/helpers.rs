//! Test utilities and fixtures for ternary quantization integration tests.
//!
//! This module provides reusable test data generation, comparison utilities,
//! and common fixtures for integration testing.

use anyhow::Result;
use candle_core::{Device, Tensor};
use std::collections::HashMap;

/// Test configuration for generating various tensor patterns.
#[derive(Debug, Clone)]
pub struct TestMatrixConfig {
    /// Matrix shape (rows, cols).
    pub shape: (usize, usize),
    /// Target sparsity level (0.0 = dense, 0.99 = 99% sparse).
    pub sparsity: f32,
    /// Value distribution type.
    pub distribution: ValueDistribution,
    /// Random seed for reproducible tests.
    pub seed: u64,
}

/// Different value distributions for test matrices.
#[derive(Debug, Clone)]
pub enum ValueDistribution {
    /// Uniform distribution in [-max, max].
    Uniform { max: f32 },
    /// Normal distribution with mean=0, std=sigma.
    Normal { std: f32 },
    /// Long-tail distribution (few large values, many small).
    LongTail { scale: f32 },
    /// Specific values for edge cases.
    EdgeCase(EdgeCaseType),
}

/// Edge case patterns for robustness testing.
#[derive(Debug, Clone)]
pub enum EdgeCaseType {
    /// All zeros.
    AllZeros,
    /// All ones.
    AllOnes,
    /// Single non-zero element.
    SingleNonZero { row: usize, col: usize, value: f32 },
    /// Alternating pattern (+1, -1, +1, -1, ...).
    Alternating,
    /// Very large values that might cause overflow.
    LargeValues { max: f32 },
}

impl Default for TestMatrixConfig {
    fn default() -> Self {
        Self {
            shape: (32, 32),
            sparsity: 0.0,
            distribution: ValueDistribution::Normal { std: 1.0 },
            seed: 42,
        }
    }
}

/// Accuracy metrics for numerical validation.
#[derive(Debug, Clone)]
pub struct AccuracyMetrics {
    /// Mean Absolute Error.
    pub mae: f32,
    /// Root Mean Square Error.
    pub rmse: f32,
    /// Maximum absolute error.
    pub max_error: f32,
    /// Cosine similarity between tensors.
    pub cosine_similarity: f32,
}

/// Memory usage statistics.
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Original tensor size in bytes.
    pub original_bytes: usize,
    /// Quantized representation size in bytes.
    pub quantized_bytes: usize,
    /// Compression ratio (original/quantized).
    pub compression_ratio: f32,
    /// Actual sparsity detected.
    pub actual_sparsity: f32,
}

/// Test fixtures for common matrix patterns.
pub struct TestFixtures;

impl TestFixtures {
    /// Generate a test matrix based on configuration.
    pub fn generate_matrix(config: &TestMatrixConfig) -> Result<Tensor> {
        let device = Device::Cpu;
        let (rows, cols) = config.shape;

        // Generate base values according to distribution
        let values = match &config.distribution {
            ValueDistribution::Uniform { max } => {
                Self::generate_uniform_values(rows * cols, *max, config.seed)
            }
            ValueDistribution::Normal { std } => {
                Self::generate_normal_values(rows * cols, *std, config.seed)
            }
            ValueDistribution::LongTail { scale } => {
                Self::generate_longtail_values(rows * cols, *scale, config.seed)
            }
            ValueDistribution::EdgeCase(edge_case) => {
                Self::generate_edge_case_values(config.shape, edge_case.clone())
            }
        };

        // Apply sparsity if requested
        let sparse_values = if config.sparsity > 0.0 {
            Self::apply_sparsity(values, config.sparsity, config.seed)
        } else {
            values
        };

        // Create tensor
        let tensor = Tensor::from_vec(sparse_values, config.shape, &device)?;
        Ok(tensor)
    }

    /// Generate common test scenarios.
    pub fn standard_test_scenarios() -> Vec<(&'static str, TestMatrixConfig)> {
        vec![
            (
                "dense_small",
                TestMatrixConfig {
                    shape: (16, 16),
                    sparsity: 0.0,
                    distribution: ValueDistribution::Normal { std: 1.0 },
                    seed: 42,
                },
            ),
            (
                "sparse_50",
                TestMatrixConfig {
                    shape: (32, 32),
                    sparsity: 0.5,
                    distribution: ValueDistribution::Normal { std: 1.0 },
                    seed: 43,
                },
            ),
            (
                "highly_sparse",
                TestMatrixConfig {
                    shape: (64, 64),
                    sparsity: 0.90,
                    distribution: ValueDistribution::Normal { std: 1.0 },
                    seed: 44,
                },
            ),
            (
                "ultra_sparse",
                TestMatrixConfig {
                    shape: (128, 128),
                    sparsity: 0.99,
                    distribution: ValueDistribution::Normal { std: 0.5 },
                    seed: 45,
                },
            ),
            (
                "rectangular_wide",
                TestMatrixConfig {
                    shape: (32, 128),
                    sparsity: 0.7,
                    distribution: ValueDistribution::Uniform { max: 2.0 },
                    seed: 46,
                },
            ),
            (
                "rectangular_tall",
                TestMatrixConfig {
                    shape: (128, 32),
                    sparsity: 0.8,
                    distribution: ValueDistribution::LongTail { scale: 1.5 },
                    seed: 47,
                },
            ),
        ]
    }

    /// Generate edge case test scenarios.
    pub fn edge_case_scenarios() -> Vec<(&'static str, TestMatrixConfig)> {
        vec![
            (
                "all_zeros",
                TestMatrixConfig {
                    shape: (32, 32),
                    sparsity: 0.0,
                    distribution: ValueDistribution::EdgeCase(EdgeCaseType::AllZeros),
                    seed: 100,
                },
            ),
            (
                "all_ones",
                TestMatrixConfig {
                    shape: (16, 16),
                    sparsity: 0.0,
                    distribution: ValueDistribution::EdgeCase(EdgeCaseType::AllOnes),
                    seed: 101,
                },
            ),
            (
                "single_nonzero",
                TestMatrixConfig {
                    shape: (32, 32),
                    sparsity: 0.0,
                    distribution: ValueDistribution::EdgeCase(EdgeCaseType::SingleNonZero {
                        row: 15,
                        col: 20,
                        value: 5.0,
                    }),
                    seed: 102,
                },
            ),
            (
                "alternating",
                TestMatrixConfig {
                    shape: (16, 16),
                    sparsity: 0.0,
                    distribution: ValueDistribution::EdgeCase(EdgeCaseType::Alternating),
                    seed: 103,
                },
            ),
            (
                "large_values",
                TestMatrixConfig {
                    shape: (16, 16),
                    sparsity: 0.2,
                    distribution: ValueDistribution::EdgeCase(EdgeCaseType::LargeValues {
                        max: 1000.0,
                    }),
                    seed: 104,
                },
            ),
        ]
    }

    // Helper functions for value generation
    fn generate_uniform_values(count: usize, max: f32, seed: u64) -> Vec<f32> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut values = Vec::with_capacity(count);
        let hash_seed = seed;

        for i in 0..count {
            let mut hasher = DefaultHasher::new();
            (hash_seed + i as u64).hash(&mut hasher);
            let hash_value = hasher.finish();

            // Convert hash to uniform [-max, max]
            let normalized = (hash_value as f64) / (u64::MAX as f64); // [0,1)
            let value = (normalized * 2.0 - 1.0) * (max as f64); // [-max, max]
            values.push(value as f32);
        }

        values
    }

    fn generate_normal_values(count: usize, std: f32, seed: u64) -> Vec<f32> {
        // Simple Box-Muller transform for normal distribution
        let uniform = Self::generate_uniform_values(count * 2, 1.0, seed);
        let mut values = Vec::with_capacity(count);

        for i in (0..count * 2).step_by(2) {
            if i + 1 < uniform.len() {
                let u1 = uniform[i].abs().max(1e-8); // Avoid log(0)
                let u2 = uniform[i + 1];

                let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                values.push(z0 * std);

                if values.len() >= count {
                    break;
                }
            }
        }

        values.truncate(count);
        values
    }

    fn generate_longtail_values(count: usize, scale: f32, seed: u64) -> Vec<f32> {
        let uniform = Self::generate_uniform_values(count, 1.0, seed);
        uniform
            .into_iter()
            .map(|u| {
                // Power law distribution: x = scale * (1-u)^(-1/alpha) where alpha=1.5
                let alpha = 1.5;
                let value = scale * (1.0 - u.abs()).powf(-1.0 / alpha);
                if u < 0.0 {
                    -value
                } else {
                    value
                }
            })
            .collect()
    }

    fn generate_edge_case_values(shape: (usize, usize), edge_case: EdgeCaseType) -> Vec<f32> {
        let (rows, cols) = shape;
        let count = rows * cols;

        match edge_case {
            EdgeCaseType::AllZeros => vec![0.0; count],
            EdgeCaseType::AllOnes => vec![1.0; count],
            EdgeCaseType::SingleNonZero { row, col, value } => {
                let mut values = vec![0.0; count];
                if row < rows && col < cols {
                    values[row * cols + col] = value;
                }
                values
            }
            EdgeCaseType::Alternating => (0..count)
                .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
                .collect(),
            EdgeCaseType::LargeValues { max } => Self::generate_uniform_values(count, max, 999),
        }
    }

    fn apply_sparsity(mut values: Vec<f32>, sparsity: f32, seed: u64) -> Vec<f32> {
        let count = values.len();
        let num_zeros = (count as f32 * sparsity) as usize;

        // Generate random indices to zero out
        let uniform = Self::generate_uniform_values(count, 1.0, seed + 1000);
        let mut indices: Vec<_> = (0..count).collect();

        // Simple shuffle based on uniform values
        for i in 0..count {
            let j = ((uniform[i].abs() * (count - i) as f32) as usize + i).min(count - 1);
            indices.swap(i, j);
        }

        // Zero out the first num_zeros elements
        for &idx in indices.iter().take(num_zeros) {
            values[idx] = 0.0;
        }

        values
    }
}

/// Utility functions for accuracy and performance validation.
pub struct ValidationUtils;

impl ValidationUtils {
    /// Reconstruct a dense tensor from a ternary tensor for testing.
    pub fn reconstruct_dense_tensor(
        ternary: &unsloth_rs::kernels::ternary::types::TernaryTensor,
    ) -> Result<Tensor> {
        let device = Device::Cpu;
        let (rows, cols) = ternary.shape;
        let mut reconstructed_data = vec![0.0f32; rows * cols];

        // Reconstruct each element
        for row in 0..rows {
            for col in 0..cols {
                let ternary_val = ternary.get_dim(row, col);
                let scale = ternary.scales[row];
                reconstructed_data[row * cols + col] = (ternary_val as f32) * scale;
            }
        }

        Ok(Tensor::from_vec(reconstructed_data, (rows, cols), &device)?)
    }
    /// Calculate accuracy metrics between two tensors.
    pub fn calculate_accuracy_metrics(
        original: &Tensor,
        reconstructed: &Tensor,
    ) -> Result<AccuracyMetrics> {
        let orig_data = original.flatten_all()?.to_vec1::<f32>()?;
        let recon_data = reconstructed.flatten_all()?.to_vec1::<f32>()?;

        assert_eq!(
            orig_data.len(),
            recon_data.len(),
            "Tensor dimensions must match"
        );

        let n = orig_data.len() as f32;
        let mut sum_abs_error = 0.0;
        let mut sum_squared_error = 0.0;
        let mut max_error: f32 = 0.0;
        let mut dot_product = 0.0;
        let mut orig_norm_sq = 0.0;
        let mut recon_norm_sq = 0.0;

        for (orig, recon) in orig_data.iter().zip(recon_data.iter()) {
            let error = (orig - recon).abs();
            sum_abs_error += error;
            sum_squared_error += error * error;
            max_error = max_error.max(error);

            dot_product += orig * recon;
            orig_norm_sq += orig * orig;
            recon_norm_sq += recon * recon;
        }

        let mae = sum_abs_error / n;
        let rmse = (sum_squared_error / n).sqrt();

        let cosine_similarity = if orig_norm_sq > 0.0 && recon_norm_sq > 0.0 {
            dot_product / (orig_norm_sq.sqrt() * recon_norm_sq.sqrt())
        } else {
            0.0
        };

        Ok(AccuracyMetrics {
            mae,
            rmse,
            max_error,
            cosine_similarity,
        })
    }

    /// Calculate memory usage statistics.
    pub fn calculate_memory_stats(
        original_shape: (usize, usize),
        ternary_bytes: usize,
        actual_sparsity: f32,
    ) -> MemoryStats {
        let (rows, cols) = original_shape;
        let num_elements = rows * cols;

        // Assuming original is FP32 (4 bytes per element)
        let original_bytes = num_elements * 4;

        let compression_ratio = if ternary_bytes > 0 {
            original_bytes as f32 / ternary_bytes as f32
        } else {
            1.0
        };

        MemoryStats {
            original_bytes,
            quantized_bytes: ternary_bytes,
            compression_ratio,
            actual_sparsity,
        }
    }

    /// Check if accuracy metrics meet acceptable bounds.
    pub fn validate_accuracy_bounds(metrics: &AccuracyMetrics) -> HashMap<String, bool> {
        let mut results = HashMap::new();

        // Define acceptable bounds for ternary quantization (very realistic)
        results.insert("mae_acceptable".to_string(), metrics.mae < 1.0);
        results.insert("rmse_acceptable".to_string(), metrics.rmse < 1.5);
        results.insert("max_error_acceptable".to_string(), metrics.max_error < 15.0);
        results.insert(
            "cosine_sim_acceptable".to_string(),
            metrics.cosine_similarity > 0.20,
        );

        results
    }

    /// Check if memory compression meets expectations.
    pub fn validate_compression_ratio(stats: &MemoryStats, expected_min_ratio: f32) -> bool {
        stats.compression_ratio >= expected_min_ratio
    }
}

/// Timing utilities for performance validation.
pub struct TimingUtils;

impl TimingUtils {
    /// Time a function execution and return (result, duration_ms).
    pub fn time_execution<F, R>(f: F) -> (R, f64)
    where
        F: FnOnce() -> R,
    {
        let start = std::time::Instant::now();
        let result = f();
        let duration = start.elapsed();
        let duration_ms = duration.as_secs_f64() * 1000.0;
        (result, duration_ms)
    }

    /// Validate that execution time is within acceptable bounds.
    pub fn validate_performance(duration_ms: f64, max_expected_ms: f64) -> bool {
        duration_ms <= max_expected_ms
    }
}
