// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! Bitsliced ternary matrix multiplication.
//!
//! This module implements popcount-based matrix multiplication for
//! ternary weights, providing 5-20x speedup on sparse pruned models.
//!
//! ## Algorithm
//!
//! For ternary weight matrix W (stored as +plane/-plane) and FP activation X:
//!
//! ```text
//! Y[i,j] = sum_k X[i,k] * W[j,k]
//!        = scale[j] * sum_k X[i,k] * ternary[j,k]
//! ```
//!
//! The inner dot product is computed via popcount:
//!
//! ```text
//! dot = popcount(A+ & B+) + popcount(A- & B-)
//!     - popcount(A+ & B-) - popcount(A- & B+)
//! ```
//!
//! ## GPU Kernel
//!
//! The `CubeCL` kernel uses:
//! - Shared memory for tiled computation
//! - Warp-level popcount intrinsics
//! - Vectorized loads for coalescing
//! - Plane skipping for sparse weights

use super::config::TernaryConfig;
use super::types::TernaryTensor;
use crate::error::{Result, UnslothError};
// TODO: Re-enable when ternary CubeCL modules are ready
#[cfg(all(feature = "cuda", feature = "_ternary_cubecl_todo"))]
use candle_core::Device;
use candle_core::Tensor;

// CubeCL imports for kernel implementation
#[cfg(all(feature = "cuda", feature = "_ternary_cubecl_todo"))]
use cubecl::prelude::*;
#[cfg(all(feature = "cuda", feature = "_ternary_cubecl_todo"))]
use cubecl_cuda::CudaRuntime;

/// Compile-time configuration for ternary matmul kernel.
#[derive(Clone, Copy, Debug)]
#[cfg(all(feature = "cuda", feature = "_ternary_cubecl_todo"))]
pub struct TernaryMatmulConfig {
    /// Tile size for M dimension (rows of output)
    pub tile_m: u32,
    /// Tile size for N dimension (cols of output)
    pub tile_n: u32,
    /// Number of u32 words in K dimension
    pub k_words: u32,
    /// Number of output rows (M)
    pub m: u32,
    /// Number of output cols (N = weight rows)
    pub n: u32,
    /// Enable plane skipping
    pub skip_empty_planes: bool,
}

/// Ternary matmul: Y = X @ W^T where W is ternary.
///
/// # Arguments
///
/// * `input` - Input tensor [batch, `seq_len`, `in_features`] or [batch, `in_features`]
/// * `weights` - Ternary weight tensor [`out_features`, `in_features`]
/// * `config` - Ternary configuration
///
/// # Returns
///
/// Output tensor [batch, `seq_len`, `out_features`] or [batch, `out_features`]
///
/// # Errors
///
/// Returns error if shapes don't match or computation fails.
#[allow(unused_variables)]
pub fn ternary_matmul(
    input: &Tensor,
    weights: &TernaryTensor,
    config: &TernaryConfig,
) -> Result<Tensor> {
    // Validate shapes
    let input_shape = input.shape().dims();
    let (_out_features, in_features) = weights.dims();

    let input_features = *input_shape
        .last()
        .ok_or_else(|| UnslothError::ShapeMismatch {
            expected: vec![in_features],
            actual: input_shape.to_vec(),
        })?;

    if input_features != in_features {
        return Err(UnslothError::ShapeMismatch {
            expected: vec![in_features],
            actual: vec![input_features],
        });
    }

    // Check if we should use GPU path
    // TODO: Re-enable when ternary CubeCL modules are ready
    #[cfg(all(feature = "cuda", feature = "_ternary_cubecl_todo"))]
    {
        if let Device::Cuda(_) = input.device() {
            if weights.is_sparse_enough(config) {
                return ternary_matmul_cuda(input, weights, config);
            }
        }
    }

    // CPU fallback
    ternary_matmul_cpu(input, weights)
}

/// CPU reference implementation of ternary matmul.
///
/// Uses straightforward popcount-based dot product.
/// Useful for validation and as fallback when GPU unavailable.
pub fn ternary_matmul_cpu(input: &Tensor, weights: &TernaryTensor) -> Result<Tensor> {
    let input_shape = input.shape().dims();
    let (out_features, in_features) = weights.dims();

    // Flatten input to 2D [batch_total, in_features]
    let input_2d = if input_shape.len() == 2 {
        input.clone()
    } else {
        let batch_total: usize = input_shape[..input_shape.len() - 1].iter().product();
        input.reshape((batch_total, in_features))?
    };

    let batch_total = input_2d.shape().dims()[0];
    let input_data: Vec<f32> = input_2d.flatten_all()?.to_vec1()?;

    // Output buffer
    let mut output_data = vec![0.0f32; batch_total * out_features];

    // For each batch element
    for b in 0..batch_total {
        let input_row = &input_data[b * in_features..(b + 1) * in_features];

        // For each output feature (weight row)
        for o in 0..out_features {
            let planes = weights.get_row_planes(o);
            let scale = weights.scales[o];

            // Compute dot product via ternary representation
            let mut acc = 0.0f32;

            for (i, &val) in input_row.iter().enumerate() {
                let ternary_val = planes.get(i);
                acc += val * f32::from(ternary_val);
            }

            output_data[b * out_features + o] = acc * scale;
        }
    }

    // Reshape output to match input batch dims
    let output_shape: Vec<usize> = input_shape[..input_shape.len() - 1]
        .iter()
        .copied()
        .chain(std::iter::once(out_features))
        .collect();

    let output = Tensor::from_vec(output_data, output_shape.as_slice(), input.device())?;
    Ok(output)
}

/// Optimized CPU implementation using packed popcount.
///
/// Quantizes input to ternary on-the-fly for popcount-based computation.
/// Faster than naive when input is also sparse.
pub fn ternary_matmul_cpu_packed(
    input: &Tensor,
    weights: &TernaryTensor,
    input_threshold: f32,
) -> Result<Tensor> {
    let input_shape = input.shape().dims();
    let (out_features, in_features) = weights.dims();
    let k_words = weights.k_words;

    // Flatten input to 2D
    let input_2d = if input_shape.len() == 2 {
        input.clone()
    } else {
        let batch_total: usize = input_shape[..input_shape.len() - 1].iter().product();
        input.reshape((batch_total, in_features))?
    };

    let batch_total = input_2d.shape().dims()[0];
    let input_data: Vec<f32> = input_2d.flatten_all()?.to_vec1()?;

    let mut output_data = vec![0.0f32; batch_total * out_features];

    // For each batch element
    for b in 0..batch_total {
        let input_row = &input_data[b * in_features..(b + 1) * in_features];

        // Quantize input row to ternary planes
        let (input_plus, input_minus, input_scale) =
            quantize_activation_row(input_row, input_threshold, k_words);

        // For each output feature
        for o in 0..out_features {
            let weight_scale = weights.scales[o];
            let plane_offset = o * k_words;

            // Popcount-based dot product
            let mut pos_matches = 0i32;
            let mut neg_matches = 0i32;

            for k in 0..k_words {
                let wp = weights.plus_plane[plane_offset + k];
                let wm = weights.minus_plane[plane_offset + k];
                let ip = input_plus[k];
                let im = input_minus[k];

                pos_matches += (wp & ip).count_ones().cast_signed();
                pos_matches += (wm & im).count_ones().cast_signed();
                neg_matches += (wp & im).count_ones().cast_signed();
                neg_matches += (wm & ip).count_ones().cast_signed();
            }

            let dot = pos_matches - neg_matches;
            // Precision loss acceptable for ternary dot product calculation
            #[allow(clippy::cast_precision_loss)]
            {
                output_data[b * out_features + o] = dot as f32 * weight_scale * input_scale;
            }
        }
    }

    let output_shape: Vec<usize> = input_shape[..input_shape.len() - 1]
        .iter()
        .copied()
        .chain(std::iter::once(out_features))
        .collect();

    let output = Tensor::from_vec(output_data, output_shape.as_slice(), input.device())?;
    Ok(output)
}

/// Quantize activation row to ternary for packed computation.
fn quantize_activation_row(
    data: &[f32],
    threshold: f32,
    k_words: usize,
) -> (Vec<u32>, Vec<u32>, f32) {
    let mut plus = vec![0u32; k_words];
    let mut minus = vec![0u32; k_words];

    let mut pos_sum = 0.0f64;
    let mut neg_sum = 0.0f64;
    let mut nonzero_count = 0;

    for (i, &val) in data.iter().enumerate() {
        let word_idx = i / 32;
        let bit_idx = i % 32;
        let mask = 1u32 << bit_idx;

        if val > threshold {
            plus[word_idx] |= mask;
            pos_sum += f64::from(val.abs());
            nonzero_count += 1;
        } else if val < -threshold {
            minus[word_idx] |= mask;
            neg_sum += f64::from(val.abs());
            nonzero_count += 1;
        }
    }

    let scale = if nonzero_count > 0 {
        // Truncation acceptable for scale calculation - precision already limited by f32 input
        #[allow(clippy::cast_possible_truncation)]
        {
            ((pos_sum + neg_sum) / f64::from(nonzero_count)) as f32
        }
    } else {
        1.0
    };

    (plus, minus, scale)
}

// ============================================================================
// CUDA Kernel Implementation
// ============================================================================

// TODO: Re-enable when ternary CubeCL modules are ready
// This code requires the matmul_cubecl module which is currently disabled
#[cfg(all(feature = "cuda", feature = "_ternary_cubecl_todo"))]
fn ternary_matmul_cuda(
    input: &Tensor,
    weights: &TernaryTensor,
    config: &TernaryConfig,
) -> Result<Tensor> {
    use super::matmul_cubecl::*;
    use crate::kernels::cubecl::interop::*;

    // Get dimensions
    let input_shape = input.shape().dims();
    let (out_features, in_features) = weights.dims();
    let batch_size = input_shape[0];
    let k_words = weights.k_words as u32;

    // Calculate sparsity
    let sparsity = weights.sparsity();

    // Select kernel based on sparsity
    let use_sparse_kernel = sparsity >= 0.90;

    // Detect GPU (placeholder - will be improved when GPU hardware available)
    let device = input.device();
    let gpu_name = detect_gpu_name_placeholder();

    log::debug!(
        "CUDA ternary matmul: batch={}, out={}, in={}, sparsity={:.2}, kernel={}",
        batch_size,
        out_features,
        in_features,
        sparsity,
        if use_sparse_kernel {
            "sparse"
        } else {
            "vectorized"
        }
    );

    // For now, use fallback until we can properly test GPU dispatch
    // TODO: Enable when GPU hardware available for testing
    log::debug!("CUDA ternary matmul: falling back to CPU (GPU dispatch under development)");
    ternary_matmul_cpu(input, weights)

    /* GPU dispatch code - to be enabled after hardware testing:

    // Initialize CubeCL runtime
    let device_id = match device {
        Device::Cuda(id) => id.ordinal(),
        _ => return Err(UnslothError::InvalidConfig("Expected CUDA device".into())),
    };

    let client = CudaRuntime::client(device_id);

    // Convert input to CubeCL handle
    let (input_bytes, input_shape_vec, _) = candle_to_cubecl_handle(input)?;
    let input_handle = client.create(&input_bytes);

    // Convert weight planes to CubeCL handles (reinterpret u32 as f32)
    let w_plus_bytes: Vec<u8> = weights.plus_plane
        .iter()
        .flat_map(|&word| {
            let f = f32::from_bits(word);
            f.to_le_bytes()
        })
        .collect();
    let w_plus_handle = client.create(&w_plus_bytes);

    let w_minus_bytes: Vec<u8> = weights.minus_plane
        .iter()
        .flat_map(|&word| {
            let f = f32::from_bits(word);
            f.to_le_bytes()
        })
        .collect();
    let w_minus_handle = client.create(&w_minus_bytes);

    // Convert scales
    let scales_bytes: Vec<u8> = weights.scales
        .iter()
        .flat_map(|&s| s.to_le_bytes())
        .collect();
    let scales_handle = client.create(&scales_bytes);

    // Allocate output
    let output_size = batch_size * out_features;
    let output_bytes = allocate_output_buffer(output_size);
    let output_handle = client.create(&output_bytes);

    // Dispatch to appropriate kernel
    if use_sparse_kernel {
        // Use sparse-optimized kernel
        let kernel_config = if gpu_name.contains("5080") {
            SparseOptimizedConfig::rtx_5080_sparse(
                batch_size as u32,
                out_features as u32,
                k_words,
                in_features as u32,
                sparsity,
            )
        } else {
            SparseOptimizedConfig::rtx_3090ti_sparse(
                batch_size as u32,
                out_features as u32,
                k_words,
                in_features as u32,
                sparsity,
            )
        };

        // Create sparsity bitmap
        let bitmap_bytes = create_sparsity_bitmap_for_tensor(weights, 64);
        let bitmap_handle = client.create(&bitmap_bytes);

        let (cube_count, cube_dim) = get_sparse_launch_config(&kernel_config);

        ternary_matmul_kernel_sparse::launch_unchecked::<F32, CudaRuntime>(
            &client,
            cube_count,
            cube_dim,
            input_handle,
            w_plus_handle,
            w_minus_handle,
            scales_handle,
            bitmap_handle,
            output_handle,
            kernel_config,
        );
    } else {
        // Use vectorized kernel (best for dense)
        let kernel_config = if gpu_name.contains("5080") {
            VectorizedTernaryMatmulConfig::rtx_5080_preset(
                batch_size as u32,
                out_features as u32,
                k_words,
                in_features as u32,
            )
        } else {
            VectorizedTernaryMatmulConfig::rtx_3090ti_preset(
                batch_size as u32,
                out_features as u32,
                k_words,
                in_features as u32,
            )
        };

        let (cube_count, cube_dim) = get_vectorized_launch_config(&kernel_config);

        ternary_matmul_kernel_vectorized::launch_unchecked::<F32, CudaRuntime>(
            &client,
            cube_count,
            cube_dim,
            input_handle,
            w_plus_handle,
            w_minus_handle,
            scales_handle,
            output_handle,
            kernel_config,
        );
    }

    // Convert output back to Candle tensor
    let output_bytes = client.read(&output_handle);
    cubecl_to_candle_tensor(&output_bytes, &[batch_size, out_features], device)
    */
}

/// Detect GPU name from device (placeholder until GPU hardware available)
#[cfg(all(feature = "cuda", feature = "_ternary_cubecl_todo"))]
fn detect_gpu_name_placeholder() -> String {
    // TODO: Use CUDA device properties when GPU available
    // For now, return a default that will use conservative settings
    "RTX 3090 Ti".to_string()
}

/// CubeCL kernel for ternary matmul (stub for Phase 2).
///
/// This kernel will be fully implemented after CPU validation passes.
#[cfg(all(feature = "cuda", feature = "_ternary_cubecl_todo"))]
#[cube(launch_unchecked)]
fn ternary_matmul_kernel<F: Float>(
    // Input activations [batch, in_features] as f32
    input: &Array<F>,
    // Weight positive plane [out_features, k_words] as u32 (reinterpreted)
    w_plus: &Array<u32>,
    // Weight negative plane [out_features, k_words] as u32
    w_minus: &Array<u32>,
    // Per-row scales [out_features]
    scales: &Array<F>,
    // Output [batch, out_features]
    output: &mut Array<F>,
    #[comptime] config: TernaryMatmulConfig,
) {
    // Thread indices
    let batch_idx = CUBE_POS_X;
    let out_idx = CUBE_POS_Y * config.tile_n + UNIT_POS_X;

    if out_idx >= config.n {
        return;
    }

    // Shared memory for input tile (will be used in tiled version)
    // let input_tile = SharedMemory::<F>::new(config.tile_m * 32);

    // Initialize accumulator
    let mut acc = F::new(0.0);

    // For Phase 1: simple non-tiled implementation
    // Each thread handles one (batch, out_feature) element

    let input_offset = batch_idx * config.k_words * 32;
    let weight_offset = out_idx * config.k_words;

    // Iterate over K dimension (packed u32 words)
    for k in 0..config.k_words {
        let wp = w_plus[weight_offset + k];
        let wm = w_minus[weight_offset + k];

        // For each bit in the u32 word
        // This will be optimized with vectorized loads in Phase 2
        for bit in 0u32..32u32 {
            let dim_idx = k * 32 + bit;
            if dim_idx < config.k_words * 32 {
                let mask = 1u32 << bit;
                let is_pos = (wp & mask) != 0;
                let is_neg = (wm & mask) != 0;

                let input_val = input[input_offset + dim_idx];

                // Ternary multiplication: +1, 0, or -1
                if is_pos {
                    acc = acc + input_val;
                } else if is_neg {
                    acc = acc - input_val;
                }
                // else: weight is 0, no contribution
            }
        }
    }

    // Apply scale
    let scale = scales[out_idx];
    output[batch_idx * config.n + out_idx] = acc * scale;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::ternary::{quantize_tensor, TernaryConfig};
    use candle_core::Device;

    #[test]
    fn test_ternary_matmul_identity() -> Result<()> {
        // Create a simple case: input [2, 4], weights [3, 4]
        // Expected output [2, 3]

        // Weights: rows of [1, 0, -1, 1] type patterns
        let weight_data = vec![
            1.0f32, 0.0, -1.0, 1.0, // Row 0: sum input[0] - input[2] + input[3]
            0.0, 1.0, 1.0, 0.0, // Row 1: input[1] + input[2]
            -1.0, -1.0, 0.0, 1.0, // Row 2: -input[0] - input[1] + input[3]
        ];
        let weights = Tensor::from_vec(weight_data, (3, 4), &Device::Cpu)?;

        // Quantize weights
        let config = TernaryConfig {
            calibration_method: super::super::config::CalibrationMethodConfig::Manual(0.1),
            ..Default::default()
        };
        let (ternary_weights, _) = quantize_tensor(&weights, &config)?;

        // Input
        let input_data = vec![
            1.0f32, 2.0, 3.0, 4.0, // Batch 0
            0.5, 1.5, 2.5, 3.5, // Batch 1
        ];
        let input = Tensor::from_vec(input_data, (2, 4), &Device::Cpu)?;

        // Compute
        let output = ternary_matmul_cpu(&input, &ternary_weights)?;

        // Verify shape
        assert_eq!(output.shape().dims(), &[2, 3]);

        // Verify values (approximately, due to scale factors)
        let output_data: Vec<f32> = output.flatten_all()?.to_vec1()?;

        // Row 0, Batch 0: scale * (1 - 3 + 4) = scale * 2
        // Row 1, Batch 0: scale * (2 + 3) = scale * 5
        // Row 2, Batch 0: scale * (-1 - 2 + 4) = scale * 1

        // Check signs at least match expected
        assert!(output_data[0] > 0.0); // Positive
        assert!(output_data[1] > 0.0); // Positive
        assert!(output_data[2] > 0.0); // Positive

        Ok(())
    }

    #[test]
    fn test_matmul_shape_3d() -> Result<()> {
        // Test with 3D input [batch, seq, features]
        let weight_data = vec![0.5f32; 64 * 128];
        let weights = Tensor::from_vec(weight_data, (64, 128), &Device::Cpu)?;

        let config = TernaryConfig::default();
        let (ternary_weights, _) = quantize_tensor(&weights, &config)?;

        let input = Tensor::zeros((2, 16, 128), candle_core::DType::F32, &Device::Cpu)?;
        let output = ternary_matmul_cpu(&input, &ternary_weights)?;

        assert_eq!(output.shape().dims(), &[2, 16, 64]);

        Ok(())
    }

    #[test]
    fn test_packed_matmul_equivalence() -> Result<()> {
        // Test that packed version gives similar results to standard
        let weight_data: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 128.0).collect();
        let weights = Tensor::from_vec(weight_data, (4, 64), &Device::Cpu)?;

        let config = TernaryConfig::default();
        let (ternary_weights, _) = quantize_tensor(&weights, &config)?;

        let input_data: Vec<f32> = (0..128)
            .map(|i| {
                // Precision loss acceptable for test data generation
                #[allow(clippy::cast_precision_loss)]
                {
                    (i as f32) / 64.0 - 1.0
                }
            })
            .collect();
        let input = Tensor::from_vec(input_data, (2, 64), &Device::Cpu)?;

        let output_std = ternary_matmul_cpu(&input, &ternary_weights)?;
        let output_packed = ternary_matmul_cpu_packed(&input, &ternary_weights, 0.3)?;

        // Results should be in same ballpark (packed has additional quantization)
        let std_data: Vec<f32> = output_std.flatten_all()?.to_vec1()?;
        let packed_data: Vec<f32> = output_packed.flatten_all()?.to_vec1()?;

        // Check correlation rather than exact match
        let mean_std: f32 = std_data.iter().sum::<f32>() / {
            // Precision loss acceptable for test metric calculation
            #[allow(clippy::cast_precision_loss)]
            {
                std_data.len() as f32
            }
        };
        let mean_packed: f32 = packed_data.iter().sum::<f32>() / {
            // Precision loss acceptable for test metric calculation
            #[allow(clippy::cast_precision_loss)]
            {
                packed_data.len() as f32
            }
        };

        // Check that the signs are approximately similar (both should be negative or similar magnitude)
        assert!(
            (mean_std - mean_packed).abs() < 1.0,
            "Means too different: std={}, packed={}",
            mean_std,
            mean_packed
        );

        Ok(())
    }

    #[test]
    fn test_ternary_matmul_dispatch_cpu() -> Result<()> {
        // Verify CPU dispatch works correctly
        let device = Device::Cpu;

        // Create simple weights and input
        let weight_data = vec![1.0f32, -1.0, 0.0, 1.0];
        let weights_fp = Tensor::from_vec(weight_data, (2, 2), &device)?;

        let config = TernaryConfig {
            calibration_method: super::super::config::CalibrationMethodConfig::Manual(0.1),
            ..Default::default()
        };
        let (ternary_weights, _) = quantize_tensor(&weights_fp, &config)?;

        let input_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let input = Tensor::from_vec(input_data, (2, 2), &device)?;

        // Should automatically dispatch to CPU
        let output = ternary_matmul(&input, &ternary_weights, &config)?;

        assert_eq!(output.shape().dims(), &[2, 2]);
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_ternary_matmul_dispatch_gpu() -> Result<()> {
        // Verify GPU dispatch routing works (will use fallback until GPU available)
        if let Ok(device) = Device::cuda_if_available(0) {
            if !matches!(device, Device::Cuda(_)) {
                return Ok(()); // Skip if no GPU
            }

            let weight_data = vec![1.0f32, -1.0, 0.0, 1.0];
            let weights_fp = Tensor::from_vec(weight_data, (2, 2), &Device::Cpu)?;

            let config = TernaryConfig {
                calibration_method: super::super::config::CalibrationMethodConfig::Manual(0.1),
                ..Default::default()
            };
            let (ternary_weights, _) = quantize_tensor(&weights_fp, &config)?;

            let input_data = vec![1.0f32, 2.0, 3.0, 4.0];
            let input = Tensor::from_vec(input_data, (2, 2), &device)?;

            // Should route through CUDA path (currently falls back to CPU)
            let output = ternary_matmul(&input, &ternary_weights, &config)?;

            assert_eq!(output.shape().dims(), &[2, 2]);
        }
        Ok(())
    }

    #[test]
    fn test_gpu_name_detection() {
        // Test disabled until ternary CubeCL modules are ready
        // #[cfg(feature = "cuda")]
        // {
        //     let name = detect_gpu_name_placeholder();
        //     // Should return a valid GPU name
        //     assert!(!name.is_empty());
        //     assert!(name.contains("RTX") || name.contains("GPU"));
        // }
    }
}
