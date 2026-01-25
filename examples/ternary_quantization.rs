// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! Ternary quantization example demonstrating weight compression.
//!
//! This example shows how to:
//! - Create a random weight tensor
//! - Quantize it to ternary representation {-1, 0, +1}
//! - Display compression statistics
//! - Verify quantization quality
//!
//! Run with: `cargo run --example ternary_quantization`

use anyhow::Result;
use candle_core::{Device, Tensor};
use unsloth_rs::kernels::ternary::{quantize_tensor, TernaryConfig};

fn main() -> Result<()> {
    println!("=== Ternary Quantization Example ===\n");

    // Create a random weight tensor
    let out_features = 512;
    let in_features = 2048;
    let device = Device::Cpu;

    println!("Creating random weight tensor:");
    println!("  Shape: [{}, {}]", out_features, in_features);
    println!("  Data type: f32");

    // Create random weights with normal distribution
    let weights = Tensor::randn(0.0f32, 0.5, (out_features, in_features), &device)?;
    println!("  Weights created.\n");

    // Calculate original size
    let original_size = out_features * in_features * 4; // f32 = 4 bytes
    let original_mb = original_size as f32 / (1024.0 * 1024.0);
    println!("Original weights:");
    println!("  Size: {:.2} MB ({} bytes)", original_mb, original_size);
    println!();

    // Create ternary configuration
    let config = TernaryConfig::default();
    println!("Quantization configuration:");
    println!("  Sparsity threshold: {}", config.sparsity_threshold);
    println!("  Calibration method: {:?}", config.calibration_method);
    println!();

    // Quantize the tensor
    println!("Quantizing weights to ternary representation...");
    let (ternary_tensor, stats) = quantize_tensor(&weights, &config)?;
    println!("Quantization completed.\n");

    // Display quantization statistics
    println!("=== Quantization Statistics ===");
    println!("Distribution:");
    println!("  Sparsity (zeros): {:.2}%", stats.sparsity * 100.0);
    println!(
        "  Positive values (+1): {:.2}%",
        stats.positive_ratio * 100.0
    );
    println!(
        "  Negative values (-1): {:.2}%",
        stats.negative_ratio * 100.0
    );
    println!();

    println!("Quantization error:");
    println!("  Mean absolute error: {:.6}", stats.mean_error);
    println!("  Max absolute error: {:.6}", stats.max_error);
    println!();

    // Display scale statistics
    let avg_scale = stats.scales.iter().sum::<f32>() / stats.scales.len() as f32;
    let min_scale = stats.scales.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max_scale = stats
        .scales
        .iter()
        .fold(f32::NEG_INFINITY, |a, &b| a.max(b));

    println!("Scale statistics:");
    println!("  Average scale: {:.6}", avg_scale);
    println!("  Min scale: {:.6}", min_scale);
    println!("  Max scale: {:.6}", max_scale);
    println!();

    // Display compression information
    println!("=== Compression Results ===");
    let compression_ratio = ternary_tensor.compression_ratio();
    println!("  Compression ratio: {:.2}x", compression_ratio);

    let (ternary_out, ternary_in) = ternary_tensor.dims();
    println!(
        "  Ternary tensor dimensions: [{}, {}]",
        ternary_out, ternary_in
    );

    let memory_saved_bytes = original_size as f32 - (original_size as f32 / compression_ratio);
    let memory_saved_mb = memory_saved_bytes / (1024.0 * 1024.0);
    println!("  Memory saved: {:.2} MB", memory_saved_mb);

    let ternary_size_mb = original_mb / compression_ratio;
    println!("  Ternary tensor size: {:.2} MB", ternary_size_mb);
    println!();

    // Verify sparsity
    let tensor_sparsity = ternary_tensor.sparsity();
    println!("=== Verification ===");
    println!("  Tensor sparsity: {:.2}%", tensor_sparsity * 100.0);
    assert!(
        (tensor_sparsity - stats.sparsity).abs() < 0.001,
        "Sparsity mismatch"
    );
    println!("  Sparsity consistency: PASSED");

    // Verify dimensions
    assert_eq!((ternary_out, ternary_in), (out_features, in_features));
    println!("  Dimension preservation: PASSED");

    println!("\n=== Example completed successfully! ===");
    println!("\nKey takeaways:");
    println!(
        "  - Ternary quantization reduces memory by {:.1}x",
        compression_ratio
    );
    println!(
        "  - {:.1}% of weights are quantized to zero",
        stats.sparsity * 100.0
    );
    println!("  - Mean quantization error: {:.6}", stats.mean_error);

    Ok(())
}
