// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! Basic attention example demonstrating FusedAttention operations.
//!
//! This example shows how to:
//! - Create a FusedAttention layer with FusedAttentionConfig
//! - Perform a forward pass with Q, K, V tensors
//! - Print output shape and verify the result
//!
//! Run with: `cargo run --example basic_attention`

use anyhow::Result;
use candle_core::{Device, Tensor};
use unsloth_rs::kernels::{FusedAttention, FusedAttentionConfig};

fn main() -> Result<()> {
    println!("=== Basic Attention Example ===\n");

    // Configuration for a small attention layer
    let config = FusedAttentionConfig {
        hidden_size: 256,
        num_heads: 4,
        head_dim: 64,
        num_kv_heads: None, // Use None for standard multi-head attention (MHA)
        dropout: 0.0,
        use_flash: false, // Disable flash attention for CPU demo
    };

    println!("Configuration:");
    println!("  Hidden size: {}", config.hidden_size);
    println!("  Number of heads: {}", config.num_heads);
    println!("  Head dimension: {}", config.head_dim);
    println!();

    // Create device (CPU for this example)
    let device = Device::Cpu;

    // Create attention layer
    println!("Creating FusedAttention layer...");
    let attention = FusedAttention::new(config.clone(), &device)?;
    println!("Attention layer created successfully.\n");

    // Create sample input tensor [batch_size, seq_len, hidden_size]
    let batch_size = 2;
    let seq_len = 8;
    let hidden_size = config.hidden_size;

    println!("Creating input tensor:");
    println!("  Shape: [{}, {}, {}]", batch_size, seq_len, hidden_size);
    let input = Tensor::randn(0.0f32, 1.0, (batch_size, seq_len, hidden_size), &device)?;
    println!("  Input tensor created.\n");

    // Forward pass
    println!("Running forward pass...");
    let output = attention.forward(&input, None, None)?;
    println!("Forward pass completed.\n");

    // Print output information
    let output_shape = output.shape();
    println!("Output:");
    println!("  Shape: {:?}", output_shape.dims());
    println!(
        "  Expected shape: [{}, {}, {}]",
        batch_size, seq_len, hidden_size
    );

    // Verify output shape
    assert_eq!(
        output_shape.dims(),
        &[batch_size, seq_len, hidden_size],
        "Output shape mismatch"
    );
    println!("  Shape verification: PASSED\n");

    // Compute and display statistics
    let output_mean = output.mean_all()?.to_scalar::<f32>()?;
    let output_sum = output.sum_all()?.to_scalar::<f32>()?;

    println!("Output statistics:");
    println!("  Mean: {:.6}", output_mean);
    println!("  Sum: {:.6}", output_sum);

    // Verify no NaN values
    assert!(!output_mean.is_nan(), "Output contains NaN values");
    assert!(!output_mean.is_infinite(), "Output contains Inf values");
    println!("  Numerical stability: PASSED\n");

    // Estimate VRAM usage
    let vram_bytes = attention.vram_estimate(batch_size, seq_len);
    let vram_mb = vram_bytes as f32 / (1024.0 * 1024.0);
    println!("Memory estimate:");
    println!("  VRAM usage: {:.2} MB ({} bytes)", vram_mb, vram_bytes);

    println!("\n=== Example completed successfully! ===");

    Ok(())
}
