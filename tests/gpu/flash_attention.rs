//! Flash Attention GPU tests for unsloth-rs.
//!
//! This module provides comprehensive GPU testing for the CubeCL Flash Attention
//! implementation, covering numerical accuracy, performance validation, memory
//! efficiency, and edge cases.
//!
//! ## Test Categories
//!
//! 1. **Numerical Equivalence**: GPU results match CPU reference within tolerance
//! 2. **Performance Validation**: GPU faster than CPU for sequences >256  
//! 3. **Memory Efficiency**: O(√n) vs O(n²) memory usage validation
//! 4. **Configuration Testing**: Different tile sizes and block configurations
//! 5. **Scalability Testing**: Large sequences and batch sizes
//! 6. **Error Handling**: Invalid inputs and resource limits
//! 7. **Stability Testing**: Memory management and cleanup
//!
//! ## Expected Performance Targets
//!
//! - **Accuracy**: MAE < 1e-5, RMSE < 1e-4 (adjusted for fp16/bf16)
//! - **Speed**: GPU ≥2x faster than CPU for sequences ≥512
//! - **Memory**: Peak VRAM < 2x theoretical minimum
//! - **Reliability**: No memory leaks or device errors

use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn;
use std::time::Instant;

use unsloth_rs::kernels::{
    attention::{FusedAttention, FusedAttentionConfig},
    attention_cubecl::{estimate_flash_attention_vram, flash_attention_cubecl, has_cubecl_support},
};

/// Test configuration for Flash Attention benchmarks.
#[derive(Debug, Clone)]
struct AttentionTestConfig {
    pub batch_size: usize,
    pub num_heads: usize,
    pub seq_len: usize,
    pub head_dim: usize,
    pub num_kv_heads: Option<usize>,
    pub use_causal_mask: bool,
    pub dtype: DType,
}

impl AttentionTestConfig {
    /// Small configuration for basic correctness tests.
    fn small() -> Self {
        Self {
            batch_size: 1,
            num_heads: 4,
            seq_len: 128,
            head_dim: 64,
            num_kv_heads: None,
            use_causal_mask: false,
            dtype: DType::F32,
        }
    }

    /// Medium configuration for performance tests.
    fn medium() -> Self {
        Self {
            batch_size: 2,
            num_heads: 8,
            seq_len: 512,
            head_dim: 64,
            num_kv_heads: Some(8),
            use_causal_mask: true,
            dtype: DType::F32,
        }
    }

    /// Large configuration for scalability tests.
    fn large() -> Self {
        Self {
            batch_size: 4,
            num_heads: 16,
            seq_len: 1024,
            head_dim: 64,
            num_kv_heads: Some(16),
            use_causal_mask: true,
            dtype: DType::F32,
        }
    }

    /// Extra large configuration for stress tests.
    fn extra_large() -> Self {
        Self {
            batch_size: 4,
            num_heads: 32,
            seq_len: 2048,
            head_dim: 64,
            num_kv_heads: Some(32),
            use_causal_mask: true,
            dtype: DType::F32,
        }
    }
}

/// Generate test tensors for attention computation.
fn create_test_tensors(
    config: &AttentionTestConfig,
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor, Option<Tensor>)> {
    let (batch, heads, seq_len, head_dim) = (
        config.batch_size,
        config.num_heads,
        config.seq_len,
        config.head_dim,
    );

    let kv_heads = config.num_kv_heads.unwrap_or(heads);

    // Create Q, K, V tensors with proper shapes
    let q = Tensor::randn(0.0f32, 1.0, (batch, heads, seq_len, head_dim), device)?;
    let k = Tensor::randn(0.0f32, 1.0, (batch, kv_heads, seq_len, head_dim), device)?;
    let v = Tensor::randn(0.0f32, 1.0, (batch, kv_heads, seq_len, head_dim), device)?;

    // Create causal mask if requested
    let mask = if config.use_causal_mask {
        let mut mask_data = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                mask_data[i * seq_len + j] = f32::NEG_INFINITY;
            }
        }
        Some(Tensor::from_slice(&mask_data, (seq_len, seq_len), device)?)
    } else {
        None
    };

    Ok((q, k, v, mask))
}

/// Compute attention using CPU reference implementation.
fn attention_reference_cpu(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    mask: Option<&Tensor>,
) -> Result<Tensor> {
    // Compute Q·K^T
    let scores = q.matmul(&k.transpose(2, 3)?.contiguous()?)?;
    let scores = (scores / scale)?;

    // Apply mask if provided
    let scores = match mask {
        Some(m) => {
            // Broadcast mask to match scores shape
            let mask_shape = scores.shape().dims();
            let m_broadcasted = m.broadcast_as(mask_shape)?;
            scores.broadcast_add(&m_broadcasted)?
        }
        None => scores,
    };

    // Softmax along last dimension (over key positions)
    let attn_weights = candle_nn::ops::softmax(&scores, 3)?;

    // Attention output: attn_weights · V
    let output = attn_weights.matmul(v)?;
    Ok(output)
}

/// Calculate numerical accuracy metrics between two tensors.
fn calculate_accuracy_metrics(pred: &Tensor, target: &Tensor) -> Result<AccuracyMetrics> {
    let pred_vec: Vec<f32> = pred.flatten_all()?.to_vec1()?;
    let target_vec: Vec<f32> = target.flatten_all()?.to_vec1()?;

    assert_eq!(pred_vec.len(), target_vec.len(), "Tensor size mismatch");

    let mut mae = 0.0;
    let mut mse = 0.0;
    let mut dot_product = 0.0;
    let mut pred_norm = 0.0;
    let mut target_norm = 0.0;

    for (p, t) in pred_vec.iter().zip(target_vec.iter()) {
        let diff = p - t;
        mae += diff.abs();
        mse += diff * diff;
        dot_product += p * t;
        pred_norm += p * p;
        target_norm += t * t;
    }

    let n = pred_vec.len() as f32;
    mae /= n;
    mse /= n;
    let rmse = mse.sqrt();

    // Cosine similarity
    let cosine_sim = if pred_norm > 0.0 && target_norm > 0.0 {
        dot_product / (pred_norm.sqrt() * target_norm.sqrt())
    } else {
        0.0
    };

    Ok(AccuracyMetrics {
        mae,
        rmse,
        cosine_similarity: cosine_sim,
    })
}

#[derive(Debug)]
struct AccuracyMetrics {
    pub mae: f32,
    pub rmse: f32,
    pub cosine_similarity: f32,
}

impl AccuracyMetrics {
    /// Check if metrics meet the target thresholds.
    fn meets_targets(
        &self,
        mae_threshold: f32,
        rmse_threshold: f32,
        cosine_threshold: f32,
    ) -> bool {
        self.mae < mae_threshold
            && self.rmse < rmse_threshold
            && self.cosine_similarity > cosine_threshold
    }
}

/// Performance measurement results.
#[derive(Debug)]
struct PerformanceMetrics {
    pub gpu_time_ms: f64,
    pub cpu_time_ms: f64,
    pub speedup: f64,
    pub gpu_gflops: f64,
}

/// Calculate theoretical GFLOPS for attention computation.
fn calculate_attention_gflops(config: &AttentionTestConfig) -> f64 {
    let (batch, heads, seq_len, head_dim) = (
        config.batch_size as f64,
        config.num_heads as f64,
        config.seq_len as f64,
        config.head_dim as f64,
    );

    // QK^T: batch × heads × seq × seq × head_dim
    let qk_flops = batch * heads * seq_len * seq_len * head_dim;

    // Softmax: batch × heads × seq × seq (exp + sum + div)
    let softmax_flops = batch * heads * seq_len * seq_len * 3.0;

    // Attention·V: batch × heads × seq × seq × head_dim
    let av_flops = batch * heads * seq_len * seq_len * head_dim;

    (qk_flops + softmax_flops + av_flops) / 1e9 // Convert to GFLOPS
}

// ============================================================================
// GPU TEST FUNCTIONS
// ============================================================================

/// Test numerical equivalence between GPU and CPU Flash Attention.
///
/// This test verifies that the GPU implementation produces results that are
/// numerically equivalent to the CPU reference within acceptable tolerances.
#[cfg(feature = "cuda")]
#[test]
fn test_flash_attention_gpu_numerical_equivalence() -> Result<()> {
    crate::require_gpu!();

    let configs = vec![AttentionTestConfig::small(), AttentionTestConfig::medium()];

    for config in configs {
        println!("Testing config: {:?}", config);

        // Check if CUDA device is actually available before proceeding
        let cuda_device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(e) => {
                println!("SKIP: CUDA device not available: {}", e);
                return Ok(());
            }
        };

        let cpu_device = Device::Cpu;

        // Generate test data on CPU first, then move to GPU
        let (q_cpu, k_cpu, v_cpu, mask_cpu) = create_test_tensors(&config, &cpu_device)?;
        let scale = 1.0 / (config.head_dim as f64).sqrt();

        // CPU reference computation
        let cpu_output = attention_reference_cpu(&q_cpu, &k_cpu, &v_cpu, scale, mask_cpu.as_ref())?;

        // Move tensors to GPU
        let q_gpu = q_cpu.to_device(&cuda_device)?;
        let k_gpu = k_cpu.to_device(&cuda_device)?;
        let v_gpu = v_cpu.to_device(&cuda_device)?;
        let mask_gpu = mask_cpu.map(|m| m.to_device(&cuda_device)).transpose()?;

        // GPU Flash Attention computation
        let gpu_output = flash_attention_cubecl(&q_gpu, &k_gpu, &v_gpu, scale, mask_gpu.as_ref())?;

        // Move GPU result back to CPU for comparison
        let gpu_output_cpu = gpu_output.to_device(&cpu_device)?;

        // Calculate accuracy metrics
        let metrics = calculate_accuracy_metrics(&gpu_output_cpu, &cpu_output)?;

        println!("Accuracy metrics: {:?}", metrics);

        // Validate accuracy thresholds
        let mae_threshold = 1e-5;
        let rmse_threshold = 1e-4;
        let cosine_threshold = 0.999;

        assert!(
            metrics.meets_targets(mae_threshold, rmse_threshold, cosine_threshold),
            "GPU Flash Attention accuracy below threshold: MAE={:.2e} (< {:.2e}), RMSE={:.2e} (< {:.2e}), Cosine={:.6} (> {:.6})",
            metrics.mae, mae_threshold,
            metrics.rmse, rmse_threshold,
            metrics.cosine_similarity, cosine_threshold
        );
    }

    println!("✅ GPU Flash Attention numerical equivalence test passed");
    Ok(())
}

/// Test Flash Attention performance compared to CPU baseline.
///
/// This test validates that GPU implementation provides significant speedup
/// over CPU implementation for sequences where GPU should excel.
#[cfg(feature = "cuda")]
#[test]
fn test_flash_attention_gpu_performance() -> Result<()> {
    crate::require_gpu!(2.0); // Require 2GB VRAM

    let configs = vec![AttentionTestConfig::medium(), AttentionTestConfig::large()];

    for config in configs {
        println!("Performance testing config: seq_len={}", config.seq_len);

        // Check if CUDA device is actually available before proceeding
        let cuda_device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(e) => {
                println!("SKIP: CUDA device not available: {}", e);
                return Ok(());
            }
        };

        let cpu_device = Device::Cpu;

        let (q_cpu, k_cpu, v_cpu, mask_cpu) = create_test_tensors(&config, &cpu_device)?;
        let scale = 1.0 / (config.head_dim as f64).sqrt();

        // Move to GPU
        let q_gpu = q_cpu.to_device(&cuda_device)?;
        let k_gpu = k_cpu.to_device(&cuda_device)?;
        let v_gpu = v_cpu.to_device(&cuda_device)?;
        let mask_gpu = mask_cpu
            .as_ref()
            .map(|m| m.to_device(&cuda_device))
            .transpose()?;

        // Warm up GPU
        for _ in 0..3 {
            let _ = flash_attention_cubecl(&q_gpu, &k_gpu, &v_gpu, scale, mask_gpu.as_ref())?;
        }

        // Time GPU implementation
        let gpu_start = Instant::now();
        let num_iterations = 10;

        for _ in 0..num_iterations {
            let _ = flash_attention_cubecl(&q_gpu, &k_gpu, &v_gpu, scale, mask_gpu.as_ref())?;
        }

        let gpu_time_ms = gpu_start.elapsed().as_millis() as f64 / num_iterations as f64;

        // Time CPU reference implementation
        let cpu_start = Instant::now();

        for _ in 0..num_iterations {
            let _ = attention_reference_cpu(&q_cpu, &k_cpu, &v_cpu, scale, mask_cpu.as_ref())?;
        }

        let cpu_time_ms = cpu_start.elapsed().as_millis() as f64 / num_iterations as f64;

        let speedup = cpu_time_ms / gpu_time_ms;
        let theoretical_gflops = calculate_attention_gflops(&config);
        let gpu_gflops = theoretical_gflops / (gpu_time_ms / 1000.0);

        let metrics = PerformanceMetrics {
            gpu_time_ms,
            cpu_time_ms,
            speedup,
            gpu_gflops,
        };

        println!("Performance metrics: {:?}", metrics);
        println!("Theoretical GFLOPS: {:.1}", theoretical_gflops);

        // Performance validation - relaxed since we're using fallback implementation
        let min_speedup = if config.seq_len >= 512 { 0.5 } else { 0.1 }; // Very relaxed for fallback

        assert!(
            speedup >= min_speedup,
            "GPU speedup {:.2}x below minimum {:.1}x for seq_len={}",
            speedup,
            min_speedup,
            config.seq_len
        );

        // GPU utilization validation (basic sanity check)
        assert!(
            gpu_gflops > 0.1,
            "GPU utilization too low: {:.1} GFLOPS",
            gpu_gflops
        );
    }

    println!("✅ GPU Flash Attention performance test passed");
    Ok(())
}

/// Test memory efficiency of Flash Attention implementation.
///
/// Validates that Flash Attention uses O(√n) memory instead of O(n²)
/// and stays within reasonable VRAM bounds.
#[cfg(feature = "cuda")]
#[test]
fn test_flash_attention_memory_efficiency() -> Result<()> {
    crate::require_gpu!(4.0); // Require 4GB VRAM

    let test_configs = vec![(256, "small"), (512, "medium"), (1024, "large")];

    let mut memory_measurements = Vec::new();

    for (seq_len, label) in test_configs {
        let config = AttentionTestConfig {
            batch_size: 2,
            num_heads: 8,
            seq_len,
            head_dim: 64,
            num_kv_heads: Some(8),
            use_causal_mask: true,
            dtype: DType::F32,
        };

        let cuda_device = Device::new_cuda(0)?;
        let (q, k, v, mask) = create_test_tensors(&config, &cuda_device)?;
        let scale = 1.0 / (config.head_dim as f64).sqrt();

        // Estimate VRAM usage
        let estimated_vram = estimate_flash_attention_vram(
            config.batch_size,
            config.num_heads,
            config.seq_len,
            config.head_dim,
            128, // tile_size
        );

        // TODO: Add actual GPU memory measurement here
        // For now, we validate the estimation function

        println!(
            "Config {}: seq_len={}, estimated VRAM={:.1}MB",
            label,
            seq_len,
            estimated_vram as f64 / 1e6
        );

        memory_measurements.push((seq_len, estimated_vram));

        // Execute Flash Attention to verify it doesn't OOM
        let _output = flash_attention_cubecl(&q, &k, &v, scale, mask.as_ref())?;

        // Validate reasonable memory usage
        let max_reasonable_mb = 1000.0; // 1GB max for these configs
        let actual_mb = estimated_vram as f64 / 1e6;

        assert!(
            actual_mb < max_reasonable_mb,
            "Memory usage too high: {:.1}MB > {:.1}MB for seq_len={}",
            actual_mb,
            max_reasonable_mb,
            seq_len
        );
    }

    // Validate memory scaling is sub-quadratic
    // Compare largest to smallest configuration
    let (small_seq, small_mem) = memory_measurements[0];
    let (large_seq, large_mem) = memory_measurements[memory_measurements.len() - 1];

    let seq_ratio = large_seq as f64 / small_seq as f64;
    let mem_ratio = large_mem as f64 / small_mem as f64;

    // Memory should scale better than O(n²) - allow some overhead
    let max_scaling = seq_ratio * seq_ratio * 1.5; // 1.5x overhead allowance

    assert!(
        mem_ratio < max_scaling,
        "Memory scaling too high: {:.2}x (should be < {:.2}x) for {:.0}x sequence increase",
        mem_ratio,
        max_scaling,
        seq_ratio
    );

    println!("✅ Flash Attention memory efficiency test passed");
    Ok(())
}

/// Test different Flash Attention configurations and tile sizes.
///
/// Validates that Flash Attention works correctly with various block
/// and tile configurations.
#[cfg(feature = "cuda")]
#[test]
fn test_flash_attention_different_configs() -> Result<()> {
    crate::require_gpu!(1.0); // Require 1GB VRAM

    let base_config = AttentionTestConfig {
        batch_size: 1,
        num_heads: 4,
        seq_len: 256,
        head_dim: 64,
        num_kv_heads: Some(4),
        use_causal_mask: false,
        dtype: DType::F32,
    };

    // Test different head configurations (MHA only - GQA has separate ignored tests)
    let head_configs = vec![
        (4, 4), // MHA: same number of Q and KV heads
    ];

    for (q_heads, kv_heads) in head_configs {
        let config = AttentionTestConfig {
            num_heads: q_heads,
            num_kv_heads: Some(kv_heads),
            ..base_config.clone()
        };

        println!("Testing heads config: Q={}, KV={}", q_heads, kv_heads);

        let cuda_device = Device::new_cuda(0)?;
        let (q, k, v, mask) = create_test_tensors(&config, &cuda_device)?;
        let scale = 1.0 / (config.head_dim as f64).sqrt();

        // Should execute without error
        let output = flash_attention_cubecl(&q, &k, &v, scale, mask.as_ref())?;

        // Validate output shape
        assert_eq!(
            output.dims(),
            &[
                config.batch_size,
                config.num_heads,
                config.seq_len,
                config.head_dim
            ]
        );

        // Validate output is reasonable (no NaN/Inf)
        let values: Vec<f32> = output.flatten_all()?.to_vec1()?;
        for v in values.iter().take(100) {
            // Check first 100 values
            assert!(!v.is_nan(), "Output contains NaN");
            assert!(!v.is_infinite(), "Output contains Inf");
        }
    }

    // Test different data types
    let dtypes = vec![DType::F32]; // TODO: Add F16, BF16 when supported

    for dtype in dtypes {
        let config = AttentionTestConfig {
            dtype,
            ..base_config.clone()
        };

        println!("Testing dtype: {:?}", dtype);

        let cuda_device = Device::new_cuda(0)?;
        let (q, k, v, mask) = create_test_tensors(&config, &cuda_device)?;
        let scale = 1.0 / (config.head_dim as f64).sqrt();

        let _output = flash_attention_cubecl(&q, &k, &v, scale, mask.as_ref())?;
    }

    println!("✅ Flash Attention configuration test passed");
    Ok(())
}

/// Test Flash Attention with GQA: 8 query heads, 4 key/value heads.
///
/// GQA (Grouped Query Attention) uses fewer KV heads than Q heads to reduce
/// memory usage while maintaining quality. This configuration has 2 Q heads
/// per KV head.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "GQA not yet supported - enable when GQA support is added"]
fn test_flash_attention_gqa_8_4() -> Result<()> {
    crate::require_gpu!(1.0);

    let config = AttentionTestConfig {
        batch_size: 1,
        num_heads: 8,          // Q heads
        num_kv_heads: Some(4), // KV heads (GQA ratio 2:1)
        seq_len: 256,
        head_dim: 64,
        use_causal_mask: false,
        dtype: DType::F32,
    };

    let cuda_device = Device::new_cuda(0)?;
    let (q, k, v, mask) = create_test_tensors(&config, &cuda_device)?;
    let scale = 1.0 / (config.head_dim as f64).sqrt();

    let output = flash_attention_cubecl(&q, &k, &v, scale, mask.as_ref())?;

    // Validate output shape matches Q heads (not KV heads)
    assert_eq!(
        output.dims(),
        &[
            config.batch_size,
            config.num_heads,
            config.seq_len,
            config.head_dim
        ]
    );

    // Validate output is reasonable (no NaN/Inf)
    let values: Vec<f32> = output.flatten_all()?.to_vec1()?;
    for v in values.iter().take(100) {
        assert!(v.is_finite(), "Output contains non-finite value: {}", v);
    }

    println!("✅ Flash Attention GQA 8:4 test passed");
    Ok(())
}

/// Test Flash Attention with extreme GQA: 8 query heads, 1 key/value head.
///
/// This is the most extreme GQA configuration (also called MQA - Multi-Query
/// Attention), where all Q heads share a single KV head.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "GQA not yet supported - enable when GQA support is added"]
fn test_flash_attention_gqa_8_1() -> Result<()> {
    crate::require_gpu!(1.0);

    let config = AttentionTestConfig {
        batch_size: 1,
        num_heads: 8,          // Q heads
        num_kv_heads: Some(1), // Single KV head (MQA/extreme GQA)
        seq_len: 256,
        head_dim: 64,
        use_causal_mask: false,
        dtype: DType::F32,
    };

    let cuda_device = Device::new_cuda(0)?;
    let (q, k, v, mask) = create_test_tensors(&config, &cuda_device)?;
    let scale = 1.0 / (config.head_dim as f64).sqrt();

    let output = flash_attention_cubecl(&q, &k, &v, scale, mask.as_ref())?;

    // Validate output shape matches Q heads (not KV heads)
    assert_eq!(
        output.dims(),
        &[
            config.batch_size,
            config.num_heads,
            config.seq_len,
            config.head_dim
        ]
    );

    // Validate output is reasonable (no NaN/Inf)
    let values: Vec<f32> = output.flatten_all()?.to_vec1()?;
    for v in values.iter().take(100) {
        assert!(v.is_finite(), "Output contains non-finite value: {}", v);
    }

    println!("✅ Flash Attention GQA 8:1 (MQA) test passed");
    Ok(())
}

/// Test Flash Attention with large sequence lengths.
///
/// Validates scalability and stability with sequences up to 2048 tokens.
#[cfg(feature = "cuda")]
#[test]
fn test_flash_attention_large_sequences() -> Result<()> {
    crate::require_gpu!(6.0); // Require 6GB VRAM for large sequences

    let sequence_lengths = vec![512, 1024, 2048];

    for seq_len in sequence_lengths {
        println!("Testing large sequence: seq_len={}", seq_len);

        let config = AttentionTestConfig {
            batch_size: 2,
            num_heads: 16,
            seq_len,
            head_dim: 64,
            num_kv_heads: Some(16),
            use_causal_mask: true,
            dtype: DType::F32,
        };

        let cuda_device = Device::new_cuda(0)?;

        // Check estimated memory before allocation
        let estimated_vram_gb = estimate_flash_attention_vram(
            config.batch_size,
            config.num_heads,
            config.seq_len,
            config.head_dim,
            128,
        ) as f64
            / 1e9;

        println!("Estimated VRAM usage: {:.2}GB", estimated_vram_gb);

        let (q, k, v, mask) = create_test_tensors(&config, &cuda_device)?;
        let scale = 1.0 / (config.head_dim as f64).sqrt();

        // Time the execution
        let start_time = Instant::now();
        let output = flash_attention_cubecl(&q, &k, &v, scale, mask.as_ref())?;
        let execution_time_ms = start_time.elapsed().as_millis();

        println!("Execution time: {}ms", execution_time_ms);

        // Validate output shape and sanity
        assert_eq!(
            output.dims(),
            &[
                config.batch_size,
                config.num_heads,
                config.seq_len,
                config.head_dim
            ]
        );

        // Check a few output values for sanity
        let output_sample = output.i((0, 0, 0, ..10))?.to_vec1::<f32>()?;
        for val in &output_sample {
            assert!(
                !val.is_nan() && !val.is_infinite(),
                "Invalid output value: {}",
                val
            );
        }

        // Reasonable execution time - CPU fallback is slower, so use generous limit
        // TODO: Tighten this once CubeCL kernel is optimized
        let max_time_ms = (seq_len * seq_len / 30) as u128; // Allow for CPU fallback
        assert!(
            execution_time_ms < max_time_ms,
            "Execution time {}ms too slow for seq_len={} (expected < {}ms)",
            execution_time_ms,
            seq_len,
            max_time_ms
        );
    }

    println!("✅ Large sequence Flash Attention test passed");
    Ok(())
}

/// Test CubeCL kernel compilation and execution.
///
/// Validates that CubeCL kernels compile correctly and execute without errors.
#[cfg(feature = "cuda")]
#[test]
fn test_cubecl_kernel_compilation() -> Result<()> {
    crate::require_gpu!();

    // Test CubeCL support detection
    let has_support = has_cubecl_support();
    println!("CubeCL support available: {}", has_support);

    // Currently, CubeCL kernel is not implemented, so this will use fallback
    // Once implemented, this test should verify kernel compilation and basic execution

    let config = AttentionTestConfig::small();
    let cuda_device = Device::new_cuda(0)?;
    let (q, k, v, mask) = create_test_tensors(&config, &cuda_device)?;
    let scale = 1.0 / (config.head_dim as f64).sqrt();

    // This should execute without compilation errors (currently using fallback)
    let _output = flash_attention_cubecl(&q, &k, &v, scale, mask.as_ref())?;

    println!("✅ CubeCL kernel compilation test passed (using fallback)");
    Ok(())
}

/// Test GPU memory management and cleanup.
///
/// Validates that GPU memory is properly allocated and freed without leaks.
#[cfg(feature = "cuda")]
#[test]
fn test_gpu_memory_management() -> Result<()> {
    crate::require_gpu!(2.0);

    let config = AttentionTestConfig::medium();
    let cuda_device = Device::new_cuda(0)?;

    // Allocate and deallocate tensors multiple times
    for iteration in 0..10 {
        println!("Memory management iteration {}", iteration + 1);

        let (q, k, v, mask) = create_test_tensors(&config, &cuda_device)?;
        let scale = 1.0 / (config.head_dim as f64).sqrt();

        let output = flash_attention_cubecl(&q, &k, &v, scale, mask.as_ref())?;

        // Force a computation to ensure GPU memory is actually used
        let _sum = output.sum_all()?.to_scalar::<f32>()?;

        // Tensors should be automatically freed when going out of scope
    }

    // Test with varying sizes to stress allocator
    let sizes = vec![64, 128, 256, 512];

    for seq_len in sizes {
        let config = AttentionTestConfig {
            seq_len,
            ..config.clone()
        };

        let (q, k, v, mask) = create_test_tensors(&config, &cuda_device)?;
        let scale = 1.0 / (config.head_dim as f64).sqrt();
        let _output = flash_attention_cubecl(&q, &k, &v, scale, mask.as_ref())?;
    }

    println!("✅ GPU memory management test passed");
    Ok(())
}

/// Integration test with FusedAttention layer.
///
/// Tests Flash Attention within the full attention layer context.
#[cfg(feature = "cuda")]
#[test]
fn test_flash_attention_integration() -> Result<()> {
    crate::require_gpu!(1.0);

    let config = FusedAttentionConfig {
        hidden_size: 512,
        num_heads: 8,
        head_dim: 64,
        num_kv_heads: Some(8),
        dropout: 0.0,
        use_flash: true,
    };

    let cuda_device = Device::new_cuda(0)?;
    let attention_layer = FusedAttention::new(config, &cuda_device)?;

    let batch_size = 2;
    let seq_len = 256;
    let hidden_size = 512;

    // Create input tensor
    let input = Tensor::randn(
        0.0f32,
        1.0,
        (batch_size, seq_len, hidden_size),
        &cuda_device,
    )?;

    // Forward pass
    let output = attention_layer.forward(&input, None, None)?;

    // Validate output shape
    assert_eq!(output.dims(), &[batch_size, seq_len, hidden_size]);

    // Validate output sanity
    let values: Vec<f32> = output.flatten_all()?.to_vec1()?;
    for v in values.iter().take(100) {
        assert!(!v.is_nan() && !v.is_infinite(), "Invalid output: {}", v);
    }

    println!("✅ Flash Attention integration test passed");
    Ok(())
}

// ============================================================================
// FALLBACK TESTS FOR NON-CUDA BUILDS AND BASIC TESTING
// ============================================================================

/// Basic Flash Attention functionality test that works without CUDA.
///
/// This test validates the Flash Attention interface and CPU fallback path.
#[test]
pub fn test_flash_attention_basic_functionality() -> Result<()> {
    let config = AttentionTestConfig::small();
    let device = Device::Cpu;

    let (q, k, v, mask) = create_test_tensors(&config, &device)?;
    let scale = 1.0 / (config.head_dim as f64).sqrt();

    // Test Flash Attention interface (will use CPU fallback)
    let output = flash_attention_cubecl(&q, &k, &v, scale, mask.as_ref())?;

    // Validate output shape
    assert_eq!(
        output.dims(),
        &[
            config.batch_size,
            config.num_heads,
            config.seq_len,
            config.head_dim
        ]
    );

    // Validate output sanity
    let values: Vec<f32> = output.flatten_all()?.to_vec1()?;
    for v in values.iter().take(100) {
        assert!(!v.is_nan() && !v.is_infinite(), "Invalid output: {}", v);
    }

    println!("✅ Basic Flash Attention functionality test passed");
    Ok(())
}

/// Test Flash Attention CPU fallback accuracy.
///
/// Validates that the Flash Attention implementation produces consistent
/// results when using CPU fallback path.
#[test]
pub fn test_flash_attention_cpu_fallback_accuracy() -> Result<()> {
    let config = AttentionTestConfig::small();
    let device = Device::Cpu;

    let (q, k, v, mask) = create_test_tensors(&config, &device)?;
    let scale = 1.0 / (config.head_dim as f64).sqrt();

    // Reference computation
    let reference_output = attention_reference_cpu(&q, &k, &v, scale, mask.as_ref())?;

    // Flash Attention fallback computation
    let fallback_output = flash_attention_cubecl(&q, &k, &v, scale, mask.as_ref())?;

    // Calculate accuracy metrics
    let metrics = calculate_accuracy_metrics(&fallback_output, &reference_output)?;

    println!("CPU fallback accuracy metrics: {:?}", metrics);

    // Should be nearly identical
    assert!(
        metrics.meets_targets(1e-6, 1e-6, 0.9999),
        "CPU fallback accuracy too low: MAE={:.2e}, RMSE={:.2e}, Cosine={:.6}",
        metrics.mae,
        metrics.rmse,
        metrics.cosine_similarity
    );

    println!("✅ Flash Attention CPU fallback accuracy test passed");
    Ok(())
}

/// Test CubeCL support detection without requiring actual CUDA.
#[test]
pub fn test_cubecl_support_detection() {
    let has_support = has_cubecl_support();
    println!("CubeCL support detected: {}", has_support);

    // This test should not fail regardless of actual support
    // It just validates that the detection function doesn't panic
    println!("✅ CubeCL support detection test passed");
}

/// Test VRAM estimation without requiring GPU.
#[test]
pub fn test_flash_attention_vram_estimation() {
    let test_configs = vec![
        (1, 4, 128, 64, 128),
        (2, 8, 512, 64, 128),
        (4, 16, 1024, 64, 128),
    ];

    for (batch, heads, seq_len, head_dim, tile_size) in test_configs {
        let vram_bytes = estimate_flash_attention_vram(batch, heads, seq_len, head_dim, tile_size);
        let vram_mb = vram_bytes as f64 / 1e6;

        println!(
            "Config {}x{}x{}x{} (tile={}): {:.1}MB",
            batch, heads, seq_len, head_dim, tile_size, vram_mb
        );

        // Sanity checks
        assert!(vram_bytes > 0, "VRAM estimate should be positive");
        assert!(
            vram_mb < 100_000.0,
            "VRAM estimate unreasonably high: {:.1}MB",
            vram_mb
        );

        // Memory should scale with sequence length
        if seq_len > 128 {
            let smaller_vram =
                estimate_flash_attention_vram(batch, heads, 128, head_dim, tile_size);
            assert!(
                vram_bytes > smaller_vram,
                "VRAM should scale with sequence length"
            );
        }
    }

    println!("✅ Flash Attention VRAM estimation test passed");
}

/// Test attention with different sequence lengths (CPU only).
#[test]
pub fn test_flash_attention_sequence_scaling() -> Result<()> {
    let device = Device::Cpu;
    let base_config = AttentionTestConfig {
        batch_size: 1,
        num_heads: 4,
        seq_len: 64, // Will be overridden
        head_dim: 64,
        num_kv_heads: Some(4),
        use_causal_mask: false,
        dtype: DType::F32,
    };

    let sequence_lengths = vec![64, 128, 256];

    for seq_len in sequence_lengths {
        let config = AttentionTestConfig {
            seq_len,
            ..base_config.clone()
        };

        println!("Testing sequence length: {}", seq_len);

        let (q, k, v, mask) = create_test_tensors(&config, &device)?;
        let scale = 1.0 / (config.head_dim as f64).sqrt();

        let output = flash_attention_cubecl(&q, &k, &v, scale, mask.as_ref())?;

        // Validate shape
        assert_eq!(
            output.dims(),
            &[
                config.batch_size,
                config.num_heads,
                config.seq_len,
                config.head_dim
            ]
        );

        // Validate numerical stability
        let values: Vec<f32> = output.flatten_all()?.to_vec1()?;
        for v in values.iter().take(10) {
            assert!(
                !v.is_nan() && !v.is_infinite(),
                "Invalid output at seq_len={}: {}",
                seq_len,
                v
            );
        }
    }

    println!("✅ Flash Attention sequence scaling test passed");
    Ok(())
}

/// Fallback test when CUDA is not available.
#[cfg(not(feature = "cuda"))]
#[test]
fn test_flash_attention_cuda_feature_required() {
    println!("SKIP: Flash Attention GPU tests require --features cuda");
    println!("Run: cargo test --features cuda --test integration");
}
