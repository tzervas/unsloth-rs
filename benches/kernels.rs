//! Kernel benchmarking suite for CPU performance profiling.
//!
//! This benchmark suite measures performance for all kernel implementations:
//! - Attention (multi-head with GQA support)
//! - Flash Attention (CubeCL / fallback)
//! - RoPE (Rotary Position Embeddings)
//! - RMSNorm
//! - SwiGLU activation
//!
//! ## Running Benchmarks
//!
//! CPU benchmarks:
//! ```bash
//! cargo bench -p unsloth-rs
//! ```
//!
//! GPU benchmarks (requires CUDA):
//! ```bash
//! CUBECL_PROFILE=1 cargo bench -p unsloth-rs --features cuda
//! ```

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use unsloth_rs::kernels::{
    cubecl::{flash_attention_kernel, FlashAttentionConfig},
    FusedAttention, FusedAttentionConfig, RmsNorm, RotaryEmbedding, SwiGLU,
};

use candle_core::{DType, Device, Tensor};

/// Benchmark configurations for testing various input sizes
const BATCH_SIZES: &[usize] = &[1, 4];
const SEQ_LENS: &[usize] = &[512, 1024, 2048];
const HIDDEN_SIZES: &[usize] = &[768, 1024];

/// Benchmark attention forward pass
fn benchmark_attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention");
    let device = Device::Cpu;

    for &batch_size in BATCH_SIZES {
        for &seq_len in SEQ_LENS {
            let config = FusedAttentionConfig {
                hidden_size: 768,
                num_heads: 12,
                head_dim: 64,
                ..Default::default()
            };

            let attention = match FusedAttention::new(config, &device) {
                Ok(a) => a,
                Err(_) => continue,
            };

            let input = match Tensor::randn(0.0f32, 1.0, (batch_size, seq_len, 768), &device) {
                Ok(t) => t,
                Err(_) => continue,
            };

            let id = format!("b{}_s{}", batch_size, seq_len);
            group.bench_with_input(
                BenchmarkId::new("cpu", &id),
                &(&attention, &input),
                |b, (attn, inp)| {
                    b.iter(|| attn.forward(inp, None, None).unwrap());
                },
            );

            // Report memory estimate
            let vram = attention.vram_estimate(batch_size, seq_len);
            println!(
                "Attention memory estimate (batch={}, seq={}): {} MB",
                batch_size,
                seq_len,
                vram / 1024 / 1024
            );
        }
    }

    group.finish();
}

/// Benchmark RoPE (Rotary Position Embedding) forward pass
fn benchmark_rope(c: &mut Criterion) {
    let mut group = c.benchmark_group("rope");
    let device = Device::Cpu;

    for &batch_size in BATCH_SIZES {
        for &seq_len in SEQ_LENS {
            let head_dim = 64;
            let num_heads = 12;

            let rope = match RotaryEmbedding::new(head_dim, 4096, 10000.0, &device) {
                Ok(r) => r,
                Err(_) => continue,
            };

            let q = match Tensor::randn(
                0.0f32,
                1.0,
                (batch_size, num_heads, seq_len, head_dim),
                &device,
            ) {
                Ok(t) => t,
                Err(_) => continue,
            };

            let k = match Tensor::randn(
                0.0f32,
                1.0,
                (batch_size, num_heads, seq_len, head_dim),
                &device,
            ) {
                Ok(t) => t,
                Err(_) => continue,
            };

            let pos = match Tensor::zeros((batch_size, seq_len), DType::I64, &device) {
                Ok(t) => t,
                Err(_) => continue,
            };

            let id = format!("b{}_s{}", batch_size, seq_len);
            group.bench_with_input(
                BenchmarkId::new("cpu", &id),
                &(&rope, &q, &k, &pos),
                |b, (rope, q, k, pos)| {
                    b.iter(|| rope.forward(q, k, pos).unwrap());
                },
            );
        }
    }

    group.finish();
}

/// Benchmark RMSNorm forward pass
fn benchmark_rmsnorm(c: &mut Criterion) {
    let mut group = c.benchmark_group("rmsnorm");
    let device = Device::Cpu;

    for &batch_size in BATCH_SIZES {
        for &seq_len in SEQ_LENS {
            for &hidden_size in HIDDEN_SIZES {
                let norm = match RmsNorm::new(hidden_size, 1e-5, &device) {
                    Ok(n) => n,
                    Err(_) => continue,
                };

                let input =
                    match Tensor::randn(0.0f32, 1.0, (batch_size, seq_len, hidden_size), &device) {
                        Ok(t) => t,
                        Err(_) => continue,
                    };

                let id = format!("b{}_s{}_h{}", batch_size, seq_len, hidden_size);
                group.bench_with_input(
                    BenchmarkId::new("cpu", &id),
                    &(&norm, &input),
                    |b, (norm, inp)| {
                        b.iter(|| norm.forward(inp).unwrap());
                    },
                );
            }
        }
    }

    group.finish();
}

/// Benchmark SwiGLU forward pass
fn benchmark_swiglu(c: &mut Criterion) {
    let mut group = c.benchmark_group("swiglu");
    let device = Device::Cpu;

    for &batch_size in BATCH_SIZES {
        for &seq_len in SEQ_LENS {
            for &hidden_size in HIDDEN_SIZES {
                // Typical intermediate size is ~2.7x hidden_size for LLaMA-style models
                let intermediate_size = (hidden_size as f64 * 2.7) as usize;

                let swiglu = match SwiGLU::new(hidden_size, intermediate_size, &device) {
                    Ok(s) => s,
                    Err(_) => continue,
                };

                let input =
                    match Tensor::randn(0.0f32, 1.0, (batch_size, seq_len, hidden_size), &device) {
                        Ok(t) => t,
                        Err(_) => continue,
                    };

                let id = format!("b{}_s{}_h{}", batch_size, seq_len, hidden_size);
                group.bench_with_input(
                    BenchmarkId::new("cpu", &id),
                    &(&swiglu, &input),
                    |b, (swiglu, inp)| {
                        b.iter(|| swiglu.forward(inp).unwrap());
                    },
                );

                // Report VRAM estimate
                let vram = swiglu.vram_estimate(batch_size, seq_len);
                println!(
                    "SwiGLU VRAM estimate (batch={}, seq={}, hidden={}): {} MB",
                    batch_size,
                    seq_len,
                    hidden_size,
                    vram / 1024 / 1024
                );
            }
        }
    }

    group.finish();
}

/// Benchmark Flash Attention (CubeCL kernel or Candle fallback)
///
/// Run with CUBECL_PROFILE=1 for kernel-level timing:
/// ```bash
/// CUBECL_PROFILE=1 cargo bench -p unsloth-rs --features cuda flash_attention
/// ```
fn benchmark_flash_attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("flash_attention");
    let device = Device::Cpu; // Will use CUDA if available

    // Small sizes for initial validation
    let configs = [
        (2, 4, 8, 64, "tiny"),
        (2, 4, 64, 64, "small"),
        (2, 8, 128, 64, "medium"),
        (1, 12, 256, 64, "llama_small"),
        (1, 32, 512, 128, "llama_med"),
    ];

    for (batch, heads, seq, dim, label) in configs {
        // Skip larger benchmarks on CI
        if seq > 128 && std::env::var("CI").is_ok() {
            continue;
        }

        let q = match Tensor::randn(0.0f32, 1.0, (batch, heads, seq, dim), &device) {
            Ok(t) => t,
            Err(_) => continue,
        };
        let k = match Tensor::randn(0.0f32, 1.0, (batch, heads, seq, dim), &device) {
            Ok(t) => t,
            Err(_) => continue,
        };
        let v = match Tensor::randn(0.0f32, 1.0, (batch, heads, seq, dim), &device) {
            Ok(t) => t,
            Err(_) => continue,
        };

        let scale = 1.0 / (dim as f64).sqrt();
        let config = FlashAttentionConfig::default();

        let id = format!(
            "{}_{}_b{}_h{}_s{}_d{}",
            label, "cpu", batch, heads, seq, dim
        );
        group.bench_with_input(
            BenchmarkId::new("forward", &id),
            &(&q, &k, &v, scale, &config),
            |b, (q, k, v, scale, cfg)| {
                b.iter(|| flash_attention_kernel(q, k, v, *scale, None, cfg).unwrap());
            },
        );

        // Calculate and print FLOP estimate
        // Flash Attention FLOPs: 4 * batch * heads * seq^2 * dim
        let flops = 4 * batch * heads * seq * seq * dim;
        println!(
            "Flash Attention {} ({} FLOPs): batch={}, heads={}, seq={}, dim={}",
            label,
            format_flops(flops as f64),
            batch,
            heads,
            seq,
            dim
        );

        // Estimate VRAM (for reference when running on GPU)
        let elem_size = 4; // f32
        let qkv_size = 3 * batch * heads * seq * dim * elem_size;
        let output_size = batch * heads * seq * dim * elem_size;
        println!(
            "  Estimated QKV size: {} KB, Output: {} KB",
            qkv_size / 1024,
            output_size / 1024
        );
    }

    group.finish();
}

/// Format FLOP count as human-readable string
fn format_flops(flops: f64) -> String {
    if flops >= 1e12 {
        format!("{:.2} TFLOP", flops / 1e12)
    } else if flops >= 1e9 {
        format!("{:.2} GFLOP", flops / 1e9)
    } else if flops >= 1e6 {
        format!("{:.2} MFLOP", flops / 1e6)
    } else {
        format!("{:.0} FLOP", flops)
    }
}

/// Combined benchmark for all kernels
fn benchmark_all_kernels(c: &mut Criterion) {
    benchmark_attention(c);
    benchmark_flash_attention(c);
    benchmark_rope(c);
    benchmark_rmsnorm(c);
    benchmark_swiglu(c);
}

criterion_group!(
    name = benches;
    config = Criterion::default()
        .sample_size(50)  // Increased from 10 for more statistically reliable results
        .measurement_time(std::time::Duration::from_secs(10));
    targets = benchmark_all_kernels
);
criterion_main!(benches);
