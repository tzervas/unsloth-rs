//! Kernel benchmarking suite for CPU performance profiling.
//!
//! This benchmark suite measures performance for all kernel implementations:
//! - Attention (multi-head with GQA support)
//! - RoPE (Rotary Position Embeddings)
//! - RMSNorm
//! - SwiGLU activation
//!
//! Note: These are CPU benchmarks. GPU benchmarks require the `cuda` feature.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use unsloth_rs::kernels::{
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

    for &batch_size in &[1, 4] {
        for &seq_len in SEQ_LENS {
            let config = FusedAttentionConfig {
                hidden_size: 768,
                num_heads: 12,
                head_dim: 64,
                ..Default::default()
            };

            let attention = match FusedAttention::new(config.clone(), &device) {
                Ok(a) => a,
                Err(_) => continue,
            };

            let input = match Tensor::randn(0.0f32, 1.0, (batch_size, seq_len, 768), &device) {
                Ok(t) => t,
                Err(_) => continue,
            };

            let id = format!("b{}_s{}", batch_size, seq_len);
            group.bench_with_input(BenchmarkId::new("cpu", &id), &(&attention, &input), |b, (attn, inp)| {
                b.iter(|| attn.forward(inp, None, None).unwrap());
            });

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

    for &batch_size in &[1, 4] {
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

    for &batch_size in &[1, 4] {
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
                group.bench_with_input(BenchmarkId::new("cpu", &id), &(&norm, &input), |b, (norm, inp)| {
                    b.iter(|| norm.forward(inp).unwrap());
                });
            }
        }
    }

    group.finish();
}

/// Benchmark SwiGLU forward pass
fn benchmark_swiglu(c: &mut Criterion) {
    let mut group = c.benchmark_group("swiglu");
    let device = Device::Cpu;

    for &batch_size in &[1, 4] {
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

/// Combined benchmark for all kernels
fn benchmark_all_kernels(c: &mut Criterion) {
    benchmark_attention(c);
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
