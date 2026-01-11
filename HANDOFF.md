# unsloth-rs Handoff - Phase 2: GPU Profiling & Optimization

**Date**: 2026-01-10  
**Target Environment**: akula-prime (RTX 5080 + RTX 3090 Ti, 28C/56T)  
**Status**: ✅ **GPU Hardware Available** - Ready for validation  
**Branch**: `experimental`

## Executive Summary

Phase 1 Flash Attention CubeCL kernel is complete with 148 passing tests. GPU hardware (RTX 5080) is now available for profiling and performance validation.

## Current State

### Implementation Status

| Component | Status | LOC | Notes |
|-----------|--------|-----|-------|
| CubeCL Module Structure | ✅ Complete | ~50 | `src/kernels/cubecl/mod.rs` |
| Candle ↔ CubeCL Interop | ✅ Complete | 263 | `src/kernels/cubecl/interop.rs` |
| GPU Config Presets | ✅ Complete | 206 | `src/kernels/cubecl/config.rs` |
| Flash Attention Kernels | ✅ Complete | 1015 | `src/kernels/cubecl/kernel.rs` |
| Causal Masking | ✅ Complete | - | Kernel-level + fallback |
| Numerical Validation | ✅ Complete | - | MAE < 1e-5 vs CPU |
| Test Suite | ✅ 65/65 | - | All passing |

### Performance Targets (Ready for Validation)

| Metric | Target | Validation Status |
|--------|--------|-------------------|
| Speedup vs Candle | 2-5x | 🔄 **Ready for GPU profiling with RTX 5080** |
| VRAM Reduction | 70-80% | 🔄 **Ready for GPU profiling with RTX 5080** |
| Numerical Accuracy | MAE < 1e-5 | ✅ Verified on CPU |

### Test Results

```
cargo test -p unsloth-rs
# 148 passed, 0 failed (updated 2026-01-10)
# Doc-tests: 2 passed, 6 ignored (require CUDA)
```

## Phase 2 Tasks

### 2.1 GPU Profiling on RTX 5080 (~50 LOC, 2-4 hrs)

**Objective**: Validate 2-5x speedup target

```rust
// benches/kernels.rs - Add GPU benchmark
#[cfg(feature = "cuda")]
fn bench_flash_attention_cuda(c: &mut Criterion) {
    let device = Device::new_cuda(0).unwrap();
    // ... setup tensors
    c.bench_function("flash_attn_cuda_seq512", |b| {
        b.iter(|| flash_attention_forward(&q, &k, &v, true))
    });
}
```

**Commands**:
```bash
# On akula-prime
docker run --gpus all -v $PWD:/workspace -w /workspace rust-cuda:latest \
    cargo bench -p unsloth-rs --features cuda -- flash_attention
```

### 2.2 VRAM Measurement (~100 LOC, 2-3 hrs)

**Objective**: Validate 70-80% VRAM reduction vs naive O(N²)

```rust
// Add to benches/kernels.rs
fn measure_vram_usage(seq_len: usize) -> (usize, usize) {
    // Measure before/after allocation
    let naive_vram = seq_len * seq_len * 4; // O(N²) for attention matrix
    let flash_vram = /* measure actual */;
    (naive_vram, flash_vram)
}
```

### 2.3 Longer Sequence Tests (~200 LOC, 3-4 hrs)

**Objective**: Validate kernel stability at production sequence lengths

| Sequence Length | Expected VRAM (Flash) | Expected VRAM (Naive) |
|-----------------|----------------------|----------------------|
| 512 | ~8 MB | ~1 MB |
| 1024 | ~16 MB | ~4 MB |
| 2048 | ~32 MB | ~16 MB |
| 4096 | ~64 MB | ~64 MB |

### 2.4 Cross-GPU Validation (~50 LOC, 2-3 hrs)

**Objective**: Ensure kernel works on both RTX 5080 and RTX 3090 Ti

```rust
// Preset validation
let rtx5080 = FlashAttentionConfig::rtx_5080_preset();  // tile_kv=256
let rtx3090 = FlashAttentionConfig::rtx_3090ti_preset(); // tile_kv=128
```

## Environment Setup

### Docker Configuration

```dockerfile
# Dockerfile.cuda
FROM nvidia/cuda:12.4-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    curl build-essential pkg-config libssl-dev

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install components
RUN rustup default 1.92.0
RUN rustup component add clippy rustfmt

WORKDIR /workspace
```

### Build Commands

```bash
# Build with CUDA feature
cargo build -p unsloth-rs --features cuda --release

# Run tests with CUDA
cargo test -p unsloth-rs --features cuda

# Run benchmarks
cargo bench -p unsloth-rs --features cuda
```

## File Locations

### Key Implementation Files

| File | Purpose |
|------|---------|
| `src/kernels/cubecl/kernel.rs` | Flash Attention-2 kernel implementation |
| `src/kernels/cubecl/config.rs` | GPU presets (RTX 5080, 3090 Ti) |
| `src/kernels/cubecl/interop.rs` | Candle ↔ CubeCL tensor conversion |
| `src/kernels/attention_cubecl.rs` | Public API and fallback |

### Documentation

| File | Purpose |
|------|---------|
| `docs/archive/HANDOFF_phase1_flash_attention.md` | Phase 1 complete handoff |
| `docs/archive/PHASE1_SUMMARY.md` | Phase 1 summary |
| `FLASH_ATTENTION_PLAN.md` | Overall implementation plan |
| `BENCHMARKING.md` | Profiling guide |

## Git Workflow

```bash
# Current branch
git checkout feature/unsloth-rs/flash-attention-completion

# After completing profiling tasks
git add -A
git commit -m "unsloth-rs: feat: add GPU profiling benchmarks

- Add RTX 5080 benchmark results
- Validate 2-5x speedup target
- Validate 70-80% VRAM reduction
- Test sequences up to 4096"

# PR to experimental (auto-merge if checks pass)
# Then PR to dev (requires @tzervas approval)
```

## Phase 3 Preview (After Profiling)

Once GPU profiling validates Phase 2 targets:

1. **f16/bf16 Support** (~300-500 LOC, 2-4 days)
2. **GQA/MQA Support** (~350-500 LOC, 2-3 days)
3. **Memory Optimizations** (~200-300 LOC, 2-3 days)
4. **Other Kernels** (RoPE, RMSNorm, SwiGLU)

## Contact

**Orchestrator**: Claude Opus (workspace coordination)  
**GPU Kernel Agent**: Claude Sonnet (complex kernel tasks)  
**Code Owner**: @tzervas
