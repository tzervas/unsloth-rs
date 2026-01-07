# Flash Attention Benchmarking Guide

This document describes how to benchmark and profile the Flash Attention CubeCL kernel on CUDA hardware.

## Prerequisites

1. **CUDA Toolkit** (12.x recommended)
2. **NVIDIA GPU** (RTX 5080, RTX 3090 Ti, or datacenter GPUs)
3. **Rust toolchain** with `cuda` feature support

## Quick Start

### CPU Benchmarks (No CUDA Required)

Run the fallback Candle implementation benchmarks:

```bash
cargo bench -p unsloth-rs flash_attention
```

### GPU Benchmarks (CUDA Required)

Build with CUDA support and run with profiling enabled:

```bash
# Build with CUDA feature
cargo build -p unsloth-rs --features cuda --release

# Run benchmarks with CubeCL profiling
CUBECL_PROFILE=1 cargo bench -p unsloth-rs --features cuda flash_attention
```

## Benchmark Configurations

The benchmark suite tests several configurations:

| Config | Batch | Heads | Seq | Dim | Description |
|--------|-------|-------|-----|-----|-------------|
| tiny | 2 | 4 | 8 | 64 | Unit test validation |
| small | 2 | 4 | 64 | 64 | Small model debugging |
| medium | 2 | 8 | 128 | 64 | Medium workload |
| llama_small | 1 | 12 | 256 | 64 | LLaMA-style small |
| llama_med | 1 | 32 | 512 | 128 | LLaMA-style medium |

## Performance Targets

### Phase 1 Goals (RTX 5080)

| Metric | Target | Measured |
|--------|--------|----------|
| Correctness | MAE < 1e-5 | ✅ Verified |
| Speedup vs Candle | > 2x | TBD |
| VRAM Reduction | > 50% | TBD |
| seq=2048 support | Working | TBD |

### Profiling Commands

```bash
# Basic profile run
CUBECL_PROFILE=1 cargo bench -p unsloth-rs --features cuda -- flash_attention

# NVIDIA Nsight Systems profiling
nsys profile --output flash_attn_profile \
    cargo bench -p unsloth-rs --features cuda -- flash_attention

# NVIDIA Nsight Compute for detailed kernel analysis
ncu --set full \
    cargo bench -p unsloth-rs --features cuda -- flash_attention
```

## RTX 5080 Specific Configuration

The kernel uses optimized settings for RTX 5080:

```rust
FlashAttentionConfig::for_rtx_5080()
// tile_size: 256
// block_size: 256
// head_dim: 64
// use_vectorized_loads: true
```

### Shared Memory Budget

RTX 5080 shared memory per block: ~48KB (configurable to 100KB)

Our kernel uses:
- Q tile: 256 × 64 × 4 bytes = 64 KB
- K tile: 256 × 64 × 4 bytes = 64 KB  
- V tile: 256 × 64 × 4 bytes = 64 KB
- Scores: 256 × 256 × 4 bytes = 256 KB
- Statistics: 256 × 2 × 4 bytes = 2 KB

**Note**: Phase 1 uses conservative tile sizes. Phase 2 will optimize shared memory usage.

## Interpreting Results

### Criterion Output

```
flash_attention/forward/llama_med_cpu_b1_h32_s512_d128
                        time:   [5.2312 ms 5.3456 ms 5.4601 ms]
                        thrpt:  [0.1832 GiB/s 0.1871 GiB/s 0.1912 GiB/s]
```

- **time**: Wall-clock time per iteration (lower is better)
- **thrpt**: Throughput (higher is better)

### CubeCL Profile Output

When `CUBECL_PROFILE=1` is set:

```
[CubeCL] flash_attention_tile: 1.234 ms
         shared memory: 65536 bytes
         occupancy: 75%
```

## Troubleshooting

### "nvcc not found"
Install CUDA Toolkit and ensure `nvcc` is in PATH.

### "Out of memory"
Reduce batch size or sequence length, or use a smaller tile_size:
```rust
let config = FlashAttentionConfig::default()
    .with_tile_size(64);  // Smaller tiles = less memory
```

### "CubeCL CUDA not available"
The kernel gracefully falls back to Candle implementation on non-CUDA systems.

## Contributing Performance Results

When profiling on new hardware, please report:

1. GPU model and VRAM
2. CUDA version
3. Benchmark configuration
4. Results (time, throughput, memory usage)

Open a GitHub issue or PR with your findings to help tune the kernel for more hardware.
