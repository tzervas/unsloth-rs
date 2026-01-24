# unsloth-rs - GPU-Optimized Transformer Kernels

## Overview

High-performance transformer building blocks using CubeCL for cross-platform GPU compute. Inspired by Unsloth's Python optimizations.

## Architecture

```
src/
├── lib.rs           # Public API exports
├── kernels/         # GPU kernel implementations
│   ├── mod.rs       # Kernel module exports
│   ├── attention.rs # Multi-head attention (GQA support)
│   ├── rope.rs      # Rotary position embeddings
│   ├── rmsnorm.rs   # RMS normalization
│   ├── swiglu.rs    # SwiGLU activation
│   └── ternary/     # Ternary quantization kernels
├── memory.rs        # Memory estimation and management
├── training.rs      # Training utilities (mixed precision)
└── error.rs         # Error types

tests/
├── integration.rs   # CPU/GPU correctness tests
├── helpers.rs       # Test utilities
└── gpu/             # GPU-specific tests (ignored without CUDA)
```

## Key Components

### Attention (`kernels/attention.rs`)
```rust
pub struct MultiHeadAttention {
    num_heads: usize,
    num_kv_heads: usize,  // For GQA
    head_dim: usize,
    scale: f32,
}

// Supports grouped-query attention (GQA)
// num_kv_heads < num_heads for memory efficiency
```

### RoPE (`kernels/rope.rs`)
```rust
pub fn apply_rotary_embedding(
    q: &Tensor,
    k: &Tensor,
    positions: &Tensor,
    rope_theta: f32,
) -> Result<(Tensor, Tensor)>
```

### Memory Estimation (`memory.rs`)
```rust
pub fn estimate_training_memory(
    model_params: usize,
    batch_size: usize,
    seq_len: usize,
    dtype: DType,
) -> MemoryEstimate
```

## GPU Compute Stack

Uses CubeCL (v0.8.1) for cross-platform GPU kernels:
```rust
use cubecl::prelude::*;
use cubecl_cuda::CudaRuntime;  // When cuda feature enabled
```

### Feature Flags
- `default`: CPU-only with Candle
- `cuda`: Enable CUDA backend via CubeCL

## Development Commands

```bash
# Check (CPU only)
cargo check -p unsloth-rs

# Check with CUDA
cargo check -p unsloth-rs --features cuda

# Test CPU
cargo test -p unsloth-rs

# Test GPU (requires CUDA)
cargo test -p unsloth-rs --features cuda -- --ignored

# GPU tests only
cargo test -p unsloth-rs --features cuda gpu:: -- --ignored

# Benchmarks
cargo bench -p unsloth-rs

# With CUDA benchmarks
cargo bench -p unsloth-rs --features cuda
```

## Critical Code Paths

### Attention Forward
```rust
// Q @ K^T / sqrt(d) -> softmax -> @ V
// Must handle GQA head broadcasting efficiently
```

### RoPE Application
```rust
// In-place rotation of Q and K tensors
// Frequency computation must match LLaMA implementation
```

### Memory Management
```rust
// Gradient checkpointing support for large models
// Activation recomputation trade-off
```

## Testing Strategy

- Unit tests: Kernel math correctness against reference
- Integration: Full attention block forward/backward
- GPU tests: CUDA kernel equivalence to CPU reference
- Property tests: Numerical stability across dtypes

## 1.0 Checklist

- [x] Multi-head attention with GQA
- [x] RoPE implementation
- [x] RMSNorm
- [x] SwiGLU activation
- [x] Memory estimation
- [x] Ternary quantization (experimental)
- [x] 160 passing tests
- [ ] FlashAttention-style memory optimization
- [ ] Fused attention kernel
- [ ] KV cache management
- [ ] Benchmark suite vs PyTorch/Triton
- [x] Examples directory
- [ ] 100% doc coverage

## Common Issues

### "CubeCL feature not found"
The `_ternary_cubecl_todo` cfg warning is expected - future kernel placeholder.

### GPU test failures
Ensure CUDA toolkit installed and `nvcc` in PATH:
```bash
nvcc --version
```

### Memory estimation off
Memory estimation is approximate. Actual usage depends on:
- Candle's memory allocator
- CUDA memory fragmentation
- Gradient accumulation strategy

## Performance Targets

| Operation | Target Throughput | Status |
|-----------|-------------------|--------|
| Attention (batch=1, seq=2048) | > 1 TFLOPS | In progress |
| RoPE | Negligible overhead | Done |
| RMSNorm | Fused with attention | Planned |
| SwiGLU | Fused MLP | Planned |

## Integration with Other Crates

This crate is standalone but designed to integrate with:
- **axolotl-rs**: Via `unsloth` feature for optimized training
- Future: May provide optimized kernels for qlora-rs dequantization
