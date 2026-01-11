# unsloth-rs

Rust implementations of transformer building blocks for LLM inference and fine-tuning.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

`unsloth-rs` provides Rust implementations of common transformer operations built on the [Candle](https://github.com/huggingface/candle) ML framework:

- Multi-head attention with grouped-query attention (GQA) support
- Rotary position embeddings (RoPE)
- RMS normalization
- SwiGLU activation

## Status

**⚠️ Early Development** - This project is in early development. Current implementations are CPU reference implementations with GPU dispatch that uses Candle's CUDA backend.

### Implemented
- ✅ Multi-head attention (CPU reference, Candle CUDA backend)
- ✅ Rotary position embeddings (RoPE)
- ✅ RMS normalization
- ✅ SwiGLU activation
- ✅ Memory estimation utilities
- ✅ Ternary quantization (5-15x compression achieved)
- ✅ Mixed precision training utilities (FP32/FP16/BF16)
- ✅ Benchmarking suite (CPU)
- ✅ 148 passing tests (100% pass rate)

### In Progress
- 🚧 Flash Attention CubeCL GPU kernel (Phase 1 complete, Phase 2 ready for RTX 5080 validation)
- 🚧 Ternary GPU kernels (Phase 2-4 implemented, awaiting GPU profiling)
- 🚧 CI/CD pipeline setup

### Planned
- ⏳ Gradient checkpointing (configuration exists, implementation planned)
- ⏳ GPU performance validation on RTX 5080/3090 Ti
- ⏳ RoPE, RMSNorm, SwiGLU GPU kernels
- ⏳ Advanced sparsity optimizations
- ⏳ Multi-GPU support

## Installation

```toml
[dependencies]
unsloth-rs = "0.1"
```

For CUDA support (uses Candle's CUDA backend):

```toml
[dependencies]
unsloth-rs = { version = "0.1", features = ["cuda"] }
```

## Usage

### Attention

```rust
use unsloth_rs::kernels::{FusedAttention, FusedAttentionConfig};
use candle_core::{Device, Tensor};

fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;
    
    let config = FusedAttentionConfig {
        hidden_size: 768,
        num_heads: 12,
        head_dim: 64,
        num_kv_heads: Some(4),  // GQA support
        ..Default::default()
    };
    
    let attention = FusedAttention::new(config, &device)?;
    
    // Create random input tensor: randn(mean, std_dev, shape, device)
    // 0.0f32 is Rust syntax for a 32-bit float literal with value 0.0
    let hidden_states = Tensor::randn(0.0f32, 1.0, (1, 128, 768), &device)?;
    let output = attention.forward(&hidden_states, None, None)?;
    
    Ok(())
}
```

### Memory Estimation

```rust
use unsloth_rs::memory::{estimate_forward_memory, CheckpointConfig};

fn main() {
    let checkpoint = CheckpointConfig {
        enabled: true,
        checkpoint_every: 2,
    };
    
    let mem_bytes = estimate_forward_memory(
        4,     // batch_size
        2048,  // seq_len
        4096,  // hidden_size
        32,    // num_layers
        &checkpoint,
    );
    
    println!("Estimated memory: {} GB", mem_bytes as f64 / 1e9);
}
```

## Benchmarks

Run benchmarks with:

```bash
cargo bench
```

Benchmarks test CPU performance across various configurations. GPU benchmarks require the `cuda` feature.

## Development Roadmap

For detailed development plans and task breakdowns, see:

- **[ROADMAP.md](ROADMAP.md)** - Strategic development plan with phases and timelines
- **[TASKS.md](TASKS.md)** - Actionable task list with priorities and estimates
- **[SUMMARY.md](SUMMARY.md)** - Project review summary and execution guide

## Contributing

Contributions are welcome, particularly:
- GPU kernel implementations using CubeCL
- Performance optimizations
- Additional transformer operations

See [TASKS.md](TASKS.md) for specific tasks that need implementation.

## License

Licensed under the MIT License. See [LICENSE](LICENSE) for details.
