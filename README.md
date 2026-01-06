# unsloth-rs

Rust implementations of transformer building blocks for LLM inference and fine-tuning.

[![License](https://img.shields.io/crates/l/unsloth-rs.svg)](LICENSE-MIT)

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
- ✅ Benchmarking suite

### Planned
- 🚧 Fused CubeCL GPU kernels
- 🚧 Gradient checkpointing
- 🚧 Mixed precision support
- 🚧 Flash Attention algorithm

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

## Contributing

Contributions are welcome, particularly:
- GPU kernel implementations using CubeCL
- Performance optimizations
- Additional transformer operations

## License

Licensed under MIT or Apache-2.0 at your option.
