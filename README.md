# unsloth-rs

<!-- FLEET-BADGES:BEGIN -->
[![CI](https://github.com/tzervas/unsloth-rs/actions/workflows/fleet-ci.yml/badge.svg?branch=main)](https://github.com/tzervas/unsloth-rs/actions/workflows/fleet-ci.yml?query=branch%3Amain)
[![Security](https://github.com/tzervas/unsloth-rs/actions/workflows/fleet-security.yml/badge.svg?branch=main)](https://github.com/tzervas/unsloth-rs/actions/workflows/fleet-security.yml?query=branch%3Amain)
<!-- FLEET-BADGES:END -->

Candle/CubeCL **transformer kernel building blocks** for LLM inference experiments.

[![Crates.io](https://img.shields.io/crates/v/unsloth-rs.svg)](https://crates.io/crates/unsloth-rs)
[![Documentation](https://docs.rs/unsloth-rs/badge.svg)](https://docs.rs/unsloth-rs)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Not a product port of [Unsloth](https://github.com/unslothai/unsloth).**
> This crate does **not** provide LoRA/QLoRA trainers, a model zoo, HuggingFace
> fine-tune pipelines, or proven Unsloth-style training speed/VRAM gains.
> Do **not** claim “2× faster” or “~70% less VRAM” for this crate without
> published measurements from this codebase.

## What this crate is

`unsloth-rs` is a MIT Rust library of **common transformer ops** built on
[Candle](https://github.com/huggingface/candle), with optional CubeCL CUDA
kernels:

| Building block | CPU (Candle) | CUDA feature |
|----------------|--------------|--------------|
| Multi-head attention + GQA | ✅ | Candle CUDA + optional Flash path |
| RoPE | ✅ | Elementwise CubeCL (partial) |
| RMSNorm | ✅ | Elementwise CubeCL (partial) |
| SwiGLU | ✅ | Elementwise CubeCL + CPU fallback |
| Memory / checkpoint **estimates** | ✅ (math only) | n/a |
| Ternary quant / linear (CPU) | ✅ experimental | GPU CubeCL **archived non-goal** |
| Mixed-precision helpers | ✅ config + scale utils | not a trainer |
| LoRA / QLoRA / SFT trainer | ❌ | use peft-rs / qlora-rs / axolotl-rs |

## Status (honest)

**Version:** `1.0.3` (see `Cargo.toml`). Semver 1.x means the **public CPU
kernel APIs** are intended to be usable; GPU paths and training utilities are
still incomplete.

### Solid today

- Multi-head attention (CPU reference; correct `1/√head_dim` scaling)
- RoPE, RMSNorm, SwiGLU on CPU
- Memory estimation helpers
- Default-feature CPU test suite (unit + integration)

### Partial / experimental

- Flash Attention via CubeCL (`cuda` feature): real kernels exist; **host D2H/H2D
  interop is a permanent limitation** with Candle 0.9 + CubeCL 0.9 public APIs
  (`interop_requires_host_roundtrip()` → true). **No end-to-end speedup claims.**
- GPU numerical equivalence gate: runs under `--features cuda` (not default CI);
  needs `/dev/nvidia0` (see [DEBT.md](DEBT.md)); **BLOCKED:env** without full device nodes
- Ternary quantization experiments (CPU compression ratios only; GPU ternary archived)
- CubeCL / Flash path **f32-only** (`interop_f32_only()`); host mixed-precision helpers only
- Mixed-precision **utilities** (no end-to-end trainer); checkpoint **estimates**
  only (no recompute training API)

### Explicit non-goals (for this crate)

- Unsloth product parity (model matrix, packing, RL/GRPO, multi-GPU train)
- Claiming training speedups or VRAM savings without evidence
- Shipping a fine-tuning CLI (that belongs in orchestration crates)
- Ternary CubeCL GPU kernels (archived under `archive/ternary_cubecl/`)
- CubeCL f16/bf16 kernels in 1.0.x (explicit f32-only interop scope)

**Docs:** [CHANGELOG.md](CHANGELOG.md) · [ROADMAP.md](ROADMAP.md) · [DEBT.md](DEBT.md) ·
[GPU_SETUP.md](GPU_SETUP.md) · [PUBLISHING.md](PUBLISHING.md) ·
[docs/DEPENDENCIES.md](docs/DEPENDENCIES.md) (no peft/qlora/axolotl deps; no cycles).

Residual risk and CUDA environment contract (`CUDA_COMPUTE_CAP`, `FAIL_ENV`) are in
[DEBT.md](DEBT.md) and [GPU_SETUP.md](GPU_SETUP.md).
## Installation

```toml
[dependencies]
unsloth-rs = "1.0.3"
```

CUDA (optional; requires toolkit + device — see [GPU_SETUP.md](GPU_SETUP.md)):

```toml
[dependencies]
unsloth-rs = { version = "1.0.3", features = ["cuda"] }
```

On hosts where the default compute capability pin fails `nvcc` (e.g. CC 12.0
reported but toolkit only builds ≤ 9.0):

```bash
CUDA_COMPUTE_CAP=90 cargo check --features cuda
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
        num_kv_heads: Some(4), // GQA support
        ..Default::default()
    };

    let attention = FusedAttention::new(config, &device)?;
    let hidden_states = Tensor::randn(0.0f32, 1.0, (1, 128, 768), &device)?;
    let output = attention.forward(&hidden_states, None, None)?;

    Ok(())
}
```

### Memory estimation

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

    println!("Estimated activation memory (model): {} GB", mem_bytes as f64 / 1e9);
}
```

## Benchmarks

```bash
cargo bench
```

Benchmarks measure **CPU** kernel performance by default. GPU benches need
`features = ["cuda"]` and a working device (not exercised as green in default CI).

**Do not** publish CubeCL Flash Attention as 2× faster than Candle while host
interop round-trips remain (`DEBT.md` / UNS-P1-01). Kernel-only microbenchmarks
that ignore D2H/H2D are not product evidence.

## Development docs

- **[ROADMAP.md](ROADMAP.md)** — strategic plan (single roadmap file; do not add a
  case-colliding `roadmap.md` — crates.io packaging fails on case-insensitive FS)
- **[TASKS.md](TASKS.md)** — task list
- **[PUBLISHING.md](PUBLISHING.md)** — packaging notes for crates.io
- **[GPU_SETUP.md](GPU_SETUP.md)** — CUDA toolkit / `CUDA_COMPUTE_CAP` contract
- **[DEBT.md](DEBT.md)** — residual technical debt and env blocks

## Packaging note

Only **`ROADMAP.md`** is kept. A historical lowercase `roadmap.md` was removed
because crates.io rejects tarballs with path case collisions
(`roadmap.md` vs `ROADMAP.md`).

Verify before any release:

```bash
cargo package --allow-dirty --list
# must not error with "Duplicate path conflicts"
```

## Contributing

Contributions welcome, especially:

- Device-side Candle↔CubeCL handoff (needs upstream APIs or Candle `CustomOp` path)
- GPU numerical gate runs with published MAE when `/dev/nvidia0` is healthy
- Additional transformer ops (e.g. fused CE) with CPU references first

## License

Licensed under the MIT License. See [LICENSE](LICENSE) for details.
