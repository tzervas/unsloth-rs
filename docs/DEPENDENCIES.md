# unsloth-rs dependency graph

## DAG (no cycles)

```text
candle-core, candle-nn, cubecl (optional), …
              │
              ▼
     triton-bridge-rs   (sibling leaf; NOT a Cargo dep in 1.0.x)
              │           hook: kernels::triton_bridge (always !ready)
              ▼
         ┌──────────┐
         │unsloth-rs│  ← foundation kernels; NO peft/qlora/axolotl deps
         └────┬─────┘
              │ optional consumer
              ▼
         axolotl-rs (feature unsloth)
```

**unsloth-rs must never depend on peft-rs, qlora-rs, or axolotl-rs.**
Compose PEFT training via those crates, not this one.

Live gap map + close-order: [ECOSYSTEM_GAPS.md](ECOSYSTEM_GAPS.md).
`cubecl` is optional; default/CPU builds do not pull that graph.

## Features

| Feature | Effect |
|---------|--------|
| *(default)* | CPU Candle kernels |
| `cuda` | CubeCL + candle CUDA paths; see [GPU_SETUP.md](../GPU_SETUP.md) |
| `triton-bridge` | Reserved cfg. **No crate linked.** [`triton_bridge_ready`](../src/kernels/triton_bridge.rs) stays `false` until [triton-bridge-rs](https://github.com/tzervas/triton-bridge-rs) Phase 1 (`v0.1.0` is a contract only). |

## Runtime notes (docs, not deps)

- Pin `CUDA_COMPUTE_CAP=90` when host reports CC 12.0 but nvcc max is 9.0  
- WSL: prefer `LD_LIBRARY_PATH=/usr/lib/wsl/lib` for a real `libcuda`  
- Host D2H/H2D interop is a permanent limitation with public Candle 0.9 / CubeCL 0.9 APIs  

## Package surface

`archive/` (ternary CubeCL drafts) is **excluded** from the crates.io package.
Only `ROADMAP.md` (not a lowercase duplicate) ships for packaging safety.
