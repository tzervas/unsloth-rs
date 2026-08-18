# Triton, CubeCL, and this crate

**Decision (2026-08-17):** Triton-the-compiler does **not** live in unsloth-rs.

| Need | Where |
|------|--------|
| Transformer ops on Candle tensors | **this crate** `kernels::custom_op` |
| Compile / load / launch *foreign* Triton or PTX | **[triton-bridge-rs v0.1.0](https://github.com/tzervas/triton-bridge-rs/releases/tag/v0.1.0)** (`bridge_ready() == false`) |

## Hook (no Cargo dep)

[`kernels::triton_bridge`](../src/kernels/triton_bridge.rs):

- `triton_bridge_ready()` → `false`
- `should_dispatch_triton_bridge()` → `false`
- feature `triton-bridge` is a **cfg reservation** only

Do **not** add `triton-bridge = "0.1"` until that crate can launch on a
device pointer (their #3). A hard dep on a `NotReady` crate would just
noise crates.io publishes.

## Host copies

CubeCL Flash Attention **still** `to_vec1`s (`interop_requires_host_roundtrip()`).
It is **no longer the default**. `flash_attention_cubecl` uses
[`attention_device`](../src/kernels/custom_op/attention.rs) unless
`UNSLOTH_CUBECL_FA` is set.

CUDA default is device-resident **online-softmax** (no `[B,H,S,S]`). That is
**not** FA2 SRAM. Tiled FA is the first real triton-bridge payload — **that**
work needs the 5080; see
[triton-bridge GPU_HANDOFF](https://github.com/tzervas/triton-bridge-rs/blob/main/docs/GPU_HANDOFF.md).

## Dispatch (non-CustomOp policy)

| Situation | Path | Why |
|-----------|------|-----|
| No extra mask | CustomOp online-softmax | Device-resident; extra `O(S·D)` |
| Extra attention mask | Candle GEMM + softmax | Vendor GEMM; scores materialize |
| `triton_bridge_ready()` | Tiled FA on a device pointer | Only after Job C launches |
| `UNSLOTH_CUBECL_FA` | CubeCL FA | Opt-in only; still host D2H |
| Fused linear+CE | CustomOp (CPU) or vocab-tile GEMM | No `[N, V]`; not CubeCL |
| QKV / SwiGLU projections | Candle / cuBLAS GEMM | Do not NVRTC-fuse large GEMMs |

Do **not** grow a third CubeCL kernel stack while
`interop_requires_host_roundtrip()` is true. Do **not** add a
`triton-bridge` Cargo dep while `bridge_ready() == false`.
