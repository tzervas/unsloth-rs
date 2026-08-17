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

CUDA default is device-resident and **not** FA2 SRAM (`[B,H,S,S]` scores).
Tiled FA is the first real triton-bridge payload — **that** work needs the
5080; see [triton-bridge GPU_HANDOFF](https://github.com/tzervas/triton-bridge-rs/blob/main/docs/GPU_HANDOFF.md).
