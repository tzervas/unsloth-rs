# Triton, CubeCL, and this crate

**Decision (2026-08-17):** Triton-the-compiler does **not** live in unsloth-rs.

| Need | Where |
|------|--------|
| Transformer ops on Candle tensors | **this crate** `kernels::custom_op` |
| Compile / load / launch *foreign* Triton or PTX | **[triton-bridge-rs](https://github.com/tzervas/triton-bridge-rs)** (stub; `bridge_ready() == false`) |

Python Unsloth kernels are the **algorithm source**. We rewrite small ones as
`CustomOp*` (RMSNorm, RoPE, SwiGLU, CE, online attention). Large tiled FA
may later load a precompiled CUBIN via triton-bridge-rs Phase 1 — still no
CPython in the training process.

## Host copies

CubeCL Flash Attention **still** `to_vec1`s (`interop_requires_host_roundtrip()`).
It is **no longer the default**. `flash_attention_cubecl` uses
[`attention_device`](../src/kernels/custom_op/attention.rs) unless
`UNSLOTH_CUBECL_FA` is set.

That default is device-resident. It is **not** FA2 SRAM tiling on CUDA
(scores are `[B,H,S,S]`). Tiled FA is the first real triton-bridge payload.

See [triton-bridge-rs/docs/PYTHON_UNSLOTH_KERNEL_MAP.md](https://github.com/tzervas/triton-bridge-rs/blob/main/docs/PYTHON_UNSLOTH_KERNEL_MAP.md).
