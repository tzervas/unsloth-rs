# Triton, CubeCL, and this crate

**Decision (2026-08-17, updated):** Triton-the-compiler does **not** live in
unsloth-rs. Load/launch of *foreign* PTX/CUBIN lives in
[triton-bridge-rs](https://github.com/tzervas/triton-bridge-rs) **0.2**.

| Need | Where |
|------|--------|
| Transformer ops on Candle tensors | **this crate** `kernels::custom_op` |
| Compile / load / launch *foreign* Triton or PTX | **triton-bridge 0.2** (`--features cuda`) |

## Why Triton FFI first (and what that actually means)

Python Unsloth is Triton kernels + a thin Python trainer. The Rust port:

1. **Now:** keep CustomOp (RMS / RoPE / SwiGLU / CE) as the ergonomic default.
2. **Now (0.2):** triton-bridge can load precompiled PTX/CUBIN on a real device
   so we can consume Unsloth’s **Flash Attention** tiling without CubeCL
   host copies and without embedding CPython in the training process.
3. **Later:** a Rust-native tiled FA (CustomOp or a small DSL) that matches
   that ABI, then drop the CUBIN.

We do **not** FFI Triton for ops we already own. Catalog:
`triton_bridge::KERNEL_CATALOG`.

## Hook (no Cargo dep yet)

[`kernels::triton_bridge`](../src/kernels/triton_bridge.rs):

- `triton_bridge_ready()` → `false` in this tree
- `should_dispatch_triton_bridge()` → `false`
- feature `triton-bridge` is a **cfg reservation** only

Do **not** add `triton-bridge = "0.2"` until a 5080 run has launched FA
on a device pointer (their GPU_HANDOFF). A hard dep that is `FAIL_ENV` on
every CPU publish is noise.

## Host copies

CubeCL Flash Attention **still** `to_vec1`s (`interop_requires_host_roundtrip()`).
It is **no longer the default**. `flash_attention_cubecl` uses
[`attention_device`](../src/kernels/custom_op/attention.rs) unless
`UNSLOTH_CUBECL_FA` is set.

CUDA default is device-resident and **not** FA2 SRAM (`[B,H,S,S]` scores).
Tiled FA is the first real triton-bridge payload.
