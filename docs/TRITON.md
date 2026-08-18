# Triton, CubeCL, and this crate

**Next Grok Build session:**
[crate-track `GROK_BUILD.md`](https://github.com/tzervas/crate-track/blob/main/GROK_BUILD.md).

**Decision (2026-08-17):** Triton-the-compiler does **not** live in unsloth-rs.

## Ergonomics bar (non-negotiable)

Call sites must stay as easy as a Triton JIT launch / Unsloth kernel import,
**regardless of backend** (CustomOp, Candle GEMM, later device-pointer FA).

- Public names live in [`unsloth_rs::ops`](../src/ops.rs): `rmsnorm`,
  `layernorm`, `rope`, `rope_with_ids`, `swiglu`, `geglu`, `attention`,
  `attention_softcap`, `cross_entropy`, `fused_linear_ce`.
- One function, tensors in, tensor out. No NVRTC, CubeCL handles, or
  `LaunchConfig` at the call site.
- Device dispatch is inside the function. CPU and CUDA share the name.
- Missing GPU / toolkit is `FAIL_ENV`, not a different API.
- Tiled FA, when it exists, must keep `attention(q,k,v,scale,mask,causal)`.

Implementation may be NVRTC, Candle GEMM, or triton-bridge. The names do not
change. Do not make users pick `*_custom_op` vs `*_cubecl` vs `*_device`.

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

CUDA default (no extra mask, head dim ≤ 128) is an **owned** SRAM-tiled
Flash-style kernel (NVRTC). That is not Unsloth `flash_attention_2` PTX.
triton-bridge Job C is still the *foreign* Unsloth JIT payload; it remains
FAIL_ENV until Unsloth ships one. See
[triton-bridge GPU_HANDOFF](https://github.com/tzervas/triton-bridge-rs/blob/main/docs/GPU_HANDOFF.md).

## Dispatch (non-CustomOp policy)

| Situation | Path | Why |
|-----------|------|-----|
| No extra mask, head dim ≤ 128 | CustomOp tiled SRAM FA | Owned NVRTC; not Unsloth PTX |
| No extra mask, wider head | CustomOp online-softmax | HBM-streaming fallback |
| Extra attention mask | Candle GEMM + softmax | Vendor GEMM; scores materialize |
| `triton_bridge_ready()` | Foreign Unsloth FA PTX | Only after Job C launches |
| `UNSLOTH_CUBECL_FA` | CubeCL FA | Opt-in only; still host D2H |
| Fused linear+CE | CustomOp (CPU) or vocab-tile GEMM | No `[N, V]`; not CubeCL |
| QKV / SwiGLU projections | Candle / cuBLAS GEMM | Do not NVRTC-fuse large GEMMs |

Do **not** grow a third CubeCL kernel stack while
`interop_requires_host_roundtrip()` is true. Do **not** add a
`triton-bridge` Cargo dep while `bridge_ready() == false`.

## Host CPU (akula-prime i7-14700K)

| Have | Do not have |
|------|-------------|
| AVX2, FMA, AVX-VNNI | AVX-512, AMX (those are Xeon, not this desktop SKU) |
| P-cores `0-15` (8×2) | — |
| E-cores `16-27` (12×1) | — |

CustomOp CPU dots use AVX2+FMA when `is_x86_feature_detected`. Fused-CE
rows parallelize with `std::thread` (no rayon / MKL dep). The crate does
**not** pin affinity — `taskset -c 0-15` for P-only if the compositor
should keep the E-cores. Large GEMMs stay Candle (already AVX2). VNNI is
INT8; unused until we have an INT8 CustomOp.

## iGPU (UHD 770) — DIY, not a product path

Hardware is present: `00:02.0` Intel UHD 770, DRM `card0` / `renderD128`
(`vendor=0x8086`). Mesa `intel_icd.json` is installed. **This is not a
supported unsloth-rs backend.**

Measured on this box (2026-08-17): `vulkaninfo` lists the 5080 and
llvmpipe only — ANV does not show the iGPU in the live instance. `clinfo
-l` is empty. No oneAPI / Level Zero. Candle has no Intel device.

**Backlog.** Do not add an `igpu` feature this cycle. A later DIY probe
would force the Intel Vulkan ICD or install compute-runtime/OpenCL, off
the 5080. Expect FAIL_ENV, no speed claims, no training story.
