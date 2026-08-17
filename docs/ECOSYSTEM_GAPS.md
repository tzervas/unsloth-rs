# unsloth-rs ecosystem gap map

**SoT for sister-crate tracking.**
**Date:** 2026-08-17
**unsloth-rs:** `1.0.3` (this repo)
**Scope:** only `tzervas/*` repos. Read-only elsewhere.

This file is the live tracker. Historical docs (`ISSUE_STATUS.md`, `HANDOFF.md`,
`NEXT_PHASE_PLAN.md`) overclaim 2x / 70% VRAM and are **not** SoT.
Prefer this file + [DEBT.md](../DEBT.md) + [README.md](../README.md).

---

## Role of this crate

`unsloth-rs` is the **kernel leaf**. It must never depend on peft / qlora /
axolotl / rust-ai-core. Sister crates consume it.

It is **not** a product port of Python Unsloth. Triton kernels from Unsloth
are an **algorithm source**, not a language we compile. The Rust path is
CubeCL (or Candle `CustomOp` on `CudaStorage`). Do not copy AGPL MoE kernels.

---

## Intended DAG (correct)

```text
                    candle-core / candle-nn 0.9
                    cubecl / cubecl-cuda 0.9   (optional feature)
                              |
                              v
                       +-------------+
                       | unsloth-rs  |   kernels: attn, RoPE, RMSNorm, SwiGLU, FA
                       |  (leaf)     |   NO peft/qlora/axolotl/core deps
                       +------+------+
                              |
              optional consume |
         +--------------------+--------------------+
         |                    |                    |
         v                    v                    v
    +---------+          +---------+          +-------------+
    | peft-rs |          |qlora-rs |          | axolotl-rs  |
    | 1.1.0   |<---------| 1.1.0   |          | 1.2.0 leaf  |
    +---------+          +---------+          | peft/qlora/ |
         ^                    ^               | unsloth opt |
         |                    |               +------+------+
         +--------------------+----------------------+
                              |
                              v   WRONG TODAY -- invert this
                       +-------------+
                       | rust-ai-core|   should be dep-free traits +
                       | 0.3.4       |   optional re-exports, NOT
                       +-------------+   hard-deps on all 8 leaves
```

**Rule:** arrows point **toward** the consumer. Foundation never imports product.

---

## What unsloth-rs actually has (1.0.3)

| Surface | Status | Notes |
|---------|--------|-------|
| CPU MHA + GQA | **done** | `1/sqrt(head_dim)` multiply convention |
| CPU RoPE / RMSNorm / SwiGLU | **done** | Reference kernels |
| CPU fused RMSNorm+RoPE / fused SwiGLU | **partial** | Named fused; still host Candle ops unless `cuda` |
| CubeCL Flash Attention | **partial** | Real kernels; **host D2H/H2D required** (`interop_requires_host_roundtrip() == true`) |
| GPU numerical gate | **PASS on healthy env** | MAE ~2e-8 on RTX 5080; `BLOCKED:env` without `/dev/nvidia0` |
| CubeCL f16/bf16 | **out of 1.0.x** | `interop_f32_only()` |
| Ternary CPU | **experimental** | Compression ratios only |
| Ternary CubeCL | **archived non-goal** | `archive/ternary_cubecl/` |
| Memory / checkpoint | **estimates only** | No recompute trainer API |
| Mixed-precision helpers | **host utils** | Not a trainer |
| Fused cross-entropy | **missing** | Python Unsloth hot path |
| Fused LoRA (A/B into linear) | **missing** | Belongs here as a kernel; peft owns the adapter math |
| Device-resident CustomOp | **BLOCKED:api** | Candle 0.9 no device-ptr export; CubeCL 0.9 no `CUdeviceptr` import |

**Do not claim 2x / 70% VRAM** while host roundtrips remain.

---

## Sister crates -- how they relate to unsloth-rs

| Crate | Version (tree) | Depends on unsloth? | Uses unsloth in src? | Gap vs unsloth |
|-------|----------------|---------------------|----------------------|----------------|
| [peft-rs](https://github.com/tzervas/peft-rs) | **1.1.0** | No (correct) | No | Fused CUDA kernels quarantined. Should consume unsloth kernels *after* G0, not grow a second kernel tree. |
| [qlora-rs](https://github.com/tzervas/qlora-rs) | **1.1.0** | No (correct); path-deps peft 1.1.0 | No | CubeCL NF4 kernels **not compiled**. Needs fused NF4-dequant-GEMM; that kernel should live in unsloth or a shared kernel crate, not a third copy. |
| [axolotl-rs](https://github.com/tzervas/axolotl-rs) | **1.2.0** | Optional `unsloth-rs = "1.0"` | **No src wiring** (Cargo feature only) | Trainer still uses candle-transformers LLaMA + own RMSNorm. `unsloth` feature is a no-op. Floor still `1.0` not `1.0.3`. GPU E2E blocked by Candle CUDA RMSNorm. |
| [rust-ai-core](https://github.com/tzervas/rust-ai-core) | **0.3.4** | Hard dep `unsloth-rs = "1.0"` | Re-export only (`pub use unsloth_rs::*`) | **Inverted DAG.** Hard-depends on all 8 product crates at **stale** versions (peft 1.0.3, qlora 1.0.5, unsloth 1.0, axolotl 1.1). Duplicate CubeCL interop also host-copies. Docs claim `FlashAttention` type that does not exist (`FusedAttention` does). |
| rust-ai workspace | vendored unsloth **1.0.2** | path member | fork drift | Do not treat workspace copy as SoT. This repo is SoT. |
| trit-vsa / bitnet-quantize / vsa-optim-rs / tritter-accel | 0.x | No | No | Ternary/VSA lane. unsloth ternary CPU is experimental overlap -- do not grow GPU ternary here (archived). |

### Version skew (as of 2026-08-17)

| Consumer pin | Actual sister tree |
|--------------|--------------------|
| rust-ai-core `peft-rs = "1.0"` + hardcoded `"1.0.3"` | peft-rs **1.1.0** |
| rust-ai-core `qlora-rs = "1.0"` + hardcoded `"1.0.5"` | qlora-rs **1.1.0** |
| rust-ai-core / axolotl `unsloth-rs = "1.0"` | unsloth-rs **1.0.3** |
| rust-ai-core `axolotl-rs = "1.1"` | axolotl-rs **1.2.0** |
| axolotl committed peft/qlora `1.0` | sisters **1.1.0** (local override via `scripts/use-local-path-deps.sh`) |

---

## Gap taxonomy

| ID | Name | Where | Blocks |
|----|------|-------|--------|
| **G0** | Device-resident tensors | unsloth `src/kernels/cubecl/interop.rs` + rust-ai-core `src/cubecl/` | Every real speed/VRAM win. Host `to_vec1` / `from_vec`. |
| **G1** | Env / compile pin | `CUDA_COMPUTE_CAP=90` on Blackwell + nvcc <= 9.0 | GPU CI honesty (`BLOCKED:env` vs silent pass) |
| **G2** | Missing Rust kernels | unsloth | Fused CE, fused LoRA linear, packing/padding-free, real checkpoint recompute |
| **G3** | Integration holes | axolotl, rust-ai-core | Feature flags / re-exports without call sites |
| **G4** | Inverted / stale DAG | rust-ai-core | Core hard-deps leaves; version lies |
| **G5** | Duplicated kernel trees | peft archive, qlora `src/kernels/` (unbuilt), tritter-accel inline, hybrid-predict `gpu.rs` | Three FA/RMSNorm/quant impls; none device-resident |
| **G6** | Doc / issue drift | this repo issues 1-19, `ISSUE_STATUS.md` | January tickets say kernels "not implemented"; 1.0.3 has them (CPU + partial CubeCL) |

---

## Triton -> this stack

Python Unsloth wins come from **fused Triton kernels** + monkey-patched autograd,
not from Python itself.

| Python Unsloth (Triton) | Rust home | Fit |
|-------------------------|-----------|-----|
| Flash Attention / online softmax | unsloth CubeCL FA | Algorithm port. Current impl is real but host-roundtrips kill E2E. |
| Fused RoPE + RMSNorm | `fused_rmsnorm_rope` | Exists; make device-resident after G0. |
| Fused SwiGLU / GeGLU | `fused_swiglu` | Same. |
| Fused LoRA (A@B into GEMM) | **new unsloth kernel**; peft stays math | Do **not** reimplement in peft. |
| Chunked / fused cross-entropy | **new unsloth kernel**; axolotl calls it | Biggest easy training-step win after G0. |
| 4-bit dequant-GEMM | qlora codec + unsloth (or shared) GEMM | qlora owns NF4 tables; kernel launch belongs with other fused ops. |
| Padding-free packing | axolotl data plane | Not a kernel crate concern. |
| RL / GRPO / multi-GPU | **non-goal** | Stay single-process honest. |
| AGPL MoE Triton | **do not port source** | Re-derive algorithms only if ever needed. |

**Why CubeCL, not "Triton in Rust":** CubeCL is the Rust-native kernel compiler
(CUDA/Metal/Vulkan). Triton is Python/MLIR. Port **math and tiling**, not the
Triton DSL. The unblock for performance is **one memory plane** (Candle
`CustomOp` on `CudaStorage`, or upstream device-ptr import) -- not more host
CubeCL launches.

Recommended G0 path: **Candle `CustomOp`** on existing `CudaStorage` so FA /
RMSNorm / SwiGLU / CE stay on the Candle allocator. Skip CubeCL `Handle` until
upstream grows external-buffer APIs.

---

## Close-order (priority)

Do not start P3+ until P0/P1 land. Efficiency gains are a **consequence** of
G0 + fused kernels, not extra crates.

### P0 -- make the DAG and memory plane honest (this week)

| # | Work | Repo | Done when |
|---|------|------|-----------|
| P0a | Invert rust-ai-core: dep-free traits + optional features for re-exports | rust-ai-core | `cargo check -p rust-ai-core --no-default-features` needs no peft/qlora/unsloth/axolotl |
| P0b | Bump consumer pins to real versions (unsloth 1.0.3, peft/qlora 1.1.0, axolotl 1.2.0) | rust-ai-core, axolotl-rs | `EcosystemInfo` matches Cargo.toml |
| P0c | Fix rust-ai-core docs: `FusedAttention` not `FlashAttention` | rust-ai-core | examples compile |
| P0d | Prototype Candle `CustomOp` RMSNorm (smallest op) on `CudaStorage` | **unsloth-rs** | one kernel, no `to_vec1`, numerical gate vs CPU |

### P1 -- training-critical kernels (after P0d works)

| # | Work | Repo |
|---|------|------|
| P1a | CustomOp fused CE (chunked, ignore-index) | unsloth-rs |
| P1b | CustomOp fused RMSNorm + RoPE (replace host-roundtrip CubeCL path) | unsloth-rs |
| P1c | CustomOp SwiGLU | unsloth-rs |
| P1d | Keep CubeCL FA as experimental; do not advertise until CustomOp FA or zero-copy | unsloth-rs |

### P2 -- consume the kernels

| # | Work | Repo |
|---|------|------|
| P2a | Wire `axolotl --features unsloth` to call unsloth RMSNorm / RoPE / SwiGLU / CE | axolotl-rs |
| P2b | Replace axolotl CUDA RMSNorm workaround with unsloth CustomOp | axolotl-rs |
| P2c | peft: delete/ignore archived kernels; add optional `unsloth` feature that fuses LoRA residual via unsloth kernel | peft-rs |
| P2d | qlora: NF4 dequant-GEMM using unsloth launch + qlora tables | qlora-rs |

### P3 -- E2E proof

| # | Work | Repo |
|---|------|------|
| P3a | TinyLlama LoRA GPU step: measure vs CPU Candle (publish numbers or don't claim) | axolotl-rs + unsloth |
| P3b | QLoRA 7B attempt only after P2d + G0 | axolotl + qlora |
| P3c | Dedup tritter-accel / hybrid-predict GPU stubs -- they must call unsloth, not reimplement | those repos |

### P4 -- explicit non-goals (close tickets, don't build)

- Ternary CubeCL in this crate (already archived)
- Python Unsloth product parity (model zoo, RL, multi-GPU)
- Speed/VRAM marketing without a published bench that includes D2H/H2D
- rust-ai-core as a required megacrate

---

## Tracking issues

Fill in numbers as they open. Do not let January `ISSUE_STATUS.md` override this file.

| Tracker | Repo | Issue |
|---------|------|-------|
| This file | unsloth-rs | -- |
| Master ecosystem tracker | unsloth-rs | TBD |
| Invert DAG + version lockstep | rust-ai-core | TBD |
| Wire `unsloth` feature | axolotl-rs | TBD |
| Consume kernels, don't fork them | peft-rs | TBD |
| NF4-GEMM via unsloth | qlora-rs | TBD |
