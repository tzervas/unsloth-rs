# unsloth-rs ecosystem gap map

**SoT for sister-crate tracking.**
**Date:** 2026-08-19
**unsloth-rs:** `1.0.4` (this repo, after #94 / #96)
**Scope:** only `tzervas/*` repos. Read-only elsewhere.

Live tracker. Historical docs (`ISSUE_STATUS.md`, January tickets) overclaim
2× / 70% VRAM and are **not** SoT. Prefer this file + [DEBT.md](../DEBT.md) +
[README.md](../README.md) + [VERSIONING.md](VERSIONING.md).

## Role

`unsloth-rs` is the **kernel leaf**. It must never depend on peft / qlora /
axolotl / rust-ai-core. Sister crates consume it. It is **not** a product port
of Python Unsloth. Triton kernels are an algorithm source. Do not copy AGPL
MoE kernels.

## What 1.0.4 actually has

| Surface | Status | Notes |
|---------|--------|-------|
| `unsloth_rs::ops` | **done** | One name, tensors in/out: rmsnorm, layernorm, rope, rope_with_ids, swiglu, geglu, attention, attention_window, attention_softcap, cross_entropy, fused_linear_ce |
| CPU MHA + GQA | **done** | `1/sqrt(head_dim)` |
| CustomOp RMSNorm / RoPE / SwiGLU / GeGLU / LayerNorm / CE | **done** | Device-resident `CpuStorage` / `CudaStorage`. No CubeCL `to_vec1` |
| CustomOp CUDA attention | **done (owned)** | SRAM-tiled FA dim≤128, no extra mask. Wider heads: HBM-streaming online. Extra mask → GEMM |
| f16 CustomOp | **partial** | rmsnorm / swiglu / attention f16 I/O, float acc. bf16 open |
| Fused linear+CE | **CPU + CUDA tiles** | Avoids `[N,V]` logits. 5080 tile smoke vs CPU |
| CubeCL FA | **opt-in** | `UNSLOTH_CUBECL_FA=1`. Still host D2H/H2D |
| Job C (Unsloth FA PTX) | **FAIL_ENV** | No Apache-2.0 FA Triton JIT in Unsloth 2026.8.18 |
| Ternary CPU | **experimental** | `prune_below_threshold` still a stub on trunk |
| Ternary CubeCL | **archived non-goal** | |
| LoRA / QLoRA / trainer | **non-goal** | peft / qlora / axolotl |
| `cubecl` crate | **optional** | Default CPU builds do not pull the cubecl graph (#96) |

**Do not claim 2× / 70% VRAM.** Tiled attn s512 is still slower than torch SDPA
on the 5080 compare harness.

## Sister crates

| Crate | Depends on unsloth? | Gap |
|-------|---------------------|-----|
| peft-rs | No (correct) | Consume unsloth kernels after G0; do not grow a second kernel tree |
| qlora-rs | No (correct) | NF4-dequant-GEMM should share a launch path, not a third copy |
| axolotl-rs | Optional `unsloth-rs` | Feature flag is still mostly unwired |
| rust-ai-core | Hard dep (wrong) | Invert DAG; optional re-exports |
| triton-bridge-rs | No Cargo dep (correct) | Job C device-pointer launch still blocked |

## Close-order (remaining)

P0d CustomOp RMSNorm and P1 CE / RoPE / SwiGLU / attention **landed in 1.0.4**.

| # | Work | Repo |
|---|------|------|
| P2a | Wire `axolotl --features unsloth` to `ops::*` | axolotl-rs |
| P2b | Invert rust-ai-core hard deps | rust-ai-core |
| P2c | Tag `v1.0.4` + decide crates.io publish | this repo (`release` workflow) |
| C-COMPARE-NEWOPS | Compare harness for geglu / layernorm / window / softcap | this repo (crate-track, after promote) |
| C-UNS-SHAPES / TILE-OCC / SM120 | Occupancy / SM 12.0 | this repo |
| G-UNS-09 | peft/qlora stay closed | — |
| G-UNS-10 | iGPU backlog | — |

## Tracking

| Tracker | Issue |
|---------|-------|
| This file | PR follow-up to #86 |
| Invert DAG | [rust-ai-core#9](https://github.com/tzervas/rust-ai-core/issues/9) |
| Wire unsloth feature | [axolotl-rs#69](https://github.com/tzervas/axolotl-rs/issues/69) |
