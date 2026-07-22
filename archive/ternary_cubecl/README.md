# Archived: ternary CubeCL GPU modules (UNS-P2-01)

**Status:** **NON-GOAL / archival research** — not part of the supported public API.

## What this is

Historical CubeCL kernel drafts for ternary bitsliced matmul / attention:

| File | Approx. size | Notes |
|------|--------------|--------|
| `matmul_cubecl.rs` | ~2.8k LOC | Popcount matmul kernel sketches (pre–CubeCL 0.9) |
| `attention_cubecl.rs` | ~1.9k LOC | Ternary attention GPU sketches |

These modules were never exported from `src/kernels/ternary/mod.rs` on default
or `cuda` builds (`// TODO: Re-enable once CubeCL API compatibility is fixed`).

## Why archived

1. **Not maintained** against CubeCL **0.9** / Candle **0.9**.
2. **Not on the release surface** — CPU ternary quant/linear remain under
   `unsloth_rs::kernels::ternary`.
3. **Honesty** — shipping multi-thousand-line unwired GPU stubs as “features”
   overclaims readiness.
4. Closing **UNS-P2-01**: re-enable is **not** a Wave-3 / 1.0.x goal.

## Supported ternary path (still in crate)

- `src/kernels/ternary/{quantize,matmul,linear,attention,types,config,model}.rs`
- CPU bitsliced matmul + experimental compression utilities
- Optional `_ternary_cubecl_todo` Cargo feature remains an empty placeholder
  (does **not** compile these files back in)

## Do not

- Claim GPU ternary speedups from this archive
- Wire these files into `mod.rs` without a dedicated port PR + numerical gate
- Include this directory in crates.io package payloads (see `Cargo.toml` `exclude`)

## Restore path (future, optional)

A future research PR would need to:

1. Port kernels to current CubeCL 0.9 `#[cube(launch)]` APIs
2. Fix host interop (same permanent D2H/H2D limits as FA — see `interop.rs`)
3. Add MAE gates under `--features cuda`
4. Only then re-export under an explicit feature (not default)

Until then: treat this folder as **history**, not product.
