# P1 CustomOp plan (unsloth-rs)

**Status:** implemented 2026-08-17 on `feat/p1-customop-kernels`; RoPE `position_ids` + CPU fused linear+CE on `feat/p1-rope-ids-fused-ce`

**Depends on:** P0d CustomOp RMSNorm (`feat/p0d-customop-rmsnorm`, [PR #88](https://github.com/tzervas/unsloth-rs/pull/88))
**Parent tracker:** [#87](https://github.com/tzervas/unsloth-rs/issues/87)

This is the planning pass. Implementation follows this document; it does not
invent extra crates, GEMM fusion, or CubeCL work.

---

## Goal

Three more **device-resident** Candle `CustomOp*` kernels so a trainer can
stay on `CpuStorage` / `CudaStorage` and never `to_vec1` into CubeCL.

| Op | Why it is P1 | What it is not |
|----|----------------|----------------|
| **Chunked CE** | Does not exist today. Biggest training VRAM win after Flash. axolotl will call this. | Not fused `lm_head @ W + CE` (that is a later axolotl/P2 cut). |
| **RoPE apply** | Current path is `narrow` + 4 broadcasts + `cat` (temps). CubeCL `rope()` D2H. | Not indexed packing RoPE as a separate product. Sequential cache is v1; position gather if cheap. |
| **SwiGLU `silu⊙up`** | Fuse the elementwise after the two GEMMs. CubeCL `swiglu()` D2H. | Not fusing the three GEMMs (cuBLAS already wins there). |

RMSNorm CustomOp is already the pattern. **Do not** rewrite CubeCL FA this pass.

## Constraints (non-negotiable)

- f32 only (`custom_op_f32_only()` stays true).
- No host round-trip on these paths (`custom_op_device_resident()` stays true).
- CubeCL launchers remain in tree but are **not** the default for RoPE/SwiGLU/RMS
  once a CustomOp exists (CubeCL is slower today because of D2H/H2D).
- No 2× / VRAM marketing numbers.
- CPU tests must match a broadcast / `log_softmax` reference (MAE tight).
- CUDA kernels: NVRTC + `CudaDevice::get_or_load_custom_func`, same as RMSNorm.
- Backward may be unfused Candle ops **except CE**, where a naive
  `log_softmax` would allocate `[N, V]` softmax — CE bwd must be chunked.

## P1a — Chunked cross-entropy

### API

```rust
pub fn chunked_cross_entropy(
    logits: &Tensor,   // f32, [..., vocab]
    targets: &Tensor,  // i64, [...]
    ignore_index: i64, // typically -100
    chunk_size: usize, // vocab chunk, default 4096
) -> Result<Tensor>   // scalar mean over valid tokens
```

`CustomOp2`: logits × targets. Reduction = mean of non-ignored tokens.
All-ignore → `0`.

### Algorithm

Forward (per row `n`, vocab `V`, chunk `C`):

```
lse = -inf
for c in 0..V step C:
    lse = logaddexp(lse, logsumexp(logits[n, c:c+C]))
if target == ignore: skip
loss_n = lse - logits[n, target]
return mean(loss_n over valid n)
```

Backward (writes `dlogits[N, V]`, required so autograd can hit `lm_head`):

```
scale = dL / n_valid
dlogits[n, v] = scale * exp(logits[n, v] - lse_n)
dlogits[n, target] -= scale
# ignored rows: 0
```

Compute `exp` in vocab chunks so **peak extra** besides `dlogits` is `O(C)`
not a second `[N, V]` softmax tensor.

### Honesty

- Forward never materializes softmax. That is the real win vs
  `log_softmax + nll`.
- Backward **does** allocate `dlogits [N, V]`. Full Unsloth VRAM (no `dlogits`)
  needs fused linear+CE — **out of this crate’s P1**, tracked for axolotl.
- Label smoothing: out of P1.

### Tests

- Match `log_softmax` + gather + mean (no ignore).
- `ignore_index` drops those rows.
- All-ignore → 0, finite.
- Wrong rank / non-f32 / target OOB → error.

## P1b — RoPE apply

### API

```rust
pub fn rope_custom_op(
    x: &Tensor,    // f32 [B, H, S, D]
    cos: &Tensor,  // f32 [S, D/2]  (or [max_s, D/2] narrowed)
    sin: &Tensor,  // same
) -> Result<Tensor>
```

`CustomOp3`. Rotation:

```
x1, x2 = split_last(x, D/2)
y1 = x1 * cos - x2 * sin
y2 = x2 * cos + x1 * sin
y  = cat(y1, y2)
```

`RotaryEmbedding::forward` calls this for Q and K after narrowing the cache
to `seq_len`. `position_ids`: if unused today, keep sequential `0..S` (parity).
If `position_ids` is present and rank-1/2 i64, gather rows of cos/sin first
(Candle index) then CustomOp — packing-ready without changing the kernel.

### Tests

- Shape preserved.
- Matches current `apply_rotary` reference.

## P1c — SwiGLU elementwise

### API

```rust
pub fn swiglu_custom_op(gate: &Tensor, up: &Tensor) -> Result<Tensor>
// y = silu(gate) * up
// silu(g) = g * sigmoid(g)
```

`CustomOp2`. `SwiGLU::forward` = two `broadcast_matmul` + this op + down
`broadcast_matmul`. `fused_swiglu::swiglu` routes here instead of CubeCL.

Backward (device, may be Candle ops):

```
sig = sigmoid(gate)
d_up   = gy * (gate * sig)
d_gate = gy * up * sig * (1 + gate * (1 - sig))
```

### Tests

- Match `candle_nn::ops::silu(gate) * up`.

## Shared CUDA helper

Extract NVRTC compile + `get_or_load_custom_func` from RMSNorm into
`kernels/custom_op/nvrtc.rs` (`#[cfg(feature = "cuda")]`). All four ops use it.
No new dependency.

## Out of this pass

| Item | Why later |
|------|-----------|
| Fused RMSNorm+RoPE CustomOp | Compose after P1b is green (P1d, same crate, follow-up). |
| Fused linear+CE | Needs `lm_head` weights in the op; axolotl concern. |
| CubeCL FA zero-copy | BLOCKED:api (Candle/CubeCL). CustomOp FA is a different project. |
| f16/bf16 CustomOp | After f32 is consumed by axolotl. |
| peft / qlora / rust-ai-core | Downstream. Do not touch until P1 is on the P0d branch. |

## File list

```
docs/P1_CUSTOMOP_PLAN.md          (this file)
src/kernels/custom_op/nvrtc.rs    (cuda only)
src/kernels/custom_op/ce.rs
src/kernels/custom_op/rope.rs
src/kernels/custom_op/swiglu.rs
src/kernels/custom_op/mod.rs      (exports)
src/kernels/rope.rs               (call CustomOp)
src/kernels/swiglu.rs             (call CustomOp)
src/kernels/fused_swiglu.rs       (route swiglu() → CustomOp)
CHANGELOG.md / DEBT.md / README
```

## Acceptance

- `cargo test --lib` green (default features).
- `cargo clippy --lib -- -D warnings` green.
- New tests cover CE / RoPE / SwiGLU vs reference.
- README table: CE / RoPE / SwiGLU CustomOp called out; CubeCL FA still
  host-roundtrip.
- No peft/qlora/axolotl edits.

## Order of implementation

1. `nvrtc.rs` + RMSNorm refactor (no behavior change).
2. P1c SwiGLU (smallest, same shape in/out — proves helper).
3. P1b RoPE.
4. P1a CE (new surface, most tests).
5. Docs + PR against `feat/p0d-customop-rmsnorm`.
