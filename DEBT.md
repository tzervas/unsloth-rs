# Technical debt — unsloth-rs

## Flash attention residual (P8 / W1 train track)

**Last updated:** 2026-07-22
**Evidence:** `/root/work/plans/evidence/48h/L1-F-unsloth/`, `/root/work/plans/evidence/P8-residuals/`

### Resolved (test reference)

- **Integration reference scaling:** `tests/gpu/flash_attention.rs` `attention_reference_cpu` used `scores / scale` while the CubeCL fallback and `kernels/cubecl/kernel.rs` `reference_attention` use `scores * scale` with `scale = 1/√head_dim`. That mismatch caused false failures on `test_flash_attention_cpu_fallback_accuracy` / `test_flash_attention_gpu_integration` (MAE ~0.73). Fixed in PR branch `fix/flash-attn-reference-scale` (multiply, not divide).

### Open — CubeCL interop host copies (UNS-P1-01 / PR-070) — **permanent with current APIs**

| Item | Status | Notes |
|------|--------|-------|
| Candle ↔ CubeCL zero-copy | **BLOCKED:api** | Candle 0.9 has no public device-ptr export from `Tensor`; CubeCL 0.9 has no external `CUdeviceptr` import into `Handle`. Separate CUDA memory managers. |
| `candle_to_cubecl_handle` | **Host D2H** | Uses `to_vec1()` — required |
| `cubecl_to_candle_tensor` | **Host H2D** | Uses `Tensor::from_vec` — required |
| FA / fused **speed claims** | **Demoted** | Do **not** claim 2× or VRAM wins for CubeCL path end-to-end while `interop_requires_host_roundtrip()` is `true` |

**Future unblock paths (not in this PR):** Candle `CustomOp` kernels on `CudaStorage` (skip CubeCL handles), or upstream external-buffer APIs.

### Open — environment / GPU numerical gate (UNS-P1-02 / PR-071)

| Item | Status | Notes |
|------|--------|-------|
| `test_flash_attention_gpu_numerical_equivalence` | **Runs under `--features cuda`** | Not `#[ignore]`; MAE&lt;1e-5, RMSE&lt;1e-4, cosine&gt;0.999; **BLOCKED:env** vs **FAIL (accuracy)** |
| Gate run (STACK-UNS-FINISH) | **PASS** | RTX 5080, `/dev/nvidia0`, `CUDA_COMPUTE_CAP=90`, `LD_LIBRARY_PATH=/usr/lib/wsl/lib:...`. Full `cargo test --features cuda`: lib 137, integration 45 ok / 3 ignored. Numerical gate MAE ~2e-8, cosine 1.0. Without WSL libcuda, CubeCL may panic `CUDA_ERROR_NO_DEVICE` → Candle fallback. **No 2× claims.** |
| Default `cargo test` (no cuda) | **Green** | Gate not compiled without feature |
| Host GPU compile pin | **Documented** | Blackwell CC 12.0 + nvcc 12.0 often needs `CUDA_COMPUTE_CAP=90` |
| CubeCL cudarc init | **env-sensitive** | Needs correct `libcuda` (WSL: prefer `/usr/lib/wsl/lib`). Panic unwrap in cubecl-cuda 0.9 if wrong; catch_unwind → Candle fallback. |

**Classification vocabulary:**

| Label | Meaning |
|-------|---------|
| **PASS** | Gate executed on real device; metrics within thresholds |
| **FAIL (accuracy)** | Device path ran; MAE/RMSE/cosine out of bounds |
| **BLOCKED:env / FAIL_ENV** | Missing `/dev/nvidia0`, `CUDA_ERROR_NO_DEVICE`, toolkit/arch pin, or suite not run |
| **BLOCKED:api** | Upstream API cannot express the optimization (e.g. zero-copy interop) |

### Open — product / algorithm (not in P8 scope)

- **FusedAttention flash scale (UNS-P0-03):** fixed in Wave-3 — `forward_flash_attention` now passes `1/sqrt(head_dim)` (multiply convention). Covered by `test_flash_path_scale_matches_cpu_one_over_sqrt_d`.
- Ternary attention CPU path uses divide-by-sqrt (correct for that API); keep conventions consistent if tests expand.
- Consumer train story (axolotl-rs E2E) remains downstream of honest GPU numerical evidence.
- **Gradient checkpoint recompute (UNS-P1-03 / PR-083):** public always-`Err` stub **removed**. Only `CheckpointConfig` + memory *estimates* remain. No marketed recompute API.

### Verification commands

```bash
# CPU default (CI honesty path)
cargo test --workspace --no-default-features
cargo test --test integration test_flash_attention

# GPU numerical gate (when env healthy: /dev/nvidia0 + toolkit)
CUDA_COMPUTE_CAP=90 cargo test --features cuda --test integration \
  test_flash_attention_gpu_numerical_equivalence -- --nocapture
```

If `/dev/nvidia0` is missing, the cuda-feature gate **fails with `BLOCKED:env`** — intentional (no silent pass).

## CUDA compute capability / CI honesty (PR-027 / UNS-P0-05)

### Environment contract

| Situation | Classification | Action |
|-----------|----------------|--------|
| No NVIDIA device nodes (e.g. only `/dev/nvidiactl`, no `/dev/nvidia0`) | **FAIL_ENV** | Skip GPU tests; do **not** mark suite green |
| `nvidia-smi` OK but `CUDA_ERROR_NO_DEVICE` at runtime | **FAIL_ENV** | Fix cgroup/device mounts; re-run |
| Default `CUDA_COMPUTE_CAP=120` (Blackwell) with nvcc that only targets ≤ 90 | **FAIL_ENV** (compile) | Pin `CUDA_COMPUTE_CAP=90` **or** install a toolkit that supports the real arch |
| `cuda` feature off / CPU CI | **N/A (CPU path)** | Default CI; expected green path |
| GPU numerical gate not executed | **Not run / BLOCKED:env** | Document; never claim PASS |

### Recommended commands

```bash
# CPU (default features) — CI honesty path
cargo test --workspace --no-default-features

# CUDA compile check (often needs CAP pin on nvcc 12.0 hosts)
CUDA_COMPUTE_CAP=90 cargo check --features cuda

# GPU numerical tests — only when device nodes + toolkit healthy
CUDA_COMPUTE_CAP=90 cargo test --features cuda --test integration \
  test_flash_attention_gpu_numerical_equivalence -- --nocapture
```

### CI policy

- Workflows under `.github/workflows/` run **CPU** check/test by default.
- GPU jobs, if added later, must:
  1. Detect device + toolkit,
  2. Exit as **skipped / FAIL_ENV** (with clear log) when missing — not a silent pass that implies GPU coverage,
  3. Never publish “GPU suite green” without device evidence.
- See [GPU_SETUP.md](GPU_SETUP.md) for toolkit install and CAP notes.

## Closed / scoped (STACK-UNS-FINISH / 1.0.3)

| ID | Result |
|----|--------|
| **UNS-P1-04** | **Scoped F32-only** — `interop_f32_only()` / `interop_supports_dtype`; no CubeCL f16/bf16 path in 1.0.x. Host `training::convert_precision` remains for dtype helpers only. |
| **UNS-P2-01** | **Archived non-goal** — ternary CubeCL drafts moved to `archive/ternary_cubecl/`; excluded from crates.io package; CPU ternary remains. |
