# Technical debt — unsloth-rs

## Flash attention residual (P8 / W1 train track)

**Last updated:** 2026-07-16  
**Evidence:** `/root/work/plans/evidence/48h/L1-F-unsloth/`, `/root/work/plans/evidence/P8-residuals/`

### Resolved (test reference)

- **Integration reference scaling:** `tests/gpu/flash_attention.rs` `attention_reference_cpu` used `scores / scale` while the CubeCL fallback and `kernels/cubecl/kernel.rs` `reference_attention` use `scores * scale` with `scale = 1/√head_dim`. That mismatch caused false failures on `test_flash_attention_cpu_fallback_accuracy` / `test_flash_attention_gpu_integration` (MAE ~0.73). Fixed in PR branch `fix/flash-attn-reference-scale` (multiply, not divide).

### Open — environment / GPU gate

| Item | Status | Notes |
|------|--------|-------|
| `test_flash_attention_gpu_numerical_equivalence` | **Not run** | Requires CUDA device + `cuda` feature |
| `cargo test --features cuda --test integration` | **Resolved** | Added missing `use candle_core::IndexOp` import in `tests/gpu/flash_attention.rs` |
| Host GPU runtime | **BLOCKED:env** | Evidence: missing `/dev/nvidia0`, `CUDA_ERROR_NO_DEVICE`; Blackwell CC 12.0 needs `CUDA_COMPUTE_CAP=90` for nvcc 12.0 candle-kernels build |
| `./scripts/gpu-test.sh` full validate | **BLOCKED:env** | Documented in L1-F `env-snapshot.log` |

### Open — product / algorithm (not in P8 scope)

- Ternary attention paths still use `/ scale` in some modules (`src/kernels/attention.rs`, `ternary/attention.rs`) — separate from flash-attn integration reference; audit if tests expand.
- Consumer train story (axolotl-rs E2E) remains downstream of honest GPU numerical evidence.

### Verification commands

```bash
# CPU default (post reference-scale fix)
cargo test --workspace --no-default-features
cargo test --test integration test_flash_attention

# GPU (when env healthy)
CUDA_COMPUTE_CAP=90 cargo test --features cuda --test integration test_flash_attention_gpu_numerical_equivalence -- --nocapture
```