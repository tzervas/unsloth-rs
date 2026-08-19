# Containerized Python Unsloth vs unsloth-rs

**Not product parity.** Same f32 shapes on one 5080 via **Podman GPU passthrough**.
Python lives in a venv *inside* the container. No 2× / 70% VRAM claims.

## What it compares

| Op | Torch reference | Python Unsloth (if importable) | unsloth-rs CustomOp CUDA |
|----|-----------------|-------------------------------|---------------------------|
| RMSNorm | yes | probe at runtime | yes |
| RoPE | yes | probe | yes |
| SwiGLU | yes | probe | yes |
| chunked CE | yes | probe | yes |
| attention (causal SDPA) | yes | probe | yes (`ops::attention`) |

Shapes: `B=2 H=8 D=64`, seq **128** and **512**, dtype **f32**, seed 0.

## Run (workstation, GPU free)

```bash
# from repo root
./compare/run.sh
```

Needs: Podman + `nvidia-ctk` + `/dev/nvidia0`. Writes `artifacts/py-rs-compare.json`.

`FAIL_ENV` if the GPU, CUDA image, or Unsloth import is missing — never a silent green.

Torch stays in the image venv. Unsloth is pip-installed into a **named
volume** (`unsloth-rs-compare-site` → `/opt/site-extra`, `PYTHONPATH`).
Later runs skip pip if `import unsloth` works. Baking Unsloth into the
image OOMs `/var/tmp`. Force-reinstall `torchvision` from the **cu128**
index (PyPI's wheel fails `nms` against `torch==2.11+cu128`). Import is
still probed at runtime. Triton JIT needs `gcc` in the image.

`COMPARE_SITE_VOL` overrides the volume name. `podman volume rm
unsloth-rs-compare-site` to force a reinstall.

Unsloth attention is **not** a standalone kernel here (patched SDPA/flex) — recorded as such, not compared.

## Measured on this 5080 (2026-08-17)

Podman `--device nvidia.com/gpu=all`, venv inside the container, `unsloth==2026.8.18`, torch `2.11.0+cu128`, rust CustomOp with `CUDA_COMPUTE_CAP=90`. Full dump: `artifacts/py-rs-compare.json`.

MAE (f32, seed 0, 2026-08-17 persist+p50 rerun). Elementwise ~1e-8.
CE is a scalar; torch-vs-rust CE MAE this run is 3.8e-6 / 1.9e-6 (s128 / s512).

| op | torch vs rust (s128 / s512) | torch vs unsloth | rust vs unsloth |
|----|----------------------------|------------------|-----------------|
| RMSNorm | 2.2e-8 / 2.5e-8 | 2.6e-8 / 2.9e-8 | 1.0e-8 / 1.1e-8 |
| RoPE | 1.5e-8 / 1.4e-8 | 1.2e-8 / 1.2e-8 | 8.9e-9 / 8.9e-9 |
| SwiGLU | 6.5e-9 / 6.3e-9 | 1.0e-8 / 9.5e-9 | 6.7e-9 / 6.3e-9 |
| CE | 3.8e-6 / 1.9e-6 | 0 / 4.8e-7 | 3.8e-6 / 2.4e-6 |
| attn | 7.8e-8 / 5.7e-8 | n/a | n/a |

Latency in **this** artifact is host+`cuda.synchronize` **p50/p99**
(warmup 5, n=100) for torch, Unsloth, and rust compare dumps
(`torch_p50_ms` / `unsloth_p50_ms` / `rust_p50_ms`). Device-only
(CUDA-event) rust p50/p99 stays in `artifacts/custom_op_cuda.json`.
Shapes are launch-bound. Do not market compare rust p50 against torch p50
as a 2× claim. Rust attention is owned SRAM-tiled FA (not Unsloth PTX).
On s512 it is still **slower** than torch SDPA (0.68 ms vs 0.097 ms p50).

Host+sync p50 ms (5080, CAP pin 90, n=100):

| op | torch s128 / s512 | unsloth s128 / s512 | rust s128 / s512 |
|----|-------------------|---------------------|------------------|
| RMSNorm | 0.023 / 0.023 | 0.027 / 0.027 | 0.008 / 0.008 |
| RoPE | 0.027 / 0.070 | 0.060 / 0.061 | 0.009 / 0.012 |
| SwiGLU | 0.010 / 0.010 | 0.014 / 0.016 | 0.007 / 0.007 |
| CE | 0.011 / 0.016 | 0.051 / 0.052 | 0.011 / 0.012 |
| attn | 0.023 / 0.097 | n/a | 0.106 / 0.679 |

Host+event p50/p99 (after cache; not a sacred-bar number):

```bash
CUDA_COMPUTE_CAP=90 TMPDIR=/home/kang/tmp cargo bench --features cuda --bench custom_op_cuda
```
