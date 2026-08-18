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
| attention (causal SDPA) | yes | probe | yes (`attention_device`) |

Shapes: `B=2 H=8 D=64`, seq **128** and **512**, dtype **f32**, seed 0.

## Run (workstation, GPU free)

```bash
# from repo root
./compare/run.sh
```

Needs: Podman + `nvidia-ctk` + `/dev/nvidia0`. Writes `artifacts/py-rs-compare.json`.

`FAIL_ENV` if the GPU, CUDA image, or Unsloth import is missing — never a silent green.

The Python image venv installs `unsloth` then force-reinstalls `torchvision` from the **cu128** index (PyPI's wheel fails `torchvision::nms` against `torch==2.11+cu128`). Import is still probed at runtime. Triton JIT needs `gcc` in the image.

Unsloth attention is **not** a standalone kernel here (patched SDPA/flex) — recorded as such, not compared.

## Measured on this 5080 (2026-08-17)

Podman `--device nvidia.com/gpu=all`, venv inside the container, `unsloth==2026.8.18`, torch `2.11.0+cu128`, rust CustomOp with `CUDA_COMPUTE_CAP=90`. Full dump: `artifacts/py-rs-compare.json`.

MAE (f32, seed 0) — all under 3e-6; elementwise ~1e-8:

| op | torch vs rust (s128 / s512) | torch vs unsloth | rust vs unsloth |
|----|----------------------------|------------------|-----------------|
| RMSNorm | 2.2e-8 / 2.5e-8 | 2.6e-8 / 2.9e-8 | 1.0e-8 / 1.1e-8 |
| RoPE | 1.5e-8 / 1.4e-8 | 1.2e-8 / 1.2e-8 | 8.9e-9 / 8.9e-9 |
| SwiGLU | 6.5e-9 / 6.3e-9 | 1.0e-8 / 9.5e-9 | 6.7e-9 / 6.3e-9 |
| CE | 1.9e-6 / 2.4e-6 | 0 / 4.8e-7 | 1.9e-6 / 2.9e-6 |
| attn | 8.3e-8 / 5.9e-8 | n/a | n/a |

Latency in **this** artifact is **one shot after 3 warmups**, not p50/p99.
`rust.ms` is **pre-cache NVRTC-per-launch** (compile on every CustomOp call)
and is **superseded** by `artifacts/custom_op_cuda.json` (host vs CUDA-event
p50/p99 after the PTX cache). torch/Unsloth remain one-shot. Do not market
the pre-cache rust.ms against torch ~0.03 ms. Rust attention is GEMM+softmax
on `CudaStorage`, not tiled FA.

Host+event p50/p99 (after cache; not a sacred-bar number):

```bash
CUDA_COMPUTE_CAP=90 TMPDIR=/home/kang/tmp cargo bench --features cuda --bench custom_op_cuda
```
