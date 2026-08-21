# Containerized Python Unsloth vs unsloth-rs

**Not product parity.** Same f32 shapes on one 5080 via **Podman GPU passthrough**.
Python lives in a venv *inside* the container. No 2× / 70% VRAM claims.

## What it compares

| Op | Torch reference | Python Unsloth (if importable) | unsloth-rs CustomOp CUDA |
|----|-----------------|-------------------------------|---------------------------|
| RMSNorm | yes | probe at runtime | yes |
| LayerNorm | yes | probe (`fast_layernorm`) | yes (`ops::layernorm`) |
| RoPE | yes | probe | yes |
| RoPE + ids | yes | probe (`rope_embedding_indices`) | yes (`ops::rope_with_ids`) |
| SwiGLU | yes | probe | yes |
| GeGLU (exact) | yes | probe (`geglu_exact_forward_kernel`) | yes (`ops::geglu`) |
| chunked CE | yes | probe | yes |
| attention (causal SDPA) | yes | **not compared** (no standalone kernel) | yes (`ops::attention`) |
| sliding-window attn | yes (masked softmax) | **not compared** (not invented) | yes (`ops::attention_window`, window=64) |
| tanh softcap attn | yes | **not compared** (not invented) | yes (`ops::attention_softcap`, cap=50) |

Shapes: `B=2 H=8 D=64`, seq **128**, **512**, and **2048**, dtype **f32**, seed 0.
s2048 OOM is `FAIL_ENV`, not a green skip.

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

## Measured on this 5080 (2026-08-18)

Podman `--device nvidia.com/gpu=all`, persist volume Unsloth `2026.8.18`,
torch `2.11.0+cu128`, rust CustomOp `CUDA_COMPUTE_CAP=90`. Full dump:
`artifacts/py-rs-compare.json`. s2048 did **not** OOM.

MAE (f32, seed 0). Elementwise ~1e-8. CE scalar ~1–2e-6. Unsloth attn
is not compared (no standalone kernel). Unsloth `rope_embedding_indices`
was **not** used (inplace QK path illegal-memory-accessed); gathered
tables + sequential Unsloth rope instead.

| op | torch vs rust s128 / s512 / s2048 | torch vs unsloth | rust vs unsloth |
|----|-----------------------------------|------------------|-----------------|
| RMSNorm | 2.2e-8 / 2.3e-8 / 2.4e-8 | ~2.6e-8 | ~1.0e-8 |
| LayerNorm | 2.1e-8 / 2.0e-8 / 2.2e-8 | ~2.1e-8 | ~1.6e-8 |
| RoPE | 1.5e-8 / 1.4e-8 / 1.3e-8 | ~1.2e-8 | ~9e-9 |
| RoPE+ids | 1.5e-8 / 1.4e-8 / 1.3e-8 | ~1.2e-8 | ~9e-9 |
| SwiGLU | 6.5e-9 / 6.4e-9 / 6.4e-9 | ~1.0e-8 | ~6.4e-9 |
| GeGLU | 0 / 0 / 0 | ~1e-9 | ~1e-9 |
| CE | 1.9e-6 / 1.4e-6 / 1.9e-6 | 0 / 4.8e-7 / 0 | ~1.9e-6 |
| attn | 8.2e-8 / 5.5e-8 / 3.4e-8 | n/a | n/a |
| attn window=64 | 2.6e-8 / 2.6e-8 / 2.6e-8 | n/a | n/a |
| attn softcap=50 | 2.9e-8 / 2.5e-8 / 2.3e-8 | n/a | n/a |

Latency in **this** artifact is host+`cuda.synchronize` **p50/p99**
(warmup 5, n=100) for torch, Unsloth, and rust compare dumps
(`torch_p50_ms` / `unsloth_p50_ms` / `rust_p50_ms`). Device-only
(CUDA-event) rust p50/p99 stays in `artifacts/custom_op_cuda.json`.
Shapes are launch-bound. Do not market compare rust p50 against torch p50
as a 2× claim. Rust attention is owned SRAM-tiled FA (not Unsloth PTX).
On s512 it is still **slower** than torch SDPA (0.68 ms vs 0.097 ms p50).

Host+sync p50 ms (5080, CAP pin 90, n=100). Elementwise is launch-bound.

| op | torch s128 / s512 / s2048 | unsloth s128 / s512 / s2048 | rust s128 / s512 / s2048 |
|----|---------------------------|-----------------------------|--------------------------|
| RMSNorm | 0.023 / 0.022 / 0.024 | 0.027 / 0.027 / 0.028 | 0.011 / 0.008 / 0.012 |
| LayerNorm | 0.009 / 0.013 / 0.016 | 0.072 / 0.073 / 0.073 | 0.011 / 0.011 / 0.014 |
| RoPE | 0.032 / 0.029 / 0.037 | 0.060 / 0.059 / 0.061 | 0.011 / 0.012 / 0.029 |
| RoPE+ids | 0.038 / 0.038 / 0.046 | 0.073 / 0.073 / 0.074 | 0.017 / 0.021 / 0.035 |
| SwiGLU | 0.012 / 0.009 / 0.011 | 0.014 / 0.014 / 0.016 | 0.008 / 0.010 / 0.011 |
| GeGLU | 0.022 / 0.022 / 0.024 | 0.017 / 0.014 / 0.015 | 0.010 / 0.011 / 0.009 |
| CE | 0.016 / 0.016 / 0.018 | 0.051 / 0.051 / 0.051 | 0.011 / 0.013 / 0.018 |
| attn | 0.026 / 0.098 / 0.766 | n/a | 0.077 / 0.553 / 7.27 |
| attn window | 0.066 / 0.120 / 3.88 | n/a | 0.068 / 0.153 / 0.494 |
| attn softcap | 0.062 / 0.143 / 6.18 | n/a | 0.104 / 0.735 / 8.43 |

Rust full causal attn is still **slower** than torch SDPA on s512/s2048.
Do not claim 2×. s2048 window is cheaper than full causal because the
tile kernel skips K tiles outside the window.

Host+event p50 after TILE-OCC (16×16 tile, parallel softmax; not compare host+sync):

| attn | event p50 | vs prior 0.68 ms s512 |
|------|-----------|------------------------|
| s128 | 0.074 ms | — |
| s512 | **0.547 ms** | faster than 0.68 ms |
| s2048 | 7.22 ms | new |

Still ~5–6× slower than torch SDPA s512. G-UNS-03 stays open.

`CUDA_COMPUTE_CAP=120` tiled s512 test **PASS** on this 5080 (SM 12.0, CUDA 13.1).
Default pin remains **90**. Not a native-Blackwell claim.

Host+event p50/p99 (after cache; not a sacred-bar number):

```bash
CUDA_COMPUTE_CAP=90 TMPDIR=/home/kang/tmp cargo bench --features cuda --bench custom_op_cuda
```
