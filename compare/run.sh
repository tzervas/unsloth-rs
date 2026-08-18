#!/usr/bin/env bash
# Containerized GPU compare: Python (venv in image) vs unsloth-rs CustomOp.
# Never fakes a green Unsloth import. No 2x/VRAM claims.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="${COMPARE_WORK:-$ROOT/artifacts/compare-work}"
IMG_PY="${COMPARE_PY_IMAGE:-localhost/unsloth-rs-compare-py:local}"
GPU_ARGS=(--device nvidia.com/gpu=all)
mkdir -p "$WORK"

if [[ ! -e /dev/nvidia0 ]]; then
  echo "FAIL_ENV: /dev/nvidia0 missing"
  exit 2
fi

echo "==> build python image (venv inside)"
podman build -t "$IMG_PY" -f "$ROOT/compare/Containerfile.py" "$ROOT"

echo "==> python fixtures + torch (+ unsloth pip-installed into image venv)"
# Scripts bind-mounted so we do not rebuild the image for probe changes.
# Unsloth is installed at runtime (venv in the container). PyPI's torchvision
# wheel fails nms vs torch+cu128 — overwrite from the cu128 index. Pip cache
# lives on /home so we do not re-download 100MB wheels every run.
PIP_CACHE="${COMPARE_PIP_CACHE:-$HOME/.cache/pip}"
TRITON_CACHE="${COMPARE_TRITON_CACHE:-$HOME/.triton}"
mkdir -p "$PIP_CACHE" "$TRITON_CACHE"
PY_MOUNTS=(
  -e COMPARE_WORK=/work/out
  -e UNSLOTH_SKIP_TORCHVISION_CHECK=1
  -e CC=gcc
  -e CXX=g++
  -e TMPDIR=/work/tmp
  -v "$WORK:/work/out:Z"
  -v "$ROOT/compare/py:/opt/compare/py:ro,Z"
  -v "$PIP_CACHE:/root/.cache/pip:Z"
  -v "$TRITON_CACHE:/root/.triton:Z"
  -v "${TMPDIR:-/home/kang/tmp}:/work/tmp:Z"
)
if [[ "${COMPARE_PIP_UNSLOTH:-1}" == "1" ]]; then
  podman run --rm "${GPU_ARGS[@]}" \
    --entrypoint /bin/bash \
    "${PY_MOUNTS[@]}" \
    "$IMG_PY" \
    -lc 'pip install unsloth && pip install --force-reinstall --no-deps --index-url https://download.pytorch.org/whl/cu128 torchvision && python /opt/compare/py/generate_and_unsloth.py'
else
  podman run --rm "${GPU_ARGS[@]}" \
    "${PY_MOUNTS[@]}" \
    "$IMG_PY" \
    /opt/compare/py/generate_and_unsloth.py
fi

echo "==> rust CustomOp in CUDA container (host rustup + CUDA bind-mounted)"
if [[ ! -x "${CARGO_HOME:-$HOME/.cargo}/bin/cargo" ]]; then
  echo "FAIL_ENV: host cargo not found for bind-mount"
  exit 2
fi
CUDA_HOST="${CUDA_HOME:-/usr/local/cuda-13.1}"
if [[ ! -d "$CUDA_HOST" ]]; then
  CUDA_HOST=/usr/local/cuda
fi

# Override the CUDA image ENTRYPOINT (it warns about ldconfig / "no driver"
# even when CDI GPU passthrough is live). Device::new_cuda is the real check.
podman run --rm "${GPU_ARGS[@]}" \
  --entrypoint /usr/bin/env \
  -e COMPARE_WORK=/work/out \
  -e CUDA_COMPUTE_CAP="${CUDA_COMPUTE_CAP:-90}" \
  -e RUSTUP_HOME=/opt/rustup \
  -e CARGO_HOME=/opt/cargo \
  -e PATH=/opt/cargo/bin:/usr/local/cuda/bin:/usr/bin \
  -e CUDA_HOME=/usr/local/cuda \
  -v "$HOME/.rustup:/opt/rustup:ro" \
  -v "$HOME/.cargo:/opt/cargo" \
  -v "$CUDA_HOST:/usr/local/cuda:ro" \
  -v "$ROOT:/src:Z" \
  -v "$WORK:/work/out:Z" \
  -w /src \
  docker.io/nvidia/cuda:12.8.0-devel-ubuntu24.04 \
  cargo run --features cuda --release --example compare_ops -- /work/out

echo "==> report"
podman run --rm \
  --entrypoint /opt/venv/bin/python \
  -e COMPARE_WORK=/work/out \
  -v "$WORK:/work/out:Z" \
  -v "$ROOT/compare/py:/opt/compare/py:ro,Z" \
  "$IMG_PY" \
  /opt/compare/py/report.py

cp -f "$WORK/py-rs-compare.json" "$ROOT/artifacts/py-rs-compare.json"
echo "wrote artifacts/py-rs-compare.json"
