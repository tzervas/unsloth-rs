#!/usr/bin/env python3
"""MAE between torch, optional Unsloth, and unsloth-rs npy dumps."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

WORK = Path(os.environ.get("COMPARE_WORK", "/work/out"))
OPS = ("rmsnorm", "rope", "swiglu", "ce", "attn")


def _stat_map(raw: dict, key: str) -> dict:
    out = {}
    for op, val in raw.items():
        if isinstance(val, dict) and key in val:
            out[op] = val[key]
        elif isinstance(val, (int, float)) and key == "p50":
            # Legacy one-shot float: do not relabel it as p50.
            continue
    return out


def _p50_map(raw: dict) -> dict:
    return _stat_map(raw, "p50")


def _p99_map(raw: dict) -> dict:
    return _stat_map(raw, "p99")


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def load_pair(root: Path, left: str, right: str):
    lf, rf = root / left, root / right
    if lf.suffix == ".npy" and lf.is_file() and rf.with_suffix(".f32").is_file():
        return np.load(lf).reshape(-1), np.fromfile(rf.with_suffix(".f32"), dtype=np.float32)
    if lf.with_suffix(".f32").is_file() and rf.with_suffix(".f32").is_file():
        return (
            np.fromfile(lf.with_suffix(".f32"), dtype=np.float32),
            np.fromfile(rf.with_suffix(".f32"), dtype=np.float32),
        )
    return None


def main() -> None:
    meta = json.loads((WORK / "python_meta.json").read_text())
    rust_meta_p = WORK / "rust_meta.json"
    rust_meta = json.loads(rust_meta_p.read_text()) if rust_meta_p.is_file() else {}
    report = {
        "sacred_bar": False,
        "note": "f32 same-shape compare. Not Unsloth product parity. No 2x/VRAM claims. torch/Unsloth/rust compare latency is host+sync p50/p99 (warmup 5, n=100). Rust host+event p50/p99 after PTX cache still lives in artifacts/custom_op_cuda.json. Shapes are launch-bound; not a sacred-bar number.",
        "caveats": [
            "torch/unsloth/rust compare ms are host+cuda-sync p50/p99 (n=100 after 5 warmups), not one-shot.",
            "Rust event (device-only) p50/p99 is artifacts/custom_op_cuda.json, not this file.",
            "Shapes are tiny (B=2 H=8 D=64); elementwise is launch-bound.",
            "Rust attention is unsloth_rs::ops::attention tiled SRAM FA (owned NVRTC). Not Unsloth PTX. Extra mask still GEMM.",
            "Rust kernels compiled with CUDA_COMPUTE_CAP=90 on SM 12.0 hardware (compile pin).",
            "Unsloth attn is not a standalone kernel; not compared.",
        ],
        "python": meta,
        "rust": rust_meta,
        "mae": {},
        "latency_ms": {},
    }
    for case in meta.get("cases", []):
        tag = case["tag"]
        root = WORK / tag
        row = {}
        for op in OPS:
            pair = load_pair(root, f"torch_{op}.npy", f"rust_{op}.npy")
            if pair and pair[0].size == pair[1].size:
                row[f"torch_vs_rust_{op}"] = mae(pair[0], pair[1])
            uns_p = root / f"unsloth_{op}.npy"
            torch_p = root / f"torch_{op}.npy"
            if torch_p.is_file() and uns_p.is_file():
                row[f"torch_vs_unsloth_{op}"] = mae(np.load(torch_p), np.load(uns_p))
            if uns_p.is_file() and (root / f"rust_{op}.f32").is_file():
                row[f"rust_vs_unsloth_{op}"] = mae(
                    np.load(uns_p).reshape(-1),
                    np.fromfile(root / f"rust_{op}.f32", dtype=np.float32),
                )
        report["mae"][tag] = row
        lat = {"torch": case.get("torch_ms", {}), "unsloth": case.get("unsloth_ms", {})}
        rust_ms = (rust_meta.get("ms") or {}).get(tag)
        if rust_ms:
            lat["rust"] = rust_ms
        report["latency_ms"][tag] = lat
        report.setdefault("torch_p50_ms", {})[tag] = _p50_map(case.get("torch_ms") or {})
        report.setdefault("torch_p99_ms", {})[tag] = _p99_map(case.get("torch_ms") or {})
        report.setdefault("unsloth_p50_ms", {})[tag] = _p50_map(case.get("unsloth_ms") or {})
        report.setdefault("unsloth_p99_ms", {})[tag] = _p99_map(case.get("unsloth_ms") or {})
        report.setdefault("rust_p50_ms", {})[tag] = _p50_map(rust_ms or {})
        report.setdefault("rust_p99_ms", {})[tag] = _p99_map(rust_ms or {})
    out = WORK / "py-rs-compare.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["mae"], indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
