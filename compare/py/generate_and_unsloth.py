#!/usr/bin/env python3
"""Write shared f32 fixtures and Python-side outputs (torch + unsloth if present)."""

from __future__ import annotations

import importlib
import json
import os
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

WORK = Path(os.environ.get("COMPARE_WORK", "/work/out"))
SEED = 0
EPS = 1e-5
WARMUP = 3
SHAPES = ((2, 8, 128, 64), (2, 8, 512, 64))  # B,H,S,D


def save(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    a = np.ascontiguousarray(arr.astype(np.float32, copy=False))
    np.save(path, a)
    a.tofile(path.with_suffix(".f32"))


def torch_rmsnorm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)
    return x / rms * w


def torch_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x [B,H,S,D]; cos/sin [S, D/2]. NeoX half-split (matches unsloth-rs CustomOp).
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def torch_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return F.silu(gate) * up


def torch_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, targets, ignore_index=-100)


def torch_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    scale = q.shape[-1] ** -0.5
    return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)


def probe_unsloth() -> dict:
    info: dict = {"importable": False, "ops": {}}
    try:
        u = importlib.import_module("unsloth")
        info["importable"] = True
        info["file"] = getattr(u, "__file__", None)
        info["version"] = getattr(u, "__version__", "unknown")
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
        return info
    for name in (
        "unsloth.kernels.rms_layernorm",
        "unsloth.kernels.rope_embedding",
        "unsloth.kernels.swiglu",
        "unsloth.kernels.cross_entropy_loss",
    ):
        try:
            importlib.import_module(name)
            info["ops"][name] = True
        except Exception as e:
            info["ops"][name] = f"{type(e).__name__}: {e}"
    info["ops"]["attn"] = (
        "no standalone Unsloth attn kernel; patched SDPA/flex is not compared"
    )
    return info


def _call(label: str, fn):
    try:
        return fn(), None
    except Exception as e:
        return None, f"{label}: {type(e).__name__}: {e}"


def try_unsloth_rmsnorm(x: torch.Tensor, w: torch.Tensor):
    # fast_rms_layernorm(layernorm, X) wants a module with .weight and .eps.
    def go():
        from unsloth.kernels.rms_layernorm import fast_rms_layernorm

        ln = types.SimpleNamespace(weight=w, eps=EPS)
        return fast_rms_layernorm(ln, x)

    return _call("rmsnorm", go)


def try_unsloth_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # fast_rope_embedding(Q, K, cos, sin): Q/K [B,H,S,D]; returns (Q_out, K_out).
    def go():
        from unsloth.kernels.rope_embedding import fast_rope_embedding

        q_out, _k_out = fast_rope_embedding(q, k, cos, sin)
        return q_out

    return _call("rope", go)


def try_unsloth_swiglu(gate: torch.Tensor, up: torch.Tensor):
    # swiglu_fg_kernel(e, g): [B,S,D] -> silu(e)*g
    def go():
        from unsloth.kernels.swiglu import swiglu_fg_kernel

        return swiglu_fg_kernel(gate, up)

    return _call("swiglu", go)


def try_unsloth_ce(logits_b_s_v: torch.Tensor, labels_b_s: torch.Tensor):
    # fast_cross_entropy_loss: logits [B,S,V], labels [B,S] -> mean scalar
    def go():
        from unsloth.kernels.cross_entropy_loss import fast_cross_entropy_loss

        return fast_cross_entropy_loss(logits_b_s_v, labels_b_s)

    return _call("ce", go)


def timed(fn, device: torch.device):
    if device.type == "cuda":
        for _ in range(WARMUP):
            fn()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return out, (time.perf_counter() - t0) * 1000.0


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    uns = probe_unsloth()
    cases = []
    for b, h, s, d in SHAPES:
        tag = f"s{s}"
        root = WORK / tag
        root.mkdir(parents=True, exist_ok=True)
        x = torch.randn(b, s, d, device=device, dtype=torch.float32)
        w = torch.randn(d, device=device, dtype=torch.float32)
        q = torch.randn(b, h, s, d, device=device, dtype=torch.float32)
        k = torch.randn(b, h, s, d, device=device, dtype=torch.float32)
        v = torch.randn(b, h, s, d, device=device, dtype=torch.float32)
        gate = torch.randn(b, s, d, device=device, dtype=torch.float32)
        up = torch.randn(b, s, d, device=device, dtype=torch.float32)
        logits = torch.randn(b * s, 128, device=device, dtype=torch.float32)
        targets = torch.randint(0, 128, (b * s,), device=device)
        half = d // 2
        pos = torch.arange(s, device=device, dtype=torch.float32)
        inv = 1.0 / (
            10000.0
            ** (torch.arange(0, half, device=device, dtype=torch.float32) / half)
        )
        freqs = torch.outer(pos, inv)
        cos = freqs.cos()
        sin = freqs.sin()

        save(root / "x.npy", x.detach().cpu().numpy())
        save(root / "w.npy", w.detach().cpu().numpy())
        save(root / "q.npy", q.detach().cpu().numpy())
        save(root / "k.npy", k.detach().cpu().numpy())
        save(root / "v.npy", v.detach().cpu().numpy())
        save(root / "gate.npy", gate.detach().cpu().numpy())
        save(root / "up.npy", up.detach().cpu().numpy())
        save(root / "logits.npy", logits.detach().cpu().numpy())
        save(root / "cos.npy", cos.detach().cpu().numpy())
        save(root / "sin.npy", sin.detach().cpu().numpy())
        tgt = targets.detach().cpu().numpy().astype(np.int64)
        np.save(root / "targets.npy", tgt)
        tgt.tofile(root / "targets.i64")
        (root / "shape.json").write_text(
            json.dumps({"B": b, "H": h, "S": s, "D": d}, separators=(",", ":")) + "\n"
        )

        rms, rms_ms = timed(lambda: torch_rmsnorm(x, w), device)
        rope, rope_ms = timed(lambda: torch_rope(q, cos, sin), device)
        swi, swi_ms = timed(lambda: torch_swiglu(gate, up), device)
        ce, ce_ms = timed(lambda: torch_ce(logits, targets), device)
        attn, attn_ms = timed(lambda: torch_attn(q, k, v), device)
        save(root / "torch_rmsnorm.npy", rms.detach().cpu().numpy())
        save(root / "torch_rope.npy", rope.detach().cpu().numpy())
        save(root / "torch_swiglu.npy", swi.detach().cpu().numpy())
        save(root / "torch_ce.npy", np.array([float(ce.detach().cpu())], dtype=np.float32))
        save(root / "torch_attn.npy", attn.detach().cpu().numpy())

        uns_err: dict[str, str] = {}
        uns_ms: dict[str, float] = {}
        uns_ok: dict[str, bool] = {}

        def record(name: str, tensor_or_none, err, ms=None):
            uns_ok[name] = tensor_or_none is not None
            if err:
                uns_err[name] = err
            if ms is not None and tensor_or_none is not None:
                uns_ms[name] = ms
            if tensor_or_none is None:
                return
            arr = tensor_or_none.detach().float().cpu().numpy()
            if arr.ndim == 0:
                arr = np.array([float(arr)], dtype=np.float32)
            save(root / f"unsloth_{name}.npy", arr)

        if uns.get("importable"):
            out, err = try_unsloth_rmsnorm(x, w)
            if out is not None:
                out, ms = timed(lambda: try_unsloth_rmsnorm(x, w)[0], device)
                record("rmsnorm", out, None, ms)
            else:
                record("rmsnorm", None, err)

            out, err = try_unsloth_rope(q, k, cos, sin)
            if out is not None:
                out, ms = timed(lambda: try_unsloth_rope(q, k, cos, sin)[0], device)
                record("rope", out, None, ms)
            else:
                record("rope", None, err)

            out, err = try_unsloth_swiglu(gate, up)
            if out is not None:
                out, ms = timed(lambda: try_unsloth_swiglu(gate, up)[0], device)
                record("swiglu", out, None, ms)
            else:
                record("swiglu", None, err)

            logits_b = logits.view(b, s, 128)
            labels_b = targets.view(b, s)
            out, err = try_unsloth_ce(logits_b, labels_b)
            if out is not None:
                out, ms = timed(lambda: try_unsloth_ce(logits_b, labels_b)[0], device)
                record("ce", out, None, ms)
            else:
                record("ce", None, err)

        cases.append(
            {
                "tag": tag,
                "shape": {"B": b, "H": h, "S": s, "D": d},
                "device": str(device),
                "torch_ms": {
                    "rmsnorm": rms_ms,
                    "rope": rope_ms,
                    "swiglu": swi_ms,
                    "ce": ce_ms,
                    "attn": attn_ms,
                },
                "unsloth_ok": uns_ok,
                "unsloth_ms": uns_ms,
                "unsloth_errors": uns_err,
            }
        )

    (WORK / "python_meta.json").write_text(
        json.dumps(
            {
                "cuda": torch.cuda.is_available(),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "cap": list(torch.cuda.get_device_capability(0))
                if torch.cuda.is_available()
                else None,
                "torch": torch.__version__,
                "unsloth": uns,
                "cases": cases,
            },
            indent=2,
        )
        + "\n"
    )
    print("wrote", WORK / "python_meta.json")


if __name__ == "__main__":
    main()
