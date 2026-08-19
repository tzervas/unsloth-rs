#!/usr/bin/env python3
"""Write shared f32 fixtures and Python-side outputs (torch + unsloth if present)."""

from __future__ import annotations

import importlib
import json
import math
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
WARMUP = 5
N_SAMPLES = 100
# C-COMPARE-NEWOPS + C-UNS-SHAPES. B=2 H=8 D=64. s2048 is FAIL_ENV on OOM, not green skip.
SHAPES = ((2, 8, 128, 64), (2, 8, 512, 64), (2, 8, 2048, 64))
WINDOW = 64
SOFTCAP = 50.0
POS_OFFSET = 8
CACHE_PAD = 16


def save(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    a = np.ascontiguousarray(arr.astype(np.float32, copy=False))
    np.save(path, a)
    a.tofile(path.with_suffix(".f32"))


def torch_rmsnorm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)
    return x / rms * w


def torch_layernorm(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.layer_norm(x, (x.shape[-1],), w, b, EPS)


def torch_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x [B,H,S,D]; cos/sin [S, D/2]. NeoX half-split (matches unsloth-rs CustomOp).
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def torch_rope_with_ids(
    x: torch.Tensor, cos_cache: torch.Tensor, sin_cache: torch.Tensor, ids: torch.Tensor
) -> torch.Tensor:
    cos = cos_cache.index_select(0, ids)
    sin = sin_cache.index_select(0, ids)
    return torch_rope(x, cos, sin)


def torch_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return F.silu(gate) * up


def torch_geglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    # Exact GELU: 0.5 * e * (1 + erf(e/√2)). Not tanh-approx.
    gelu = 0.5 * gate * (1.0 + torch.erf(gate / math.sqrt(2.0)))
    return gelu * up


def torch_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, targets, ignore_index=-100)


def torch_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    scale = q.shape[-1] ** -0.5
    return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)


def _causal_window_mask(seq: int, window: int, device: torch.device) -> torch.Tensor:
    """Keys in [q - window + 1, q], matching unsloth-rs window_lo / window_hi_excl."""
    q_idx = torch.arange(seq, device=device)
    k_idx = torch.arange(seq, device=device)
    lo = q_idx.unsqueeze(-1) - (window - 1)
    return (k_idx <= q_idx.unsqueeze(-1)) & (k_idx >= lo)


def torch_attn_window(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window: int
) -> torch.Tensor:
    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    allowed = _causal_window_mask(q.shape[-2], window, q.device)
    scores = scores.masked_fill(~allowed, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, v)


def torch_attn_softcap(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cap: float
) -> torch.Tensor:
    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    scores = cap * torch.tanh(scores / cap)
    q_idx = torch.arange(q.shape[-2], device=q.device)
    k_idx = torch.arange(q.shape[-2], device=q.device)
    causal = k_idx <= q_idx.unsqueeze(-1)
    scores = scores.masked_fill(~causal, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, v)


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
        "unsloth.kernels.layernorm",
        "unsloth.kernels.rope_embedding",
        "unsloth.kernels.swiglu",
        "unsloth.kernels.geglu",
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
    info["ops"]["attn_window"] = (
        "no standalone Unsloth attn kernel; window not compared (not invented)"
    )
    info["ops"]["attn_softcap"] = (
        "no standalone Unsloth attn kernel; softcap not compared (not invented)"
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


def try_unsloth_layernorm(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor):
    def go():
        from unsloth.kernels.layernorm import fast_layernorm

        ln = torch.nn.LayerNorm(
            x.shape[-1], eps=EPS, elementwise_affine=True, device=x.device, dtype=x.dtype
        )
        with torch.no_grad():
            ln.weight.copy_(w)
            ln.bias.copy_(b)
        return fast_layernorm(ln, x)

    return _call("layernorm", go)


def try_unsloth_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # fast_rope_embedding(Q, K, cos, sin): Q/K [B,H,S,D]; returns (Q_out, K_out).
    def go():
        from unsloth.kernels.rope_embedding import fast_rope_embedding

        q_out, _k_out = fast_rope_embedding(q, k, cos, sin)
        return q_out

    return _call("rope", go)


def try_unsloth_rope_ids(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    ids: torch.Tensor,
):
    def go():
        from unsloth.kernels.rope_embedding import fast_rope_embedding

        # 2026.8.18 `rope_embedding_indices` uses Fast_RoPE_Embedding_QK, which
        # is inplace on contiguous Q/K and illegal-memory-accessed on this
        # NeoX [max, D/2] cache. Probe Unsloth's sequential rope on gathered
        # tables instead. Do not invent an attn kernel.
        cos_g = cos_cache.index_select(0, ids)
        sin_g = sin_cache.index_select(0, ids)
        q_out, _k_out = fast_rope_embedding(q, k, cos_g, sin_g)
        return q_out

    return _call("rope_with_ids", go)


def try_unsloth_swiglu(gate: torch.Tensor, up: torch.Tensor):
    # swiglu_fg_kernel(e, g): [B,S,D] -> silu(e)*g
    def go():
        from unsloth.kernels.swiglu import swiglu_fg_kernel

        return swiglu_fg_kernel(gate, up)

    return _call("swiglu", go)


def try_unsloth_geglu(gate: torch.Tensor, up: torch.Tensor):
    def go():
        from unsloth.kernels.geglu import geglu_exact_forward_kernel

        return geglu_exact_forward_kernel(gate, up)

    return _call("geglu", go)


def try_unsloth_ce(logits_b_s_v: torch.Tensor, labels_b_s: torch.Tensor):
    # fast_cross_entropy_loss: logits [B,S,V], labels [B,S] -> mean scalar
    def go():
        from unsloth.kernels.cross_entropy_loss import fast_cross_entropy_loss

        return fast_cross_entropy_loss(logits_b_s_v, labels_b_s)

    return _call("ce", go)


def sorted_percentile(samples: list[float], p: float) -> float:
    """Match unsloth_rs::kernels::custom_op::sorted_percentile."""
    n = len(samples)
    if n == 0:
        return float("nan")
    ordered = sorted(samples)
    if abs(p - 0.50) < 1e-12:
        if n % 2 == 1:
            return ordered[n // 2]
        return 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])
    idx = min(int(math.ceil(p * (n - 1))), n - 1)
    return ordered[idx]


def timed(fn, device: torch.device) -> tuple[object, dict]:
    """Warmup then n host+sync samples. Returns last output + p50/p99 (ms)."""
    if device.type == "cuda":
        for _ in range(WARMUP):
            out = fn()
        torch.cuda.synchronize()
    else:
        out = None
    samples: list[float] = []
    for _ in range(N_SAMPLES):
        t0 = time.perf_counter()
        out = fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return out, {
        "p50": sorted_percentile(samples, 0.50),
        "p99": sorted_percentile(samples, 0.99),
        "n": len(samples),
    }


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
        # Extra fixtures after the original randn stream so s128/s512 tensors stay bit-stable.
        bias = torch.randn(d, device=device, dtype=torch.float32)
        cache_s = s + CACHE_PAD
        pos_c = torch.arange(cache_s, device=device, dtype=torch.float32)
        freqs_c = torch.outer(pos_c, inv)
        cos_cache = freqs_c.cos()
        sin_cache = freqs_c.sin()
        ids = torch.arange(s, device=device, dtype=torch.int64) + POS_OFFSET

        save(root / "x.npy", x.detach().cpu().numpy())
        save(root / "w.npy", w.detach().cpu().numpy())
        save(root / "bias.npy", bias.detach().cpu().numpy())
        save(root / "q.npy", q.detach().cpu().numpy())
        save(root / "k.npy", k.detach().cpu().numpy())
        save(root / "v.npy", v.detach().cpu().numpy())
        save(root / "gate.npy", gate.detach().cpu().numpy())
        save(root / "up.npy", up.detach().cpu().numpy())
        save(root / "logits.npy", logits.detach().cpu().numpy())
        save(root / "cos.npy", cos.detach().cpu().numpy())
        save(root / "sin.npy", sin.detach().cpu().numpy())
        save(root / "cos_cache.npy", cos_cache.detach().cpu().numpy())
        save(root / "sin_cache.npy", sin_cache.detach().cpu().numpy())
        tgt = targets.detach().cpu().numpy().astype(np.int64)
        np.save(root / "targets.npy", tgt)
        tgt.tofile(root / "targets.i64")
        ids_np = ids.detach().cpu().numpy().astype(np.int64)
        np.save(root / "pos_ids.npy", ids_np)
        ids_np.tofile(root / "pos_ids.i64")
        (root / "shape.json").write_text(
            json.dumps(
                {
                    "B": b,
                    "H": h,
                    "S": s,
                    "D": d,
                    "CACHE_S": cache_s,
                    "WINDOW": WINDOW,
                    "SOFTCAP": SOFTCAP,
                    "POS_OFFSET": POS_OFFSET,
                },
                separators=(",", ":"),
            )
            + "\n"
        )

        rms, rms_ms = timed(lambda: torch_rmsnorm(x, w), device)
        ln, ln_ms = timed(lambda: torch_layernorm(x, w, bias), device)
        rope, rope_ms = timed(lambda: torch_rope(q, cos, sin), device)
        rope_ids, rope_ids_ms = timed(
            lambda: torch_rope_with_ids(q, cos_cache, sin_cache, ids), device
        )
        swi, swi_ms = timed(lambda: torch_swiglu(gate, up), device)
        geg, geg_ms = timed(lambda: torch_geglu(gate, up), device)
        ce, ce_ms = timed(lambda: torch_ce(logits, targets), device)
        attn, attn_ms = timed(lambda: torch_attn(q, k, v), device)
        attn_w, attn_w_ms = timed(lambda: torch_attn_window(q, k, v, WINDOW), device)
        attn_c, attn_c_ms = timed(lambda: torch_attn_softcap(q, k, v, SOFTCAP), device)
        save(root / "torch_rmsnorm.npy", rms.detach().cpu().numpy())
        save(root / "torch_layernorm.npy", ln.detach().cpu().numpy())
        save(root / "torch_rope.npy", rope.detach().cpu().numpy())
        save(root / "torch_rope_with_ids.npy", rope_ids.detach().cpu().numpy())
        save(root / "torch_swiglu.npy", swi.detach().cpu().numpy())
        save(root / "torch_geglu.npy", geg.detach().cpu().numpy())
        save(root / "torch_ce.npy", np.array([float(ce.detach().cpu())], dtype=np.float32))
        save(root / "torch_attn.npy", attn.detach().cpu().numpy())
        save(root / "torch_attn_window.npy", attn_w.detach().cpu().numpy())
        save(root / "torch_attn_softcap.npy", attn_c.detach().cpu().numpy())

        uns_err: dict[str, str] = {}
        uns_ms: dict[str, dict] = {}
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
            try:
                # Clone Q/K: some Unsloth rope paths are inplace.
                q_u, k_u = q.detach().clone(), k.detach().clone()

                def probe_timed(name, first, again):
                    out, err = first()
                    if out is None:
                        record(name, None, err)
                        return
                    try:
                        out, ms = timed(again, device)
                        record(name, out, None, ms)
                    except Exception as e:
                        record(name, None, f"{name} timed: {type(e).__name__}: {e}")

                probe_timed(
                    "rmsnorm",
                    lambda: try_unsloth_rmsnorm(x, w),
                    lambda: try_unsloth_rmsnorm(x, w)[0],
                )
                probe_timed(
                    "layernorm",
                    lambda: try_unsloth_layernorm(x, w, bias),
                    lambda: try_unsloth_layernorm(x, w, bias)[0],
                )
                probe_timed(
                    "rope",
                    lambda: try_unsloth_rope(q_u, k_u, cos, sin),
                    lambda: try_unsloth_rope(q_u, k_u, cos, sin)[0],
                )
                probe_timed(
                    "rope_with_ids",
                    lambda: try_unsloth_rope_ids(q_u, k_u, cos_cache, sin_cache, ids),
                    lambda: try_unsloth_rope_ids(q_u, k_u, cos_cache, sin_cache, ids)[0],
                )
                probe_timed(
                    "swiglu",
                    lambda: try_unsloth_swiglu(gate, up),
                    lambda: try_unsloth_swiglu(gate, up)[0],
                )
                probe_timed(
                    "geglu",
                    lambda: try_unsloth_geglu(gate, up),
                    lambda: try_unsloth_geglu(gate, up)[0],
                )
                logits_b = logits.view(b, s, 128)
                labels_b = targets.view(b, s)
                probe_timed(
                    "ce",
                    lambda: try_unsloth_ce(logits_b, labels_b),
                    lambda: try_unsloth_ce(logits_b, labels_b)[0],
                )
                if device.type == "cuda":
                    torch.cuda.synchronize()
            except Exception as e:
                uns_err["block"] = f"{type(e).__name__}: {e}"
                # Illegal address poisons the context; skip Unsloth on later shapes.
                uns["importable"] = False

        cases.append(
            {
                "tag": tag,
                "shape": {
                    "B": b,
                    "H": h,
                    "S": s,
                    "D": d,
                    "CACHE_S": cache_s,
                    "WINDOW": WINDOW,
                    "SOFTCAP": SOFTCAP,
                },
                "device": str(device),
                "torch_ms": {
                    "rmsnorm": rms_ms,
                    "layernorm": ln_ms,
                    "rope": rope_ms,
                    "rope_with_ids": rope_ids_ms,
                    "swiglu": swi_ms,
                    "geglu": geg_ms,
                    "ce": ce_ms,
                    "attn": attn_ms,
                    "attn_window": attn_w_ms,
                    "attn_softcap": attn_c_ms,
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
                "timing": {"warmup": WARMUP, "n_samples": N_SAMPLES, "unit": "ms"},
                "compare_constants": {
                    "window": WINDOW,
                    "softcap": SOFTCAP,
                    "pos_offset": POS_OFFSET,
                    "cache_pad": CACHE_PAD,
                },
                "cases": cases,
            },
            indent=2,
        )
        + "\n"
    )
    print("wrote", WORK / "python_meta.json")


if __name__ == "__main__":
    main()
