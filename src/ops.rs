// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! Triton-shaped names over whatever backend is honest today.
//!
//! Callers should use this module, not `custom_op::*` / CubeCL / the
//! triton-bridge hook. Internals may change; these names stay.
//!
//! | Name | Today | Not |
//! |------|--------|-----|
//! | [`rmsnorm`] | CustomOp f32/f16 | CubeCL D2H |
//! | [`layernorm`] | CustomOp mean/var + affine | RMSNorm |
//! | [`rope`] / [`rope_with_ids`] | CustomOp | host `to_vec1` |
//! | [`swiglu`] | CustomOp `silu⊙up` | fusing the GEMMs |
//! | [`geglu`] | CustomOp exact GELU⊙up | tanh-approx GELU |
//! | [`attention`] | CustomOp tiled SRAM FA (CUDA, no extra mask) | Unsloth FA PTX |
//! | [`attention_window`] | same path, sliding-window causal | flex `torch.compile` |
//! | [`attention_softcap`] | tanh softcap then softmax | flex `torch.compile` |
//! | [`cross_entropy`] | chunked CustomOp | full `[N,V]` softmax |
//! | [`fused_linear_ce`] | CPU CustomOp / device vocab tiles | a Triton JIT |

use candle_core::{Result as CandleResult, Tensor};

use crate::kernels::custom_op::{
    attention_device, attention_device_softcap, attention_device_window, chunked_cross_entropy,
    fused_linear_cross_entropy, geglu_custom_op, layernorm_custom_op, rmsnorm_custom_op,
    rope_custom_op, rope_with_position_ids, swiglu_custom_op, DEFAULT_CE_CHUNK,
};

/// `y = x / rms(x) * weight` over the last dim. [`DType::F32`] or [`DType::F16`].
///
/// # Errors
///
/// Dtype/shape errors from the CustomOp.
pub fn rmsnorm(x: &Tensor, weight: &Tensor, eps: f32) -> CandleResult<Tensor> {
    rmsnorm_custom_op(x, weight, eps)
}

/// Last-dim LayerNorm: `(x - mean) / sqrt(var + eps) * weight + bias`.
///
/// # Errors
///
/// Dtype/shape errors from the CustomOp.
pub fn layernorm(x: &Tensor, weight: &Tensor, bias: &Tensor, eps: f32) -> CandleResult<Tensor> {
    layernorm_custom_op(x, weight, bias, eps)
}

/// NeoX half-split RoPE. `x` is `[B,H,S,D]`; `cos`/`sin` are `[S, D/2]`.
///
/// # Errors
///
/// Dtype/shape errors from the CustomOp.
pub fn rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> CandleResult<Tensor> {
    rope_custom_op(x, cos, sin)
}

/// RoPE with packed `position_ids` (`[S]` or `[B,S]` i64) over a `[max, D/2]` cache.
///
/// # Errors
///
/// Shape/dtype errors from the gather or CustomOp.
pub fn rope_with_ids(
    x: &Tensor,
    cos_cache: &Tensor,
    sin_cache: &Tensor,
    position_ids: &Tensor,
) -> CandleResult<Tensor> {
    rope_with_position_ids(x, cos_cache, sin_cache, position_ids)
}

/// `silu(gate) * up`. Does not fuse the surrounding GEMMs.
///
/// # Errors
///
/// Dtype/shape errors from the CustomOp.
pub fn swiglu(gate: &Tensor, up: &Tensor) -> CandleResult<Tensor> {
    swiglu_custom_op(gate, up)
}

/// Exact GeGLU: `gelu(gate) * up` with `gelu(e) = 0.5 e (1 + erf(e/√2))`.
///
/// # Errors
///
/// Dtype/shape errors from the CustomOp.
pub fn geglu(gate: &Tensor, up: &Tensor) -> CandleResult<Tensor> {
    geglu_custom_op(gate, up)
}

/// Causal/non-causal attention. Extra `mask` still uses Candle GEMM+softmax.
///
/// CUDA default (no extra mask, head dim ≤ 128) is SRAM-tiled Flash-style
/// softmax. Scale is typically `1/sqrt(head_dim)`.
///
/// # Errors
///
/// Shape/dtype errors from [`attention_device`].
pub fn attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    mask: Option<&Tensor>,
    causal: bool,
) -> CandleResult<Tensor> {
    attention_device(q, k, v, scale, mask, causal)
}

/// Sliding-window attention. `window == 0` is full [`attention`] (no extra mask).
///
/// Causal: each query attends to at most `window` keys ending at itself.
/// CUDA tiled path honors the window (owned NVRTC, not Unsloth PTX).
///
/// # Errors
///
/// Shape/dtype errors from [`attention_device_window`].
pub fn attention_window(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    causal: bool,
    window: usize,
) -> CandleResult<Tensor> {
    attention_device_window(q, k, v, scale, causal, window)
}

/// Attention with tanh score softcap (`score = cap * tanh(score / cap)`).
///
/// `softcap <= 0` is identical to [`attention`]. Gemma-style logits softcap.
///
/// # Errors
///
/// Shape/dtype errors from [`attention_device`].
pub fn attention_softcap(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    mask: Option<&Tensor>,
    causal: bool,
    softcap: f32,
) -> CandleResult<Tensor> {
    attention_device_softcap(q, k, v, scale, mask, causal, softcap)
}

/// Mean cross-entropy, ignore index `-100` by default. Vocab is chunked.
///
/// # Errors
///
/// Dtype/shape errors from chunked CE.
pub fn cross_entropy(logits: &Tensor, targets: &Tensor) -> CandleResult<Tensor> {
    chunked_cross_entropy(logits, targets, -100, DEFAULT_CE_CHUNK)
}

/// `softmax(hidden @ weight.T)` CE without a full `[N, V]` logits tensor.
///
/// # Errors
///
/// Dtype/shape errors from fused linear+CE.
pub fn fused_linear_ce(
    hidden: &Tensor,
    weight: &Tensor,
    targets: &Tensor,
    chunk_size: usize,
) -> CandleResult<Tensor> {
    fused_linear_cross_entropy(hidden, weight, targets, -100, chunk_size)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    #[test]
    fn names_match_custom_op() {
        let d = Device::Cpu;
        let x = Tensor::randn(0.0f32, 1.0, (2, 8), &d).unwrap();
        let w = Tensor::ones((8,), candle_core::DType::F32, &d).unwrap();
        let y = rmsnorm(&x, &w, 1e-5).unwrap();
        assert_eq!(y.dims(), x.dims());
        let b = Tensor::zeros((8,), candle_core::DType::F32, &d).unwrap();
        let ln = layernorm(&x, &w, &b, 1e-5).unwrap();
        assert_eq!(ln.dims(), x.dims());
        let xf = x.to_dtype(DType::F16).unwrap();
        let wf = w.to_dtype(DType::F16).unwrap();
        assert_eq!(rmsnorm(&xf, &wf, 1e-5).unwrap().dtype(), DType::F16);
        let g = Tensor::randn(0.0f32, 1.0, (2, 8), &d).unwrap();
        let u = Tensor::randn(0.0f32, 1.0, (2, 8), &d).unwrap();
        assert_eq!(geglu(&g, &u).unwrap().dims(), g.dims());
    }

    #[test]
    fn attention_window_is_the_public_name() {
        let d = Device::Cpu;
        let q = Tensor::randn(0.0f32, 1.0, (1, 1, 8, 4), &d).unwrap();
        let k = Tensor::randn(0.0f32, 1.0, (1, 1, 8, 4), &d).unwrap();
        let v = Tensor::randn(0.0f32, 1.0, (1, 1, 8, 4), &d).unwrap();
        let y = attention_window(&q, &k, &v, 0.5, true, 3).unwrap();
        assert_eq!(y.dims(), q.dims());
    }

    fn mae(a: &Tensor, b: &Tensor) -> f32 {
        (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .mean_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }

    #[test]
    fn public_ops_mae_vs_cpu_ref() {
        let d = Device::Cpu;
        let gate = Tensor::randn(0.0f32, 1.0, (2, 8), &d).unwrap();
        let up = Tensor::randn(0.0f32, 1.0, (2, 8), &d).unwrap();
        let silu = candle_nn::ops::silu(&gate).unwrap().mul(&up).unwrap();
        assert!(mae(&swiglu(&gate, &up).unwrap(), &silu) < 1e-6);

        let q = Tensor::randn(0.0f32, 1.0, (1, 2, 8, 8), &d).unwrap();
        let k = Tensor::randn(0.0f32, 1.0, (1, 2, 8, 8), &d).unwrap();
        let v = Tensor::randn(0.0f32, 1.0, (1, 2, 8, 8), &d).unwrap();
        let scale = 1.0f64 / 8.0f64.sqrt();
        let scores = q
            .matmul(&k.transpose(2, 3).unwrap().contiguous().unwrap())
            .unwrap();
        let ref_attn = candle_nn::ops::softmax(&(scores * scale).unwrap(), 3)
            .unwrap()
            .matmul(&v)
            .unwrap();
        assert!(
            mae(
                &attention(&q, &k, &v, scale, None, false).unwrap(),
                &ref_attn
            ) < 1e-5
        );

        let half = 4usize;
        let s = 8usize;
        let cos = Tensor::randn(0.0f32, 1.0, (s, half), &d).unwrap();
        let sin = Tensor::randn(0.0f32, 1.0, (s, half), &d).unwrap();
        let x1 = q.narrow(3, 0, half).unwrap();
        let x2 = q.narrow(3, half, half).unwrap();
        let c = cos.reshape((1, 1, s, half)).unwrap();
        let n = sin.reshape((1, 1, s, half)).unwrap();
        let rope_ref = Tensor::cat(
            &[
                x1.broadcast_mul(&c)
                    .unwrap()
                    .sub(&x2.broadcast_mul(&n).unwrap())
                    .unwrap(),
                x2.broadcast_mul(&c)
                    .unwrap()
                    .add(&x1.broadcast_mul(&n).unwrap())
                    .unwrap(),
            ],
            3,
        )
        .unwrap();
        assert!(mae(&rope(&q, &cos, &sin).unwrap(), &rope_ref) < 1e-5);

        let logits = Tensor::randn(0.0f32, 1.0, (4, 8), &d).unwrap();
        let targets = Tensor::from_vec(vec![0i64, 3, 7, 1], (4,), &d).unwrap();
        let log_sm = candle_nn::ops::log_softmax(&logits, 1).unwrap();
        let nll = log_sm
            .gather(&targets.reshape((4, 1)).unwrap(), 1)
            .unwrap()
            .neg()
            .unwrap()
            .mean_all()
            .unwrap();
        let ce = cross_entropy(&logits, &targets).unwrap();
        assert!(mae(&ce, &nll) < 1e-5);
    }
}
