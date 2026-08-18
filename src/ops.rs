// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! Triton-shaped names over whatever backend is honest today.
//!
//! Callers should use this module, not `custom_op::*` / CubeCL / the
//! triton-bridge hook. Internals may change; these names stay.
//!
//! | Name | Today | Not |
//! |------|--------|-----|
//! | [`rmsnorm`] | CustomOp | CubeCL D2H |
//! | [`rope`] | CustomOp | host `to_vec1` |
//! | [`swiglu`] | CustomOp elementwise | fusing the GEMMs |
//! | [`attention`] | CustomOp tiled SRAM FA (CUDA, no extra mask) | Unsloth FA PTX / extra-mask GEMM |
//! | [`cross_entropy`] | chunked CustomOp | full `[N,V]` softmax |
//! | [`fused_linear_ce`] | CPU CustomOp / device vocab tiles | a Triton JIT |

use candle_core::{Result as CandleResult, Tensor};

use crate::kernels::custom_op::{
    attention_device, chunked_cross_entropy, fused_linear_cross_entropy, rmsnorm_custom_op,
    rope_custom_op, swiglu_custom_op, DEFAULT_CE_CHUNK,
};

/// `y = x / rms(x) * weight` over the last dim. f32.
///
/// # Errors
///
/// Dtype/shape errors from the CustomOp.
pub fn rmsnorm(x: &Tensor, weight: &Tensor, eps: f32) -> CandleResult<Tensor> {
    rmsnorm_custom_op(x, weight, eps)
}

/// NeoX half-split RoPE. `x` is `[B,H,S,D]`; `cos`/`sin` are `[S, D/2]`.
///
/// # Errors
///
/// Dtype/shape errors from the CustomOp.
pub fn rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> CandleResult<Tensor> {
    rope_custom_op(x, cos, sin)
}

/// `silu(gate) * up`. Does not fuse the surrounding GEMMs.
///
/// # Errors
///
/// Dtype/shape errors from the CustomOp.
pub fn swiglu(gate: &Tensor, up: &Tensor) -> CandleResult<Tensor> {
    swiglu_custom_op(gate, up)
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
    use candle_core::{Device, Tensor};

    #[test]
    fn names_match_custom_op() {
        let d = Device::Cpu;
        let x = Tensor::randn(0.0f32, 1.0, (2, 8), &d).unwrap();
        let w = Tensor::ones((8,), candle_core::DType::F32, &d).unwrap();
        let y = rmsnorm(&x, &w, 1e-5).unwrap();
        assert_eq!(y.dims(), x.dims());
    }
}
