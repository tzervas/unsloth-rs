// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! Device-resident RoPE apply via Candle [`CustomOp3`].
//!
//! `x`: `[B, H, S, D]`, `cos`/`sin`: `[S, D/2]`. Half-rotate along last dim.

use candle_core::{CpuStorage, CustomOp3, DType, Layout, Result as CandleResult, Shape, Tensor};

/// Apply rotary embedding (sequential positions `0..S`).
#[derive(Clone, Debug, Default)]
pub struct RopeOp;

/// Apply RoPE. f32 only. `position_ids` are **not** consumed — caller must
/// narrow or gather `cos`/`sin` first (keeps current `RotaryEmbedding` semantics).
///
/// # Errors
///
/// Rank/shape/dtype mismatch or backend error.
pub fn rope_custom_op(x: &Tensor, cos: &Tensor, sin: &Tensor) -> CandleResult<Tensor> {
    if x.dtype() != DType::F32 || cos.dtype() != DType::F32 || sin.dtype() != DType::F32 {
        candle_core::bail!(
            "CustomOp RoPE is f32-only (got x={:?} cos={:?} sin={:?})",
            x.dtype(),
            cos.dtype(),
            sin.dtype()
        );
    }
    if x.rank() != 4 {
        candle_core::bail!("RoPE x must be [B,H,S,D], got {:?}", x.shape());
    }
    let dims = x.dims();
    let seq = dims[2];
    let head = dims[3];
    if !head.is_multiple_of(2) {
        candle_core::bail!("RoPE head_dim must be even, got {head}");
    }
    let half = head / 2;
    if cos.dims() != [seq, half] || sin.dims() != [seq, half] {
        candle_core::bail!(
            "RoPE cos/sin must be [{seq}, {half}], got {:?} / {:?}",
            cos.shape(),
            sin.shape()
        );
    }
    let x = x.contiguous()?;
    let cos = cos.contiguous()?;
    let sin = sin.contiguous()?;
    x.apply_op3(&cos, &sin, RopeOp)
}

fn cpu_rope(
    x: &[f32],
    cos: &[f32],
    sin: &[f32],
    batch: usize,
    heads: usize,
    seq: usize,
    dim: usize,
) -> Vec<f32> {
    let half = dim / 2;
    let mut out = vec![0.0f32; batch * heads * seq * dim];
    for batch_i in 0..batch {
        for head_i in 0..heads {
            for seq_i in 0..seq {
                let base = ((batch_i * heads + head_i) * seq + seq_i) * dim;
                let cbase = seq_i * half;
                for pair in 0..half {
                    let x1 = x[base + pair];
                    let x2 = x[base + half + pair];
                    let c = cos[cbase + pair];
                    let sn = sin[cbase + pair];
                    out[base + pair] = x1 * c - x2 * sn;
                    out[base + half + pair] = x2 * c + x1 * sn;
                }
            }
        }
    }
    out
}

impl CustomOp3 for RopeOp {
    fn name(&self) -> &'static str {
        "unsloth_rope"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> CandleResult<(CpuStorage, Shape)> {
        let x_span = l1.contiguous_offsets().ok_or_else(|| {
            candle_core::Error::Msg("RoPE CustomOp: x must be contiguous".into()).bt()
        })?;
        let cos_span = l2.contiguous_offsets().ok_or_else(|| {
            candle_core::Error::Msg("RoPE CustomOp: cos must be contiguous".into()).bt()
        })?;
        let sin_span = l3.contiguous_offsets().ok_or_else(|| {
            candle_core::Error::Msg("RoPE CustomOp: sin must be contiguous".into()).bt()
        })?;
        let dims = l1.dims();
        if dims.len() != 4 {
            candle_core::bail!("RoPE storage rank {}", dims.len());
        }
        let x = &s1.as_slice::<f32>()?[x_span.0..x_span.1];
        let cos = &s2.as_slice::<f32>()?[cos_span.0..cos_span.1];
        let sin = &s3.as_slice::<f32>()?[sin_span.0..sin_span.1];
        let out = cpu_rope(x, cos, sin, dims[0], dims[1], dims[2], dims[3]);
        Ok((CpuStorage::F32(out), l1.shape().clone()))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle_core::CudaStorage,
        l1: &Layout,
        s2: &candle_core::CudaStorage,
        l2: &Layout,
        s3: &candle_core::CudaStorage,
        l3: &Layout,
    ) -> CandleResult<(candle_core::CudaStorage, Shape)> {
        cuda_rope(s1, l1, s2, l2, s3, l3)
    }

    fn bwd(
        &self,
        _x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        _y: &Tensor,
        gy: &Tensor,
    ) -> CandleResult<(Option<Tensor>, Option<Tensor>, Option<Tensor>)> {
        // RoPE is orthogonal: dx = rope(gy, cos, -sin). Caches do not take grad.
        let neg_sin = sin.neg()?;
        let dx = rope_custom_op(gy, cos, &neg_sin)?;
        Ok((Some(dx), None, None))
    }
}

#[cfg(feature = "cuda")]
fn cuda_rope(
    sx: &candle_core::CudaStorage,
    lx: &Layout,
    sc: &candle_core::CudaStorage,
    lc: &Layout,
    ss: &candle_core::CudaStorage,
    ls: &Layout,
) -> CandleResult<(candle_core::CudaStorage, Shape)> {
    use super::nvrtc::load_func;
    use candle_core::cuda::{cudarc, CudaStorage, WrapErr};
    use cudarc::driver::{LaunchConfig, PushKernelArg};

    let (a, b) = lx
        .contiguous_offsets()
        .ok_or_else(|| candle_core::Error::Msg("RoPE CUDA: x must be contiguous".into()).bt())?;
    let (c, d) = lc
        .contiguous_offsets()
        .ok_or_else(|| candle_core::Error::Msg("RoPE CUDA: cos must be contiguous".into()).bt())?;
    let (e, f) = ls
        .contiguous_offsets()
        .ok_or_else(|| candle_core::Error::Msg("RoPE CUDA: sin must be contiguous".into()).bt())?;
    let dims = lx.dims();
    let (batch, heads, seq, dim) = (dims[0], dims[1], dims[2], dims[3]);
    let rows = batch * heads * seq;
    let half = dim / 2;
    let dev = sx.device.clone();
    let x = sx.as_cuda_slice::<f32>()?.slice(a..b);
    let cos = sc.as_cuda_slice::<f32>()?.slice(c..d);
    let sin = ss.as_cuda_slice::<f32>()?.slice(e..f);
    let mut y = unsafe { dev.alloc::<f32>(b - a) }?;
    let func = load_func(&dev, "rope_f32", "unsloth_rope_f32", ROPE_SRC)?;
    let block = 128u32;
    let cfg = LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    let half_i = half as i32;
    let seq_i = seq as i32;
    let stream = dev.cuda_stream();
    let mut builder = stream.launch_builder(&func);
    builder.arg(&x);
    builder.arg(&cos);
    builder.arg(&sin);
    builder.arg(&y);
    builder.arg(&half_i);
    builder.arg(&seq_i);
    unsafe { builder.launch(cfg) }.w()?;
    Ok((CudaStorage::wrap_cuda_slice(y, dev), lx.shape().clone()))
}

#[cfg(feature = "cuda")]
const ROPE_SRC: &str = r#"
extern "C" __global__ void rope_f32(
    const float* __restrict__ x,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    float* __restrict__ y,
    int half,
    int seq
) {
    int row = (int)blockIdx.x;
    int tid = (int)threadIdx.x;
    int s = row % seq;
    const float* xr = x + (size_t)row * (size_t)(2 * half);
    float* yr = y + (size_t)row * (size_t)(2 * half);
    const float* cr = cos + (size_t)s * (size_t)half;
    const float* sr = sin + (size_t)s * (size_t)half;
    for (int i = tid; i < half; i += (int)blockDim.x) {
        float x1 = xr[i];
        float x2 = xr[half + i];
        float c = cr[i];
        float sn = sr[i];
        yr[i] = x1 * c - x2 * sn;
        yr[half + i] = x2 * c + x1 * sn;
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn reference(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Tensor {
        let half = x.dim(3).unwrap() / 2;
        let x1 = x.narrow(3, 0, half).unwrap();
        let x2 = x.narrow(3, half, half).unwrap();
        let y1 = (x1.broadcast_mul(cos).unwrap() - x2.broadcast_mul(sin).unwrap()).unwrap();
        let y2 = (x2.broadcast_mul(cos).unwrap() + x1.broadcast_mul(sin).unwrap()).unwrap();
        Tensor::cat(&[&y1, &y2], 3).unwrap()
    }

    #[test]
    fn matches_split_rotate() {
        let d = Device::Cpu;
        let x = Tensor::randn(0.0f32, 1.0, (2, 3, 5, 8), &d).unwrap();
        let cos = Tensor::randn(0.0f32, 0.2, (5, 4), &d).unwrap();
        let sin = Tensor::randn(0.0f32, 0.2, (5, 4), &d).unwrap();
        let y = rope_custom_op(&x, &cos, &sin).unwrap();
        let r = reference(&x, &cos, &sin);
        let mae = (y - r)
            .unwrap()
            .abs()
            .unwrap()
            .mean_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(mae < 1e-6, "mae={mae}");
    }

    #[test]
    fn rejects_odd_head() {
        let d = Device::Cpu;
        let x = Tensor::zeros((1, 1, 2, 5), DType::F32, &d).unwrap();
        let cos = Tensor::zeros((2, 2), DType::F32, &d).unwrap();
        let sin = Tensor::zeros((2, 2), DType::F32, &d).unwrap();
        assert!(rope_custom_op(&x, &cos, &sin).is_err());
    }
}
