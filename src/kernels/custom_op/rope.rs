// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! Device-resident RoPE apply via Candle [`CustomOp3`].
//!
//! `x`: `[B, H, S, D]`. `cos`/`sin` are either `[S, D/2]` (same positions for
//! every batch row) or `[B, S, D/2]` (per-row gather — packed / `position_ids`).

use candle_core::{CpuStorage, CustomOp3, DType, Layout, Result as CandleResult, Shape, Tensor};

/// Apply rotary embedding.
#[derive(Clone, Debug, Default)]
pub struct RopeOp {
    /// `true` when cos/sin are `[B, S, D/2]` instead of `[S, D/2]`.
    pub batched_cache: bool,
}

/// Apply RoPE. f32 only.
///
/// `cos` / `sin`: `[S, D/2]` or `[B, S, D/2]`. For `position_ids`, prefer
/// [`rope_with_position_ids`].
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
    let batch = dims[0];
    let seq = dims[2];
    let head = dims[3];
    if !head.is_multiple_of(2) {
        candle_core::bail!("RoPE head_dim must be even, got {head}");
    }
    let half = head / 2;
    let batched_cache = match cos.rank() {
        2 => {
            if cos.dims() != [seq, half] || sin.dims() != [seq, half] {
                candle_core::bail!(
                    "RoPE cos/sin must be [{seq}, {half}], got {:?} / {:?}",
                    cos.shape(),
                    sin.shape()
                );
            }
            false
        }
        3 => {
            if cos.dims() != [batch, seq, half] || sin.dims() != [batch, seq, half] {
                candle_core::bail!(
                    "RoPE batched cos/sin must be [{batch}, {seq}, {half}], got {:?} / {:?}",
                    cos.shape(),
                    sin.shape()
                );
            }
            true
        }
        _ => candle_core::bail!("RoPE cos rank {} (want 2 or 3)", cos.rank()),
    };
    let x = x.contiguous()?;
    let cos = cos.contiguous()?;
    let sin = sin.contiguous()?;
    x.apply_op3(&cos, &sin, RopeOp { batched_cache })
}

/// Apply RoPE using a full cache and `position_ids`.
///
/// `cos_cache` / `sin_cache`: `[max_seq, D/2]`.
/// `position_ids`: i64 `[S]` or `[B, S]`. Device-resident gather (`index_select`).
///
/// # Errors
///
/// Shape/dtype mismatch or out-of-range ids (Candle `index_select`).
pub fn rope_with_position_ids(
    x: &Tensor,
    cos_cache: &Tensor,
    sin_cache: &Tensor,
    position_ids: &Tensor,
) -> CandleResult<Tensor> {
    if x.rank() != 4 {
        candle_core::bail!("RoPE x must be [B,H,S,D], got {:?}", x.shape());
    }
    if cos_cache.rank() != 2 || sin_cache.rank() != 2 {
        candle_core::bail!(
            "RoPE cache must be [max, D/2], got {:?} / {:?}",
            cos_cache.shape(),
            sin_cache.shape()
        );
    }
    let batch = x.dim(0)?;
    let seq = x.dim(2)?;
    let half = x.dim(3)? / 2;
    if cos_cache.dim(1)? != half || sin_cache.dim(1)? != half {
        candle_core::bail!("RoPE cache last dim {} vs head/2 {half}", cos_cache.dim(1)?);
    }
    let ids = position_ids.to_dtype(DType::I64)?.contiguous()?;
    let (cos, sin) = gather_rope_cache(cos_cache, sin_cache, &ids, batch, seq)?;
    rope_custom_op(x, &cos, &sin)
}

fn gather_rope_cache(
    cos_cache: &Tensor,
    sin_cache: &Tensor,
    ids: &Tensor,
    batch: usize,
    seq: usize,
) -> CandleResult<(Tensor, Tensor)> {
    match ids.rank() {
        1 => {
            if ids.dim(0)? != seq {
                candle_core::bail!("position_ids [S] len {} != seq {seq}", ids.dim(0)?);
            }
            let cos = cos_cache.index_select(ids, 0)?;
            let sin = sin_cache.index_select(ids, 0)?;
            Ok((cos, sin))
        }
        2 => {
            if ids.dims() != [batch, seq] {
                candle_core::bail!(
                    "position_ids must be [{batch}, {seq}], got {:?}",
                    ids.shape()
                );
            }
            let flat = ids.flatten_all()?;
            let cos = cos_cache
                .index_select(&flat, 0)?
                .reshape((batch, seq, cos_cache.dim(1)?))?;
            let sin = sin_cache
                .index_select(&flat, 0)?
                .reshape((batch, seq, sin_cache.dim(1)?))?;
            Ok((cos, sin))
        }
        rank => candle_core::bail!("position_ids rank {rank} (want 1 or 2)"),
    }
}

struct RopeGeom {
    batch: usize,
    heads: usize,
    seq: usize,
    dim: usize,
    batched_cache: bool,
}

fn cpu_rope(x: &[f32], cos: &[f32], sin: &[f32], geom: RopeGeom) -> Vec<f32> {
    let RopeGeom {
        batch,
        heads,
        seq,
        dim,
        batched_cache,
    } = geom;
    let half = dim / 2;
    let mut out = vec![0.0f32; batch * heads * seq * dim];
    for batch_i in 0..batch {
        for head_i in 0..heads {
            for seq_i in 0..seq {
                let base = ((batch_i * heads + head_i) * seq + seq_i) * dim;
                let cbase = if batched_cache {
                    (batch_i * seq + seq_i) * half
                } else {
                    seq_i * half
                };
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
        let out = cpu_rope(
            x,
            cos,
            sin,
            RopeGeom {
                batch: dims[0],
                heads: dims[1],
                seq: dims[2],
                dim: dims[3],
                batched_cache: self.batched_cache,
            },
        );
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
        cuda_rope(s1, l1, s2, l2, s3, l3, self.batched_cache)
    }

    fn bwd(
        &self,
        _x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        _y: &Tensor,
        gy: &Tensor,
    ) -> CandleResult<(Option<Tensor>, Option<Tensor>, Option<Tensor>)> {
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
    batched_cache: bool,
) -> CandleResult<(candle_core::CudaStorage, Shape)> {
    use super::nvrtc::{alloc_f32, launch, launch_config, load_func};
    use candle_core::cuda::cudarc::driver::PushKernelArg;
    use candle_core::cuda::CudaStorage;

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
    let y = alloc_f32(&dev, b - a)?;
    if rows == 0 {
        return Ok((CudaStorage::wrap_cuda_slice(y, dev), lx.shape().clone()));
    }
    let func = load_func(&dev, "rope_f32", "unsloth_rope_f32", ROPE_SRC)?;
    let cfg = launch_config(rows, 128, 0)?;
    let half_i = i32::try_from(half)
        .map_err(|_| candle_core::Error::Msg(format!("RoPE half {half} exceeds i32")).bt())?;
    let seq_i = i32::try_from(seq)
        .map_err(|_| candle_core::Error::Msg(format!("RoPE seq {seq} exceeds i32")).bt())?;
    let heads_i = i32::try_from(heads)
        .map_err(|_| candle_core::Error::Msg(format!("RoPE heads {heads} exceeds i32")).bt())?;
    let batched_i = i32::from(batched_cache);
    let stream = dev.cuda_stream();
    let mut builder = stream.launch_builder(&func);
    builder.arg(&x);
    builder.arg(&cos);
    builder.arg(&sin);
    builder.arg(&y);
    builder.arg(&half_i);
    builder.arg(&seq_i);
    builder.arg(&heads_i);
    builder.arg(&batched_i);
    launch(&mut builder, cfg)?;
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
    int seq,
    int heads,
    int batched
) {
    int row = (int)blockIdx.x;
    int tid = (int)threadIdx.x;
    int s = row % seq;
    int b = 0;
    if (batched) {
        int rows_per_batch = heads * seq;
        b = row / rows_per_batch;
    }
    int cache_row = batched ? (b * seq + s) : s;
    const float* xr = x + (size_t)row * (size_t)(2 * half);
    float* yr = y + (size_t)row * (size_t)(2 * half);
    const float* cr = cos + (size_t)cache_row * (size_t)half;
    const float* sr = sin + (size_t)cache_row * (size_t)half;
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

    #[test]
    fn position_ids_offset_differs_from_sequential() {
        let d = Device::Cpu;
        let x = Tensor::randn(0.0f32, 1.0, (1, 2, 4, 8), &d).unwrap();
        let cache_len = 16;
        let half = 4;
        let cos = Tensor::randn(0.0f32, 0.3, (cache_len, half), &d).unwrap();
        let sin = Tensor::randn(0.0f32, 0.3, (cache_len, half), &d).unwrap();
        let seq_ids =
            Tensor::from_vec((0..4).map(i64::from).collect::<Vec<_>>(), (4,), &d).unwrap();
        let off_ids =
            Tensor::from_vec((3..7).map(i64::from).collect::<Vec<_>>(), (4,), &d).unwrap();
        let y0 = rope_with_position_ids(&x, &cos, &sin, &seq_ids).unwrap();
        let y1 = rope_with_position_ids(&x, &cos, &sin, &off_ids).unwrap();
        let seq_narrow = rope_custom_op(
            &x,
            &cos.narrow(0, 0, 4).unwrap(),
            &sin.narrow(0, 0, 4).unwrap(),
        )
        .unwrap();
        let mae_seq = (y0.clone() - seq_narrow)
            .unwrap()
            .abs()
            .unwrap()
            .mean_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(mae_seq < 1e-6, "seq gather mae={mae_seq}");
        let mae_off = (y0 - y1)
            .unwrap()
            .abs()
            .unwrap()
            .mean_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            mae_off > 1e-4,
            "offset ids should change RoPE, mae={mae_off}"
        );
    }

    #[test]
    fn packed_batch_rows_use_own_ids() {
        let d = Device::Cpu;
        let x = Tensor::randn(0.0f32, 1.0, (2, 1, 3, 4), &d).unwrap();
        let cos = Tensor::randn(0.0f32, 0.4, (12, 2), &d).unwrap();
        let sin = Tensor::randn(0.0f32, 0.4, (12, 2), &d).unwrap();
        // row0: 0,1,2  row1: 5,6,7 (reset / pack)
        let ids = Tensor::from_vec(vec![0i64, 1, 2, 5, 6, 7], (2, 3), &d).unwrap();
        let y = rope_with_position_ids(&x, &cos, &sin, &ids).unwrap();
        let y0 = rope_custom_op(
            &x.narrow(0, 0, 1).unwrap(),
            &cos.narrow(0, 0, 3).unwrap(),
            &sin.narrow(0, 0, 3).unwrap(),
        )
        .unwrap();
        let y1 = rope_custom_op(
            &x.narrow(0, 1, 1).unwrap(),
            &cos.narrow(0, 5, 3).unwrap(),
            &sin.narrow(0, 5, 3).unwrap(),
        )
        .unwrap();
        let got0 = y.narrow(0, 0, 1).unwrap();
        let got1 = y.narrow(0, 1, 1).unwrap();
        let m0 = (got0 - y0)
            .unwrap()
            .abs()
            .unwrap()
            .mean_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let m1 = (got1 - y1)
            .unwrap()
            .abs()
            .unwrap()
            .mean_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(m0 < 1e-6 && m1 < 1e-6, "packed mae {m0} {m1}");
    }
}
