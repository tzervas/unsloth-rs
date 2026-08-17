// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! Device-resident attention (Flash-style online softmax) via [`CustomOp3`].
//!
//! Default path for `flash_attention_cubecl`: no CubeCL handle, no `to_vec1`.
//! CPU: online softmax O(S·D) extra. CUDA: Candle GEMM+softmax on `CudaStorage`
//! (still no host copy; materializes `[B,H,S,S]`). Tiled FA is
//! [triton-bridge-rs](https://github.com/tzervas/triton-bridge-rs) Phase 1.

use candle_core::{CpuStorage, CustomOp3, DType, Layout, Result as CandleResult, Shape, Tensor};

/// Causal / non-causal scaled dot-product with online softmax.
#[derive(Clone, Debug)]
pub struct AttentionOp {
    /// Multiplier on QK^T (usually `1/sqrt(head_dim)`).
    pub scale: f32,
    /// If true, mask `key_pos > query_pos`.
    pub causal: bool,
}

/// Attention on `[B, H, S, D]` Q/K/V. f32. Device-resident.
///
/// # Errors
///
/// Rank/dtype/shape mismatch.
pub fn attention_custom_op(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    causal: bool,
) -> CandleResult<Tensor> {
    if q.dtype() != DType::F32 || k.dtype() != DType::F32 || v.dtype() != DType::F32 {
        candle_core::bail!(
            "CustomOp attention is f32-only (q={:?} k={:?} v={:?})",
            q.dtype(),
            k.dtype(),
            v.dtype()
        );
    }
    if q.rank() != 4 || k.rank() != 4 || v.rank() != 4 {
        candle_core::bail!(
            "attention expects [B,H,S,D], got q={:?} k={:?} v={:?}",
            q.shape(),
            k.shape(),
            v.shape()
        );
    }
    let k = repeat_kv(k, q.dim(1)?)?;
    let v = repeat_kv(v, q.dim(1)?)?;
    if q.dims() != k.dims() || q.dims() != v.dims() {
        candle_core::bail!(
            "Q/K/V shapes must match after GQA repeat: {:?} / {:?} / {:?}",
            q.shape(),
            k.shape(),
            v.shape()
        );
    }
    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    q.apply_op3(&k, &v, AttentionOp { scale, causal })
}

fn repeat_kv(x: &Tensor, n_heads: usize) -> CandleResult<Tensor> {
    let kv_heads = x.dim(1)?;
    if kv_heads == n_heads {
        return Ok(x.clone());
    }
    if kv_heads == 0 || !n_heads.is_multiple_of(kv_heads) {
        candle_core::bail!("cannot repeat {kv_heads} KV heads to {n_heads}");
    }
    let rep = n_heads / kv_heads;
    let (batch, _, seq, dim) = x.dims4()?;
    x.unsqueeze(2)?
        .repeat((1, 1, rep, 1, 1))?
        .reshape((batch, n_heads, seq, dim))
}

struct AttnGeom {
    batch: usize,
    heads: usize,
    seq: usize,
    dim: usize,
}

fn cpu_online_attn(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    geom: AttnGeom,
    scale: f32,
    causal: bool,
) -> Vec<f32> {
    let AttnGeom {
        batch,
        heads,
        seq,
        dim,
    } = geom;
    let mut out = vec![0.0f32; q.len()];
    let blocks = batch * heads;
    for block in 0..blocks {
        let q_base = block * seq * dim;
        for query_i in 0..seq {
            let qrow = &q[q_base + query_i * dim..q_base + query_i * dim + dim];
            let mut running_max = f32::NEG_INFINITY;
            let mut running_sum = 0.0f32;
            let mut acc = vec![0.0f32; dim];
            let k_end = if causal { query_i + 1 } else { seq };
            for key_j in 0..k_end {
                let krow = &k[q_base + key_j * dim..q_base + key_j * dim + dim];
                let mut dot = 0.0f32;
                for pos in 0..dim {
                    dot += qrow[pos] * krow[pos];
                }
                let score = dot * scale;
                let new_max = running_max.max(score);
                let alpha = if running_max.is_finite() {
                    (running_max - new_max).exp()
                } else {
                    0.0
                };
                let weight = (score - new_max).exp();
                running_sum = running_sum * alpha + weight;
                let vrow = &v[q_base + key_j * dim..q_base + key_j * dim + dim];
                for pos in 0..dim {
                    acc[pos] = acc[pos] * alpha + weight * vrow[pos];
                }
                running_max = new_max;
            }
            let inv = if running_sum > 0.0 {
                running_sum.recip()
            } else {
                0.0
            };
            let dest = &mut out[q_base + query_i * dim..q_base + query_i * dim + dim];
            for pos in 0..dim {
                dest[pos] = acc[pos] * inv;
            }
        }
    }
    out
}

impl CustomOp3 for AttentionOp {
    fn name(&self) -> &'static str {
        "unsloth_attn_online"
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
        let q_span = l1
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("attn: q must be contiguous".into()).bt())?;
        let k_span = l2
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("attn: k must be contiguous".into()).bt())?;
        let v_span = l3
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("attn: v must be contiguous".into()).bt())?;
        let dims = l1.dims();
        if dims.len() != 4 {
            candle_core::bail!("attn rank {}", dims.len());
        }
        let q = &s1.as_slice::<f32>()?[q_span.0..q_span.1];
        let k = &s2.as_slice::<f32>()?[k_span.0..k_span.1];
        let v = &s3.as_slice::<f32>()?[v_span.0..v_span.1];
        let out = cpu_online_attn(
            q,
            k,
            v,
            AttnGeom {
                batch: dims[0],
                heads: dims[1],
                seq: dims[2],
                dim: dims[3],
            },
            self.scale,
            self.causal,
        );
        Ok((CpuStorage::F32(out), l1.shape().clone()))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        _s1: &candle_core::CudaStorage,
        _l1: &Layout,
        _s2: &candle_core::CudaStorage,
        _l2: &Layout,
        _s3: &candle_core::CudaStorage,
        _l3: &Layout,
    ) -> CandleResult<(candle_core::CudaStorage, Shape)> {
        candle_core::bail!(
            "CustomOp online-attn cuda_fwd not implemented; use attention_device (Candle CUDA, no D2H)"
        )
    }
}

/// Device-resident attention: CustomOp on CPU, Candle GEMM+softmax on CUDA.
///
/// CUDA path **does not** `to_vec1`. It materializes `[B,H,S,S]` scores
/// (not FA SRAM). Tiled FA is the Triton-bridge / NVRTC job.
///
/// # Errors
///
/// Shape/dtype errors from [`attention_custom_op`] or Candle.
pub fn attention_device(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    mask: Option<&Tensor>,
    causal: bool,
) -> CandleResult<Tensor> {
    let k = repeat_kv(k, q.dim(1)?)?;
    let v = repeat_kv(v, q.dim(1)?)?;
    if q.device().is_cpu() && mask.is_none() {
        return attention_custom_op(q, &k, &v, scale as f32, causal);
    }
    let scores = q.matmul(&k.transpose(2, 3)?.contiguous()?)?;
    let scores = (scores * scale)?;
    let scores = match mask {
        Some(m) => scores.broadcast_add(m)?,
        None if causal => {
            let seq = q.dim(2)?;
            let causal_mask = causal_mask_tensor(seq, q.device())?;
            scores.broadcast_add(&causal_mask)?
        }
        None => scores,
    };
    let weights = candle_nn::ops::softmax(&scores, 3)?;
    weights.matmul(&v)
}

fn causal_mask_tensor(seq: usize, device: &candle_core::Device) -> CandleResult<Tensor> {
    let mut data = vec![0.0f32; seq * seq];
    for row in 0..seq {
        for col in 0..seq {
            if col > row {
                data[row * seq + col] = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::from_vec(data, (seq, seq), device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn reference(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64) -> Tensor {
        let scores = q
            .matmul(&k.transpose(2, 3).unwrap().contiguous().unwrap())
            .unwrap();
        let scores = (scores * scale).unwrap();
        let w = candle_nn::ops::softmax(&scores, 3).unwrap();
        w.matmul(v).unwrap()
    }

    #[test]
    fn online_matches_softmax() {
        let device = Device::Cpu;
        let q = Tensor::randn(0.0f32, 1.0, (1, 2, 8, 16), &device).unwrap();
        let k = Tensor::randn(0.0f32, 1.0, (1, 2, 8, 16), &device).unwrap();
        let v = Tensor::randn(0.0f32, 1.0, (1, 2, 8, 16), &device).unwrap();
        let scale = 1.0f64 / 16.0f64.sqrt();
        let y = attention_custom_op(&q, &k, &v, scale as f32, false).unwrap();
        let r = reference(&q, &k, &v, scale);
        let mae = (y - r)
            .unwrap()
            .abs()
            .unwrap()
            .mean_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(mae < 1e-5, "mae={mae}");
    }

    #[test]
    fn causal_hides_future() {
        let device = Device::Cpu;
        let q = Tensor::ones((1, 1, 4, 2), DType::F32, &device).unwrap();
        let k = q.clone();
        let v = Tensor::from_vec(
            vec![1.0f32, 0., 2., 0., 3., 0., 4., 0.],
            (1, 1, 4, 2),
            &device,
        )
        .unwrap();
        let y = attention_custom_op(&q, &k, &v, 1.0, true)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        assert!(
            y[0][0] < y[3][0],
            "causal should accumulate more on later rows"
        );
    }
}
