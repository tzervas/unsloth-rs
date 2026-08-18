// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! Device-resident attention (Flash-style online softmax) via [`CustomOp3`].
//!
//! Default path for `flash_attention_cubecl`: no CubeCL handle, no `to_vec1`.
//! CPU and CUDA CustomOp: online softmax, extra working set `O(S·D)` per
//! query (no `[B,H,S,S]` scores). CUDA still streams K/V from HBM — this is
//! **not** tiled SRAM Flash Attention (that is
//! [triton-bridge-rs](https://github.com/tzervas/triton-bridge-rs) Phase 1).
//! Extra masks still take Candle GEMM+softmax on `CudaStorage`.

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
        s1: &candle_core::CudaStorage,
        l1: &Layout,
        s2: &candle_core::CudaStorage,
        l2: &Layout,
        s3: &candle_core::CudaStorage,
        l3: &Layout,
    ) -> CandleResult<(candle_core::CudaStorage, Shape)> {
        cuda_online_attn(self.scale, self.causal, s1, l1, s2, l2, s3, l3)
    }
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn cuda_online_attn(
    scale: f32,
    causal: bool,
    sq: &candle_core::CudaStorage,
    lq: &Layout,
    sk: &candle_core::CudaStorage,
    lk: &Layout,
    sv: &candle_core::CudaStorage,
    lv: &Layout,
) -> CandleResult<(candle_core::CudaStorage, Shape)> {
    use super::nvrtc::{alloc_f32, launch, launch_config, load_func, next_pow2};
    use candle_core::cuda::cudarc::driver::PushKernelArg;
    use candle_core::cuda::CudaStorage;

    let q_span = lq
        .contiguous_offsets()
        .ok_or_else(|| candle_core::Error::Msg("attn CUDA: q must be contiguous".into()).bt())?;
    let k_span = lk
        .contiguous_offsets()
        .ok_or_else(|| candle_core::Error::Msg("attn CUDA: k must be contiguous".into()).bt())?;
    let v_span = lv
        .contiguous_offsets()
        .ok_or_else(|| candle_core::Error::Msg("attn CUDA: v must be contiguous".into()).bt())?;
    let dims = lq.dims();
    if dims.len() != 4 {
        candle_core::bail!("attn CUDA rank {}", dims.len());
    }
    if lk.dims() != dims || lv.dims() != dims {
        candle_core::bail!(
            "attn CUDA Q/K/V shape mismatch {:?} / {:?} / {:?}",
            dims,
            lk.dims(),
            lv.dims()
        );
    }
    let (batch, heads, seq, dim) = (dims[0], dims[1], dims[2], dims[3]);
    let rows = batch * heads * seq;
    let dev = sq.device.clone();
    if rows == 0 || dim == 0 {
        let y = alloc_f32(&dev, 0)?;
        return Ok((CudaStorage::wrap_cuda_slice(y, dev), lq.shape().clone()));
    }

    let q = sq.as_cuda_slice::<f32>()?.slice(q_span.0..q_span.1);
    let k = sk.as_cuda_slice::<f32>()?.slice(k_span.0..k_span.1);
    let v = sv.as_cuda_slice::<f32>()?.slice(v_span.0..v_span.1);
    let y = alloc_f32(&dev, rows * dim)?;

    let func = load_func(&dev, "attn_online_f32", "unsloth_attn_online_f32", ATTN_SRC)?;
    let block = next_pow2(dim.min(1024)).max(1);
    let smem = (block + dim) * std::mem::size_of::<f32>();
    let cfg = launch_config(rows, block, smem)?;
    let seq_i = i32::try_from(seq)
        .map_err(|_| candle_core::Error::Msg(format!("attn CUDA seq {seq} exceeds i32")).bt())?;
    let dim_i = i32::try_from(dim)
        .map_err(|_| candle_core::Error::Msg(format!("attn CUDA dim {dim} exceeds i32")).bt())?;
    let causal_i = i32::from(causal);
    let stream = dev.cuda_stream();
    let mut builder = stream.launch_builder(&func);
    builder.arg(&q);
    builder.arg(&k);
    builder.arg(&v);
    builder.arg(&y);
    builder.arg(&seq_i);
    builder.arg(&dim_i);
    builder.arg(&scale);
    builder.arg(&causal_i);
    launch(&mut builder, cfg)?;
    Ok((CudaStorage::wrap_cuda_slice(y, dev), lq.shape().clone()))
}

#[cfg(feature = "cuda")]
const ATTN_SRC: &str = r#"
extern "C" __global__ void attn_online_f32(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ out,
    int seq,
    int dim,
    float scale,
    int causal
) {
    extern __shared__ float smem[];
    const int row = (int)blockIdx.x;
    const int tid = (int)threadIdx.x;
    const int bdx = (int)blockDim.x;
    const int query_i = row % seq;
    const int block = row / seq;
    const float* qrow = q + (size_t)row * (size_t)dim;
    const float* kbase = k + (size_t)block * (size_t)seq * (size_t)dim;
    const float* vbase = v + (size_t)block * (size_t)seq * (size_t)dim;
    float* dest = out + (size_t)row * (size_t)dim;
    float* red = smem;
    float* acc = smem + bdx;

    for (int i = tid; i < dim; i += bdx) {
        acc[i] = 0.f;
    }
    __syncthreads();

    float running_max = -1.0e30f;
    float running_sum = 0.f;
    const int k_end = causal ? (query_i + 1) : seq;

    for (int key_j = 0; key_j < k_end; key_j++) {
        const float* krow = kbase + (size_t)key_j * (size_t)dim;
        float local = 0.f;
        for (int i = tid; i < dim; i += bdx) {
            local += qrow[i] * krow[i];
        }
        red[tid] = local;
        __syncthreads();
        for (int stride = bdx >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                red[tid] += red[tid + stride];
            }
            __syncthreads();
        }
        float score = red[0] * scale;
        float new_max = fmaxf(running_max, score);
        float alpha = (running_max > -1.0e30f) ? expf(running_max - new_max) : 0.f;
        float weight = expf(score - new_max);
        running_sum = running_sum * alpha + weight;
        const float* vrow = vbase + (size_t)key_j * (size_t)dim;
        for (int i = tid; i < dim; i += bdx) {
            acc[i] = acc[i] * alpha + weight * vrow[i];
        }
        running_max = new_max;
        __syncthreads();
    }

    float inv = (running_sum > 0.f) ? (1.f / running_sum) : 0.f;
    for (int i = tid; i < dim; i += bdx) {
        dest[i] = acc[i] * inv;
    }
}
"#;

/// Device-resident attention: CustomOp online-softmax when there is no extra mask.
///
/// CUDA CustomOp does **not** `to_vec1` and does **not** allocate `[B,H,S,S]`.
/// An explicit `mask` still uses Candle GEMM+softmax (scores materialize).
/// Tiled SRAM FA is the Triton-bridge job.
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
    if mask.is_none() {
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

    #[cfg(feature = "cuda")]
    fn cuda_or_skip() -> Option<Device> {
        Device::new_cuda(0).ok()
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_online_matches_softmax() {
        let Some(device) = cuda_or_skip() else {
            return;
        };
        let q = Tensor::randn(0.0f32, 1.0, (2, 4, 16, 32), &device).unwrap();
        let k = Tensor::randn(0.0f32, 1.0, (2, 4, 16, 32), &device).unwrap();
        let v = Tensor::randn(0.0f32, 1.0, (2, 4, 16, 32), &device).unwrap();
        let scale = 1.0f64 / 32.0f64.sqrt();
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
        assert!(mae < 1e-5, "cuda mae={mae}");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_causal_matches_masked_softmax() {
        let Some(device) = cuda_or_skip() else {
            return;
        };
        let q = Tensor::randn(0.0f32, 1.0, (1, 2, 8, 16), &device).unwrap();
        let k = Tensor::randn(0.0f32, 1.0, (1, 2, 8, 16), &device).unwrap();
        let v = Tensor::randn(0.0f32, 1.0, (1, 2, 8, 16), &device).unwrap();
        let scale = 1.0f64 / 16.0f64.sqrt();
        let y = attention_custom_op(&q, &k, &v, scale as f32, true).unwrap();
        let scores = q
            .matmul(&k.transpose(2, 3).unwrap().contiguous().unwrap())
            .unwrap();
        let scores = (scores * scale).unwrap();
        let mask = causal_mask_tensor(8, &device).unwrap();
        let scores = scores.broadcast_add(&mask).unwrap();
        let r = candle_nn::ops::softmax(&scores, 3)
            .unwrap()
            .matmul(&v)
            .unwrap();
        let mae = (y - r)
            .unwrap()
            .abs()
            .unwrap()
            .mean_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(mae < 1e-5, "cuda causal mae={mae}");
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
