// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! Device-resident attention via [`CustomOp3`].
//!
//! * CPU: per-query online softmax (no `[B,H,S,S]`).
//! * CUDA, no extra mask, head dim ≤ [`ATTN_TILE_DIM_MAX`]: SRAM-tiled
//!   Flash-style kernel (Q/K/V tiles in shared memory). Owned NVRTC, not
//!   Unsloth PTX. Wider heads fall back to the HBM-streaming online kernel.
//! * Extra masks still take Candle GEMM+softmax (scores materialize).
//!
//! CubeCL FA still `to_vec1`s and is not the default.

#[cfg(any(test, feature = "cuda"))]
use candle_core::DType;
use candle_core::{CpuStorage, CustomOp3, Layout, Result as CandleResult, Shape, Tensor};

/// Conservative Q rows per CUDA SRAM tile (always fits dim ≤ 128).
pub const ATTN_TILE_BR: usize = 16;
/// Conservative K/V rows per CUDA SRAM tile.
pub const ATTN_TILE_BC: usize = 16;
/// Occupancy tile that fits 48 KiB smem at dim 64. Measured slower on the 5080; not the default.
pub const ATTN_TILE_BR_OCC: usize = 32;
/// Occupancy K/V tile paired with [`ATTN_TILE_BR_OCC`].
pub const ATTN_TILE_BC_OCC: usize = 32;
/// Tiled path is only used at or below this head dim (48 KiB smem cap).
pub const ATTN_TILE_DIM_MAX: usize = 128;

/// Causal / non-causal scaled dot-product with online softmax.
#[derive(Clone, Debug)]
pub struct AttentionOp {
    /// Multiplier on QK^T (usually `1/sqrt(head_dim)`).
    pub scale: f32,
    /// If true, mask `key_pos > query_pos`.
    pub causal: bool,
    /// Gemma-style tanh softcap. `<= 0` disables.
    pub softcap: f32,
    /// Sliding window in tokens. `<= 0` disables (full causal / full).
    /// Causal: keys in `[q - window + 1, q]`. Non-causal: `[q - window + 1, q + window - 1]`.
    pub window: i32,
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
    attention_custom_op_cfg(q, k, v, scale, causal, 0.0, 0)
}

/// Attention with optional tanh score softcap.
///
/// # Errors
///
/// Same as [`attention_custom_op`].
pub fn attention_custom_op_softcap(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    causal: bool,
    softcap: f32,
) -> CandleResult<Tensor> {
    attention_custom_op_cfg(q, k, v, scale, causal, softcap, 0)
}

/// Attention with a sliding window. `window <= 0` is full [`attention_custom_op`].
///
/// # Errors
///
/// Same as [`attention_custom_op`].
pub fn attention_custom_op_window(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    causal: bool,
    window: i32,
) -> CandleResult<Tensor> {
    attention_custom_op_cfg(q, k, v, scale, causal, 0.0, window)
}

fn attention_custom_op_cfg(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    causal: bool,
    softcap: f32,
    window: i32,
) -> CandleResult<Tensor> {
    if q.dtype() != k.dtype() || q.dtype() != v.dtype() || !super::is_f32_or_f16(q.dtype()) {
        candle_core::bail!(
            "CustomOp attention is f32/f16 (q={:?} k={:?} v={:?})",
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
    q.apply_op3(
        &k,
        &v,
        AttentionOp {
            scale,
            causal,
            softcap,
            window,
        },
    )
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

fn apply_softcap(score: f32, softcap: f32) -> f32 {
    if softcap > 0.0 {
        (score / softcap).tanh() * softcap
    } else {
        score
    }
}

/// First allowed key index for `query_i` under an optional sliding window.
fn window_lo(query_i: usize, window: i32) -> usize {
    if window <= 0 {
        return 0;
    }
    let span = usize::try_from(window)
        .unwrap_or(usize::MAX)
        .saturating_sub(1);
    query_i.saturating_sub(span)
}

/// Exclusive last allowed key index.
fn window_hi_excl(query_i: usize, seq: usize, window: i32, causal: bool) -> usize {
    let causal_end = if causal {
        query_i.saturating_add(1).min(seq)
    } else {
        seq
    };
    if window <= 0 || causal {
        return causal_end;
    }
    let span = usize::try_from(window).unwrap_or(usize::MAX);
    causal_end.min(query_i.saturating_add(span))
}

#[allow(clippy::too_many_arguments)]
fn cpu_online_attn(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    geom: AttnGeom,
    scale: f32,
    causal: bool,
    softcap: f32,
    window: i32,
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
            let k_start = window_lo(query_i, window);
            let k_end = window_hi_excl(query_i, seq, window, causal);
            for key_j in k_start..k_end {
                let krow = &k[q_base + key_j * dim..q_base + key_j * dim + dim];
                let score = apply_softcap(super::cpu_isa::dot_f32(qrow, krow) * scale, softcap);
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
        let geom = AttnGeom {
            batch: dims[0],
            heads: dims[1],
            seq: dims[2],
            dim: dims[3],
        };
        match (s1, s2, s3) {
            (CpuStorage::F32(_), CpuStorage::F32(_), CpuStorage::F32(_)) => {
                let q = &s1.as_slice::<f32>()?[q_span.0..q_span.1];
                let k = &s2.as_slice::<f32>()?[k_span.0..k_span.1];
                let v = &s3.as_slice::<f32>()?[v_span.0..v_span.1];
                let out = cpu_online_attn(
                    q,
                    k,
                    v,
                    geom,
                    self.scale,
                    self.causal,
                    self.softcap,
                    self.window,
                );
                Ok((CpuStorage::F32(out), l1.shape().clone()))
            }
            (CpuStorage::F16(_), CpuStorage::F16(_), CpuStorage::F16(_)) => {
                let q: Vec<f32> = s1.as_slice::<half::f16>()?[q_span.0..q_span.1]
                    .iter()
                    .map(|v| v.to_f32())
                    .collect();
                let k: Vec<f32> = s2.as_slice::<half::f16>()?[k_span.0..k_span.1]
                    .iter()
                    .map(|v| v.to_f32())
                    .collect();
                let v: Vec<f32> = s3.as_slice::<half::f16>()?[v_span.0..v_span.1]
                    .iter()
                    .map(|v| v.to_f32())
                    .collect();
                let out = cpu_online_attn(
                    &q,
                    &k,
                    &v,
                    geom,
                    self.scale,
                    self.causal,
                    self.softcap,
                    self.window,
                );
                let out16: Vec<half::f16> = out.into_iter().map(half::f16::from_f32).collect();
                Ok((CpuStorage::F16(out16), l1.shape().clone()))
            }
            _ => candle_core::bail!("attn CPU dtype mismatch"),
        }
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
        use candle_core::backend::BackendStorage;
        if s1.dtype() == DType::F16 {
            return cuda_online_attn_f16(
                self.scale,
                self.causal,
                self.softcap,
                self.window,
                s1,
                l1,
                s2,
                l2,
                s3,
                l3,
            );
        }
        let dim = l1.dims().get(3).copied().unwrap_or(0);
        if tiled_smem_ok(dim) {
            cuda_tiled_attn(
                self.scale,
                self.causal,
                self.softcap,
                self.window,
                s1,
                l1,
                s2,
                l2,
                s3,
                l3,
            )
        } else {
            cuda_online_attn(
                self.scale,
                self.causal,
                self.softcap,
                self.window,
                s1,
                l1,
                s2,
                l2,
                s3,
                l3,
            )
        }
    }
}

#[cfg(any(feature = "cuda", test))]
#[must_use]
fn tiled_smem_bytes_cfg(br: usize, bc: usize, dim: usize) -> usize {
    let tiles = (2 * bc + 2 * br) * dim + br * bc + 3 * br;
    tiles * std::mem::size_of::<f32>()
}

#[cfg(any(feature = "cuda", test))]
#[must_use]
fn choose_tiles(dim: usize) -> (usize, usize) {
    // 32×32 fits 48 KiB at dim 64 but drops concurrent blocks on the 5080
    // (s512 event p50 0.68 → 0.93 ms). Keep 16×16; occupancy work is the
    // parallel row-softmax / flattened O update, not a bigger tile.
    let _ = dim;
    (ATTN_TILE_BR, ATTN_TILE_BC)
}

#[cfg(any(feature = "cuda", test))]
#[must_use]
fn tiled_smem_bytes(dim: usize) -> usize {
    let (br, bc) = choose_tiles(dim);
    tiled_smem_bytes_cfg(br, bc, dim)
}

#[cfg(any(feature = "cuda", test))]
#[must_use]
fn tiled_smem_ok(dim: usize) -> bool {
    dim > 0 && dim <= ATTN_TILE_DIM_MAX && tiled_smem_bytes(dim) <= 48 * 1024
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn cuda_online_attn(
    scale: f32,
    causal: bool,
    softcap: f32,
    window: i32,
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

    let func = load_func(
        &dev,
        "attn_online_f32",
        "unsloth_attn_online_f32_w",
        ATTN_SRC,
    )?;
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
    builder.arg(&softcap);
    builder.arg(&window);
    launch(&mut builder, cfg)?;
    Ok((CudaStorage::wrap_cuda_slice(y, dev), lq.shape().clone()))
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn cuda_online_attn_f16(
    scale: f32,
    causal: bool,
    softcap: f32,
    window: i32,
    sq: &candle_core::CudaStorage,
    lq: &Layout,
    sk: &candle_core::CudaStorage,
    lk: &Layout,
    sv: &candle_core::CudaStorage,
    lv: &Layout,
) -> CandleResult<(candle_core::CudaStorage, Shape)> {
    use super::nvrtc::{alloc_f16, launch, launch_config, load_func, next_pow2};
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
        let y = alloc_f16(&dev, 0)?;
        return Ok((CudaStorage::wrap_cuda_slice(y, dev), lq.shape().clone()));
    }

    let q = sq.as_cuda_slice::<half::f16>()?.slice(q_span.0..q_span.1);
    let k = sk.as_cuda_slice::<half::f16>()?.slice(k_span.0..k_span.1);
    let v = sv.as_cuda_slice::<half::f16>()?.slice(v_span.0..v_span.1);
    let y = alloc_f16(&dev, rows * dim)?;

    let src = format!("{}{}", super::nvrtc::F16_CONV_SRC, ATTN_F16_SRC);
    let func = load_func(
        &dev,
        "attn_online_f16",
        "unsloth_attn_online_f16_bits",
        &src,
    )?;
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
    builder.arg(&softcap);
    builder.arg(&window);
    launch(&mut builder, cfg)?;
    Ok((CudaStorage::wrap_cuda_slice(y, dev), lq.shape().clone()))
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn cuda_tiled_attn(
    scale: f32,
    causal: bool,
    softcap: f32,
    window: i32,
    sq: &candle_core::CudaStorage,
    lq: &Layout,
    sk: &candle_core::CudaStorage,
    lk: &Layout,
    sv: &candle_core::CudaStorage,
    lv: &Layout,
) -> CandleResult<(candle_core::CudaStorage, Shape)> {
    use super::nvrtc::{alloc_f32, launch, launch_config, load_func};
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
    let n_elem = batch * heads * seq * dim;
    let dev = sq.device.clone();
    if n_elem == 0 {
        let y = alloc_f32(&dev, 0)?;
        return Ok((CudaStorage::wrap_cuda_slice(y, dev), lq.shape().clone()));
    }

    let q = sq.as_cuda_slice::<f32>()?.slice(q_span.0..q_span.1);
    let k = sk.as_cuda_slice::<f32>()?.slice(k_span.0..k_span.1);
    let v = sv.as_cuda_slice::<f32>()?.slice(v_span.0..v_span.1);
    let y = alloc_f32(&dev, n_elem)?;

    let (br, bc) = choose_tiles(dim);
    let src = attn_tiled_src(br, bc);
    let module = format!("unsloth_attn_tiled_f32_br{br}_bc{bc}_w");
    let func = load_func(&dev, "attn_tiled_f32", &module, &src)?;
    let n_qtiles = seq.div_ceil(br);
    let grid = batch
        .checked_mul(heads)
        .and_then(|bh| bh.checked_mul(n_qtiles))
        .ok_or_else(|| candle_core::Error::Msg("attn CUDA tiled grid overflow".into()).bt())?;
    let cfg = launch_config(grid, 128, tiled_smem_bytes_cfg(br, bc, dim))?;
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
    builder.arg(&softcap);
    builder.arg(&window);
    launch(&mut builder, cfg)?;
    Ok((CudaStorage::wrap_cuda_slice(y, dev), lq.shape().clone()))
}

#[cfg(feature = "cuda")]
fn attn_tiled_src(br: usize, bc: usize) -> String {
    ATTN_TILED_SRC
        .replace("const int BR = 16;", &format!("const int BR = {br};"))
        .replace("const int BC = 16;", &format!("const int BC = {bc};"))
}

#[cfg(feature = "cuda")]
const ATTN_TILED_SRC: &str = r#"
extern "C" __global__ void attn_tiled_f32(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ out,
    int seq,
    int dim,
    float scale,
    int causal,
    float softcap,
    int window
) {
    const int BR = 16;
    const int BC = 16;
    const int n_qtiles = (seq + BR - 1) / BR;
    const int bh = (int)blockIdx.x / n_qtiles;
    const int qtile = (int)blockIdx.x % n_qtiles;
    const int q0 = qtile * BR;
    const int tid = (int)threadIdx.x;
    const int bdx = (int)blockDim.x;

    extern __shared__ float smem[];
    float* Ki = smem;
    float* Vi = Ki + BC * dim;
    float* Qi = Vi + BC * dim;
    float* Oi = Qi + BR * dim;
    float* S = Oi + BR * dim;
    float* mstat = S + BR * BC;
    float* lstat = mstat + BR;
    float* alph = lstat + BR;

    const size_t bh_off = (size_t)bh * (size_t)seq * (size_t)dim;
    const float* qbase = q + bh_off;
    const float* kbase = k + bh_off;
    const float* vbase = v + bh_off;
    float* obase = out + bh_off;

    for (int i = tid; i < BR * dim; i += bdx) {
        int qi = i / dim;
        int d = i - qi * dim;
        int qrow = q0 + qi;
        Qi[i] = (qrow < seq) ? qbase[(size_t)qrow * (size_t)dim + (size_t)d] : 0.f;
        Oi[i] = 0.f;
    }
    if (tid < BR) {
        mstat[tid] = -1.0e30f;
        lstat[tid] = 0.f;
    }
    __syncthreads();

    const int n_ktiles = (seq + BC - 1) / BC;
    for (int kt = 0; kt < n_ktiles; kt++) {
        int k0 = kt * BC;
        if (causal && k0 > (q0 + BR - 1)) {
            break;
        }
        if (window > 0) {
            int lo = q0 - window + 1;
            if (k0 + BC - 1 < lo) {
                continue;
            }
            if (!causal && k0 > (q0 + BR - 1 + window - 1)) {
                continue;
            }
        }
        for (int i = tid; i < BC * dim; i += bdx) {
            int kj = i / dim;
            int d = i - kj * dim;
            int krow = k0 + kj;
            float kval = 0.f;
            float vval = 0.f;
            if (krow < seq) {
                kval = kbase[(size_t)krow * (size_t)dim + (size_t)d];
                vval = vbase[(size_t)krow * (size_t)dim + (size_t)d];
            }
            Ki[i] = kval;
            Vi[i] = vval;
        }
        __syncthreads();

        for (int sidx = tid; sidx < BR * BC; sidx += bdx) {
            int qi = sidx / BC;
            int kj = sidx - qi * BC;
            int qrow = q0 + qi;
            int krow = k0 + kj;
            float score = -1.0e30f;
            int allowed = (qrow < seq && krow < seq);
            if (causal && krow > qrow) {
                allowed = 0;
            }
            if (window > 0 && allowed) {
                int lo = qrow - window + 1;
                int hi = causal ? qrow : (qrow + window - 1);
                if (krow < lo || krow > hi) {
                    allowed = 0;
                }
            }
            if (allowed) {
                const float* qr = Qi + qi * dim;
                const float* kr = Ki + kj * dim;
                float dot = 0.f;
                for (int d = 0; d < dim; d++) {
                    dot += qr[d] * kr[d];
                }
                score = dot * scale;
                if (softcap > 0.f) {
                    score = tanhf(score / softcap) * softcap;
                }
            }
            S[sidx] = score;
        }
        __syncthreads();

        // One thread per query row in the tile (BR ≤ 32, block is 128).
        if (tid < BR) {
            int qi = tid;
            float row_max = -1.0e30f;
            for (int kj = 0; kj < BC; kj++) {
                row_max = fmaxf(row_max, S[qi * BC + kj]);
            }
            float m_old = mstat[qi];
            float m_new = fmaxf(m_old, row_max);
            float alpha = (m_old > -1.0e30f) ? expf(m_old - m_new) : 0.f;
            float row_sum = 0.f;
            for (int kj = 0; kj < BC; kj++) {
                float p = expf(S[qi * BC + kj] - m_new);
                S[qi * BC + kj] = p;
                row_sum += p;
            }
            alph[qi] = alpha;
            mstat[qi] = m_new;
            lstat[qi] = lstat[qi] * alpha + row_sum;
        }
        __syncthreads();
        for (int i = tid; i < BR * dim; i += bdx) {
            int qi = i / dim;
            int d = i - qi * dim;
            float acc = 0.f;
            for (int kj = 0; kj < BC; kj++) {
                acc += S[qi * BC + kj] * Vi[kj * dim + d];
            }
            Oi[i] = Oi[i] * alph[qi] + acc;
        }
        __syncthreads();
    }

    for (int i = tid; i < BR * dim; i += bdx) {
        int qi = i / dim;
        int d = i - qi * dim;
        int qrow = q0 + qi;
        if (qrow < seq) {
            float inv = (lstat[qi] > 0.f) ? (1.f / lstat[qi]) : 0.f;
            obase[(size_t)qrow * (size_t)dim + (size_t)d] = Oi[i] * inv;
        }
    }
}
"#;

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
    int causal,
    float softcap,
    int window
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
    int k_end = causal ? (query_i + 1) : seq;
    int k_start = 0;
    if (window > 0) {
        int lo = query_i - window + 1;
        if (lo > 0) {
            k_start = lo;
        }
        if (!causal) {
            int hi = query_i + window;
            if (hi < k_end) {
                k_end = hi;
            }
        }
    }

    for (int key_j = k_start; key_j < k_end; key_j++) {
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
        if (softcap > 0.f) {
            score = tanhf(score / softcap) * softcap;
        }
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

#[cfg(feature = "cuda")]
const ATTN_F16_SRC: &str = r#"
extern "C" __global__ void attn_online_f16(
    const unsigned short* __restrict__ q,
    const unsigned short* __restrict__ k,
    const unsigned short* __restrict__ v,
    unsigned short* __restrict__ out,
    int seq,
    int dim,
    float scale,
    int causal,
    float softcap,
    int window
) {
    extern __shared__ float smem[];
    const int row = (int)blockIdx.x;
    const int tid = (int)threadIdx.x;
    const int bdx = (int)blockDim.x;
    const int query_i = row % seq;
    const int block = row / seq;
    const unsigned short* qrow = q + (size_t)row * (size_t)dim;
    const unsigned short* kbase = k + (size_t)block * (size_t)seq * (size_t)dim;
    const unsigned short* vbase = v + (size_t)block * (size_t)seq * (size_t)dim;
    unsigned short* dest = out + (size_t)row * (size_t)dim;
    float* red = smem;
    float* acc = smem + bdx;

    for (int i = tid; i < dim; i += bdx) {
        acc[i] = 0.f;
    }
    __syncthreads();

    float running_max = -1.0e30f;
    float running_sum = 0.f;
    int k_end = causal ? (query_i + 1) : seq;
    int k_start = 0;
    if (window > 0) {
        int lo = query_i - window + 1;
        if (lo > 0) {
            k_start = lo;
        }
        if (!causal) {
            int hi = query_i + window;
            if (hi < k_end) {
                k_end = hi;
            }
        }
    }

    for (int key_j = k_start; key_j < k_end; key_j++) {
        const unsigned short* krow = kbase + (size_t)key_j * (size_t)dim;
        float local = 0.f;
        for (int i = tid; i < dim; i += bdx) {
            local += u16_as_f32(qrow[i]) * u16_as_f32(krow[i]);
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
        if (softcap > 0.f) {
            score = tanhf(score / softcap) * softcap;
        }
        float new_max = fmaxf(running_max, score);
        float alpha = (running_max > -1.0e30f) ? expf(running_max - new_max) : 0.f;
        float weight = expf(score - new_max);
        running_sum = running_sum * alpha + weight;
        const unsigned short* vrow = vbase + (size_t)key_j * (size_t)dim;
        for (int i = tid; i < dim; i += bdx) {
            acc[i] = acc[i] * alpha + weight * u16_as_f32(vrow[i]);
        }
        running_max = new_max;
        __syncthreads();
    }

    float inv = (running_sum > 0.f) ? (1.f / running_sum) : 0.f;
    for (int i = tid; i < dim; i += bdx) {
        dest[i] = f32_as_u16(acc[i] * inv);
    }
}
"#;

/// Device-resident attention: tiled SRAM FA on CUDA when there is no extra mask
/// and head dim ≤ [`ATTN_TILE_DIM_MAX`]; otherwise online-softmax.
///
/// Does **not** `to_vec1`. An explicit `mask` still uses Candle GEMM+softmax.
/// This is an owned kernel, not Unsloth PTX.
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
    attention_device_softcap(q, k, v, scale, mask, causal, 0.0)
}

/// Sliding-window attention. `window == 0` is full [`attention_device`] (no extra mask).
///
/// # Errors
///
/// Shape/dtype errors from [`attention_custom_op_window`].
pub fn attention_device_window(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    causal: bool,
    window: usize,
) -> CandleResult<Tensor> {
    let window_i = i32::try_from(window).unwrap_or(i32::MAX);
    attention_custom_op_window(q, k, v, scale as f32, causal, window_i)
}

/// Like [`attention_device`] with optional tanh score softcap.
///
/// # Errors
///
/// Shape/dtype errors from [`attention_custom_op_softcap`] or Candle.
pub fn attention_device_softcap(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    mask: Option<&Tensor>,
    causal: bool,
    softcap: f32,
) -> CandleResult<Tensor> {
    let k = repeat_kv(k, q.dim(1)?)?;
    let v = repeat_kv(v, q.dim(1)?)?;
    if mask.is_none() {
        return attention_custom_op_softcap(q, &k, &v, scale as f32, causal, softcap);
    }
    let scores = q.matmul(&k.transpose(2, 3)?.contiguous()?)?;
    let mut scores = (scores * scale)?;
    if softcap > 0.0 {
        scores = ((&scores / f64::from(softcap))?.tanh()? * f64::from(softcap))?;
    }
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

/// Additive mask: `-inf` outside the sliding window (and future keys if `causal`).
#[cfg(test)]
fn window_mask_tensor(
    seq: usize,
    window: usize,
    causal: bool,
    device: &candle_core::Device,
) -> CandleResult<Tensor> {
    let mut data = vec![0.0f32; seq * seq];
    for row in 0..seq {
        let lo = window_lo(row, i32::try_from(window).unwrap_or(i32::MAX));
        let hi = window_hi_excl(row, seq, i32::try_from(window).unwrap_or(i32::MAX), causal);
        for col in 0..seq {
            if col < lo || col >= hi {
                data[row * seq + col] = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::from_vec(data, (seq, seq), device)
}

#[cfg(test)]
#[allow(clippy::many_single_char_names)]
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
    fn tiled_smem_fits_compare_heads() {
        assert!(tiled_smem_ok(64));
        assert!(tiled_smem_ok(128));
        assert!(!tiled_smem_ok(0));
        assert!(!tiled_smem_ok(256));
        assert_eq!(choose_tiles(64), (ATTN_TILE_BR, ATTN_TILE_BC));
        assert_eq!(choose_tiles(128), (ATTN_TILE_BR, ATTN_TILE_BC));
        assert!(tiled_smem_bytes_cfg(ATTN_TILE_BR_OCC, ATTN_TILE_BC_OCC, 64) <= 48 * 1024);
        assert!(tiled_smem_bytes_cfg(ATTN_TILE_BR_OCC, ATTN_TILE_BC_OCC, 128) > 48 * 1024);
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
    fn softcap_changes_output() {
        let device = Device::Cpu;
        let q = Tensor::randn(0.0f32, 2.0, (1, 1, 4, 8), &device).unwrap();
        let k = Tensor::randn(0.0f32, 2.0, (1, 1, 4, 8), &device).unwrap();
        let v = Tensor::randn(0.0f32, 2.0, (1, 1, 4, 8), &device).unwrap();
        let y0 = attention_custom_op(&q, &k, &v, 1.0, false).unwrap();
        let y1 = attention_custom_op_softcap(&q, &k, &v, 1.0, false, 1.0).unwrap();
        let mae = (y0 - y1)
            .unwrap()
            .abs()
            .unwrap()
            .mean_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(mae > 1e-6, "softcap should change attn, mae={mae}");
    }

    #[cfg(feature = "cuda")]
    fn cuda_or_fail() -> Device {
        Device::new_cuda(0).unwrap_or_else(|e| {
            eprintln!("FAIL_ENV: no CUDA device ({e})");
            std::process::exit(2);
        })
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_tiled_matches_softmax_s512() {
        let device = cuda_or_fail();
        let q = Tensor::randn(0.0f32, 1.0, (2, 8, 512, 64), &device).unwrap();
        let k = Tensor::randn(0.0f32, 1.0, (2, 8, 512, 64), &device).unwrap();
        let v = Tensor::randn(0.0f32, 1.0, (2, 8, 512, 64), &device).unwrap();
        let scale = 1.0f64 / 64.0f64.sqrt();
        let y = attention_custom_op(&q, &k, &v, scale as f32, true).unwrap();
        let scores = q
            .matmul(&k.transpose(2, 3).unwrap().contiguous().unwrap())
            .unwrap();
        let scores = (scores * scale).unwrap();
        let mask = causal_mask_tensor(512, &device).unwrap();
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
        assert!(mae < 1e-4, "tiled s512 causal mae={mae}");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_online_matches_softmax() {
        let device = cuda_or_fail();
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
        let device = cuda_or_fail();
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

    #[test]
    fn window_matches_masked_softmax_cpu() {
        let device = Device::Cpu;
        let q = Tensor::randn(0.0f32, 1.0, (1, 2, 16, 8), &device).unwrap();
        let k = Tensor::randn(0.0f32, 1.0, (1, 2, 16, 8), &device).unwrap();
        let v = Tensor::randn(0.0f32, 1.0, (1, 2, 16, 8), &device).unwrap();
        let scale = 1.0f64 / 8.0f64.sqrt();
        let window = 4usize;
        let y = attention_custom_op_window(&q, &k, &v, scale as f32, true, 4).unwrap();
        let scores = q
            .matmul(&k.transpose(2, 3).unwrap().contiguous().unwrap())
            .unwrap();
        let scores = (scores * scale).unwrap();
        let mask = window_mask_tensor(16, window, true, &device).unwrap();
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
        assert!(mae < 1e-5, "cpu window mae={mae}");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_tiled_window_matches_masked_softmax_s512() {
        let device = cuda_or_fail();
        let q = Tensor::randn(0.0f32, 1.0, (1, 4, 512, 64), &device).unwrap();
        let k = Tensor::randn(0.0f32, 1.0, (1, 4, 512, 64), &device).unwrap();
        let v = Tensor::randn(0.0f32, 1.0, (1, 4, 512, 64), &device).unwrap();
        let scale = 1.0f64 / 64.0f64.sqrt();
        let window = 128usize;
        let y = attention_custom_op_window(&q, &k, &v, scale as f32, true, 128).unwrap();
        let scores = q
            .matmul(&k.transpose(2, 3).unwrap().contiguous().unwrap())
            .unwrap();
        let scores = (scores * scale).unwrap();
        let mask = window_mask_tensor(512, window, true, &device).unwrap();
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
        assert!(mae < 1e-4, "tiled s512 window mae={mae}");
    }

    #[test]
    fn f16_cpu_mae_vs_f32_ref() {
        let device = Device::Cpu;
        let q = Tensor::randn(0.0f32, 1.0, (1, 2, 8, 16), &device).unwrap();
        let k = Tensor::randn(0.0f32, 1.0, (1, 2, 8, 16), &device).unwrap();
        let v = Tensor::randn(0.0f32, 1.0, (1, 2, 8, 16), &device).unwrap();
        let scale = 1.0f64 / 16.0f64.sqrt();
        let y = attention_custom_op(
            &q.to_dtype(DType::F16).unwrap(),
            &k.to_dtype(DType::F16).unwrap(),
            &v.to_dtype(DType::F16).unwrap(),
            scale as f32,
            false,
        )
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
        let r = reference(&q, &k, &v, scale);
        let mae = (y - r)
            .unwrap()
            .abs()
            .unwrap()
            .mean_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(mae < 2e-3, "f16 cpu attn mae={mae}");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn f16_cuda_mae_vs_f32_ref() {
        let device = Device::new_cuda(0).unwrap_or_else(|e| {
            eprintln!("FAIL_ENV: no CUDA device ({e})");
            std::process::exit(2);
        });
        let q = Tensor::randn(0.0f32, 1.0, (1, 2, 16, 32), &device).unwrap();
        let k = Tensor::randn(0.0f32, 1.0, (1, 2, 16, 32), &device).unwrap();
        let v = Tensor::randn(0.0f32, 1.0, (1, 2, 16, 32), &device).unwrap();
        let scale = 1.0f64 / 32.0f64.sqrt();
        let y = attention_custom_op(
            &q.to_dtype(DType::F16).unwrap(),
            &k.to_dtype(DType::F16).unwrap(),
            &v.to_dtype(DType::F16).unwrap(),
            scale as f32,
            false,
        )
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
        let r = reference(&q, &k, &v, scale);
        let mae = (y - r)
            .unwrap()
            .abs()
            .unwrap()
            .mean_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(mae < 2e-3, "f16 cuda attn mae={mae}");
    }
}
