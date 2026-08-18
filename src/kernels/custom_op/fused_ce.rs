// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! Fused `hidden @ weight.T` + chunked CE. Never allocates `[N, V]`.
//!
//! CPU: row-at-a-time CustomOp, scratch `O(chunk)`.
//! Other devices: Candle GEMM per vocab tile, peak extra `[N, chunk]`,
//! not `[N, V]`. Not a Triton/CubeCL job. Not a 2× claim.

use candle_core::{CpuStorage, CustomOp3, DType, Layout, Result as CandleResult, Shape, Tensor, D};

/// Fused linear + mean CE. `hidden` `[..., D]`, `weight` `[V, D]`, `targets` i64 `[...]`.
#[derive(Clone, Debug)]
pub struct FusedLinearCrossEntropyOp {
    /// HF-style ignore index (usually `-100`).
    pub ignore_index: i64,
    /// Vocab chunk. Peak extra is `O(chunk * D)`, not `O(V)`.
    pub chunk_size: usize,
}

/// Mean CE of `softmax(hidden @ W.T)` vs `targets`, without a full `[N, V]`
/// logits tensor.
///
/// # Errors
///
/// Shape/dtype mismatch, `chunk_size == 0`, or target out of range.
pub fn fused_linear_cross_entropy(
    hidden: &Tensor,
    weight: &Tensor,
    targets: &Tensor,
    ignore_index: i64,
    chunk_size: usize,
) -> CandleResult<Tensor> {
    if chunk_size == 0 {
        candle_core::bail!("fused CE chunk_size must be > 0");
    }
    if hidden.dtype() != DType::F32 || weight.dtype() != DType::F32 {
        candle_core::bail!(
            "fused CE is f32-only (hidden={:?} weight={:?})",
            hidden.dtype(),
            weight.dtype()
        );
    }
    if targets.dtype() != DType::I64 {
        candle_core::bail!("fused CE targets must be i64 (got {:?})", targets.dtype());
    }
    if weight.rank() != 2 {
        candle_core::bail!("lm_head weight must be [V, D], got {:?}", weight.shape());
    }
    let dim = weight.dim(1)?;
    if hidden.dim(D::Minus1)? != dim {
        candle_core::bail!(
            "hidden last dim {} != weight dim {dim}",
            hidden.dim(D::Minus1)?
        );
    }
    let n = hidden.elem_count() / dim;
    if targets.elem_count() != n {
        candle_core::bail!("targets numel {} != hidden rows {n}", targets.elem_count());
    }

    let hidden = hidden.contiguous()?;
    let weight = weight.contiguous()?;
    let targets = targets.contiguous()?;
    if hidden.device().is_cpu() {
        return hidden.apply_op3(
            &weight,
            &targets,
            FusedLinearCrossEntropyOp {
                ignore_index,
                chunk_size,
            },
        );
    }
    device_fused_fwd(&hidden, &weight, &targets, ignore_index, chunk_size)
}

/// `true` when fused CE avoids a full `[N, V]` logits tensor.
///
/// Both CPU CustomOp and the device GEMM-tile path do. The `device_is_cpu`
/// argument is kept so existing callers compile; it is unused.
#[must_use]
pub fn fused_linear_ce_avoids_full_logits(_device_is_cpu: bool) -> bool {
    true
}

fn as_i64(n: usize) -> CandleResult<i64> {
    i64::try_from(n)
        .map_err(|_| candle_core::Error::Msg(format!("fused CE dim {n} exceeds i64")).bt())
}

fn tensor_logsumexp_last(x: &Tensor) -> CandleResult<Tensor> {
    let max = x.max_keepdim(D::Minus1)?;
    let ex = x.broadcast_sub(&max)?.exp()?.sum_keepdim(D::Minus1)?;
    ex.log()? + max
}

fn tensor_logaddexp(a: &Tensor, b: &Tensor) -> CandleResult<Tensor> {
    let m = a.maximum(b)?;
    let ea = a.broadcast_sub(&m)?.exp()?;
    let eb = b.broadcast_sub(&m)?.exp()?;
    m.broadcast_add(&(ea + eb)?.log()?)
}

/// Vocab-tile GEMM + online LSE. Peak extra is `[N, chunk]`, not `[N, V]`.
fn device_fused_fwd(
    hidden: &Tensor,
    weight: &Tensor,
    targets: &Tensor,
    ignore: i64,
    chunk: usize,
) -> CandleResult<Tensor> {
    let dim = hidden.dim(D::Minus1)?;
    let vocab = weight.dim(0)?;
    let rows = hidden.elem_count() / dim;
    let hidden2 = hidden.reshape((rows, dim))?;
    let targets = targets.reshape((rows,))?;
    let device = hidden.device();

    let ok_ignore = targets.eq(ignore)?;
    let ok_range = targets.ge(0i64)?.mul(&targets.lt(as_i64(vocab)?)?)?;
    let ok = ok_ignore.maximum(&ok_range)?;
    if ok.min_all()?.to_dtype(DType::U8)?.to_scalar::<u8>()? == 0 {
        candle_core::bail!("fused CE target out of range for vocab {vocab}");
    }

    let valid = targets.ne(ignore)?.to_dtype(DType::F32)?;
    let mut lse: Option<Tensor> = None;
    let mut target_logit = Tensor::zeros((rows,), DType::F32, device)?;
    let mut c0 = 0usize;
    while c0 < vocab {
        let width = chunk.min(vocab - c0);
        let w_c = weight.narrow(0, c0, width)?;
        let logits_c = hidden2.matmul(&w_c.t()?)?;
        let lse_c = tensor_logsumexp_last(&logits_c)?.squeeze(D::Minus1)?;
        lse = Some(match lse {
            None => lse_c,
            Some(prev) => tensor_logaddexp(&prev, &lse_c)?,
        });
        let arange = Tensor::arange(as_i64(c0)?, as_i64(c0 + width)?, device)?;
        let hit = targets
            .unsqueeze(1)?
            .broadcast_eq(&arange)?
            .to_dtype(DType::F32)?;
        target_logit = (target_logit + (logits_c * hit)?.sum(D::Minus1)?)?;
        c0 += width;
    }
    let lse = lse.ok_or_else(|| candle_core::Error::Msg("fused CE: empty vocab".into()).bt())?;
    let per = (lse - target_logit)?;
    let n_valid = valid.sum_all()?;
    let n_valid_f = n_valid.to_dtype(DType::F32)?.to_scalar::<f32>()?;
    if n_valid_f == 0.0 {
        return Tensor::zeros((), DType::F32, device);
    }
    let weighted = (per * valid)?.sum_all()?;
    weighted.broadcast_div(&n_valid.to_dtype(DType::F32)?)
}

fn device_fused_bwd(
    hidden: &Tensor,
    weight: &Tensor,
    targets: &Tensor,
    ignore: i64,
    chunk: usize,
    scale: f32,
) -> CandleResult<(Tensor, Tensor)> {
    let dim = hidden.dim(D::Minus1)?;
    let vocab = weight.dim(0)?;
    let rows = hidden.elem_count() / dim;
    let device = hidden.device();
    let hidden2 = hidden.reshape((rows, dim))?;
    let targets = targets.reshape((rows,))?;
    if scale == 0.0 {
        return Ok((
            Tensor::zeros(hidden.shape(), DType::F32, device)?,
            Tensor::zeros(weight.shape(), DType::F32, device)?,
        ));
    }
    let scale_t = Tensor::new(scale, device)?;
    let mut dhidden = Tensor::zeros((rows, dim), DType::F32, device)?;

    let mut lse: Option<Tensor> = None;
    let mut c0 = 0usize;
    while c0 < vocab {
        let width = chunk.min(vocab - c0);
        let w_c = weight.narrow(0, c0, width)?;
        let logits_c = hidden2.matmul(&w_c.t()?)?;
        let lse_c = tensor_logsumexp_last(&logits_c)?.squeeze(D::Minus1)?;
        lse = Some(match lse {
            None => lse_c,
            Some(prev) => tensor_logaddexp(&prev, &lse_c)?,
        });
        c0 += width;
    }
    let lse =
        lse.ok_or_else(|| candle_core::Error::Msg("fused CE bwd: empty vocab".into()).bt())?;
    let lse_c1 = lse.unsqueeze(1)?;

    let mut dw_chunks: Vec<Tensor> = Vec::new();
    c0 = 0;
    while c0 < vocab {
        let width = chunk.min(vocab - c0);
        let w_c = weight.narrow(0, c0, width)?;
        let logits_c = hidden2.matmul(&w_c.t()?)?;
        let mut p = logits_c
            .broadcast_sub(&lse_c1)?
            .exp()?
            .broadcast_mul(&scale_t)?;
        let arange = Tensor::arange(as_i64(c0)?, as_i64(c0 + width)?, device)?;
        let hit = targets
            .unsqueeze(1)?
            .broadcast_eq(&arange)?
            .to_dtype(DType::F32)?;
        p = (p - hit.broadcast_mul(&scale_t)?)?;
        let ignore_m = targets.eq(ignore)?.to_dtype(DType::F32)?.unsqueeze(1)?;
        let keep = (Tensor::ones(ignore_m.shape(), DType::F32, device)? - ignore_m)?;
        p = p.broadcast_mul(&keep)?;
        dhidden = (dhidden + p.matmul(&w_c)?)?;
        dw_chunks.push(p.t()?.matmul(&hidden2)?);
        c0 += width;
    }
    let dw_refs: Vec<&Tensor> = dw_chunks.iter().collect();
    let dweight = Tensor::cat(&dw_refs, 0)?;
    Ok((
        dhidden.reshape(hidden.shape())?,
        dweight.reshape(weight.shape())?,
    ))
}

fn logaddexp(a: f32, b: f32) -> f32 {
    if a == f32::NEG_INFINITY {
        return b;
    }
    if b == f32::NEG_INFINITY {
        return a;
    }
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}

fn logsumexp(xs: &[f32]) -> f32 {
    let mut m = f32::NEG_INFINITY;
    for &v in xs {
        if v > m {
            m = v;
        }
    }
    if !m.is_finite() {
        return m;
    }
    let mut s = 0.0f32;
    for &v in xs {
        s += (v - m).exp();
    }
    m + s.ln()
}

struct FusedFwd {
    mean: f32,
    n_valid: usize,
}

#[derive(Clone, Copy)]
struct FusedGeom {
    rows: usize,
    dim: usize,
    vocab: usize,
    ignore: i64,
    chunk: usize,
}

fn cpu_fused_fwd(
    hidden: &[f32],
    weight: &[f32],
    targets: &[i64],
    geom: &FusedGeom,
) -> CandleResult<FusedFwd> {
    let rows = geom.rows;
    let workers = super::cpu_isa::cpu_worker_threads(rows);
    if workers == 1 || rows < 8 {
        return cpu_fused_fwd_range(hidden, weight, targets, geom, 0, rows);
    }
    let chunk_rows = rows.div_ceil(workers);
    std::thread::scope(|scope| {
        let mut joins = Vec::with_capacity(workers);
        let mut start = 0usize;
        while start < rows {
            let end = (start + chunk_rows).min(rows);
            let geom = *geom;
            joins
                .push(scope.spawn(move || {
                    cpu_fused_fwd_range(hidden, weight, targets, &geom, start, end)
                }));
            start = end;
        }
        let mut sum = 0.0f32;
        let mut n_valid = 0usize;
        for j in joins {
            let part = j.join().map_err(|_| {
                candle_core::Error::Msg("fused CE CPU worker panicked".into()).bt()
            })??;
            // Re-weight: each part.mean is over its own n_valid. Recover sums.
            if part.n_valid > 0 {
                sum += part.mean * part.n_valid as f32;
                n_valid += part.n_valid;
            }
        }
        let mean = if n_valid == 0 {
            0.0
        } else {
            sum / n_valid as f32
        };
        Ok(FusedFwd { mean, n_valid })
    })
}

fn cpu_fused_fwd_range(
    hidden: &[f32],
    weight: &[f32],
    targets: &[i64],
    geom: &FusedGeom,
    row0: usize,
    row1: usize,
) -> CandleResult<FusedFwd> {
    let FusedGeom {
        dim,
        vocab,
        ignore,
        chunk,
        ..
    } = *geom;
    let mut scratch = vec![0.0f32; chunk.min(vocab)];
    let mut sum = 0.0f32;
    let mut n_valid = 0usize;
    for r in row0..row1 {
        let t = targets[r];
        if t == ignore {
            continue;
        }
        if t < 0 || (t as usize) >= vocab {
            candle_core::bail!("fused CE target {t} out of range for vocab {vocab}");
        }
        let h = &hidden[r * dim..(r + 1) * dim];
        let mut lse = f32::NEG_INFINITY;
        let mut target_logit = 0.0f32;
        let mut c0 = 0usize;
        while c0 < vocab {
            let end = (c0 + chunk).min(vocab);
            let width = end - c0;
            for (k, slot) in scratch[..width].iter_mut().enumerate() {
                let wrow = &weight[(c0 + k) * dim..(c0 + k + 1) * dim];
                *slot = super::cpu_isa::dot_f32(h, wrow);
            }
            lse = logaddexp(lse, logsumexp(&scratch[..width]));
            let ti = t as usize;
            if ti >= c0 && ti < end {
                target_logit = scratch[ti - c0];
            }
            c0 = end;
        }
        sum += lse - target_logit;
        n_valid += 1;
    }
    let mean = if n_valid == 0 {
        0.0
    } else {
        sum / n_valid as f32
    };
    Ok(FusedFwd { mean, n_valid })
}

struct FusedBwd {
    dhidden: Vec<f32>,
    dweight: Vec<f32>,
}

fn cpu_fused_bwd(
    hidden: &[f32],
    weight: &[f32],
    targets: &[i64],
    geom: &FusedGeom,
    scale: f32,
) -> FusedBwd {
    let FusedGeom {
        rows,
        dim,
        vocab,
        ignore,
        chunk,
    } = *geom;
    let mut dhidden = vec![0.0f32; hidden.len()];
    let mut dweight = vec![0.0f32; weight.len()];
    if scale == 0.0 {
        return FusedBwd { dhidden, dweight };
    }
    let mut scratch = vec![0.0f32; chunk.min(vocab)];
    for r in 0..rows {
        let t = targets[r];
        if t == ignore {
            continue;
        }
        let h = &hidden[r * dim..(r + 1) * dim];
        let dh = &mut dhidden[r * dim..(r + 1) * dim];
        // lse first
        let mut lse = f32::NEG_INFINITY;
        let mut c0 = 0usize;
        while c0 < vocab {
            let end = (c0 + chunk).min(vocab);
            let width = end - c0;
            for (k, slot) in scratch[..width].iter_mut().enumerate() {
                let wrow = &weight[(c0 + k) * dim..(c0 + k + 1) * dim];
                *slot = super::cpu_isa::dot_f32(h, wrow);
            }
            lse = logaddexp(lse, logsumexp(&scratch[..width]));
            c0 = end;
        }
        c0 = 0;
        while c0 < vocab {
            let end = (c0 + chunk).min(vocab);
            let width = end - c0;
            for (k, slot) in scratch[..width].iter_mut().enumerate() {
                let wrow = &weight[(c0 + k) * dim..(c0 + k + 1) * dim];
                let dot = super::cpu_isa::dot_f32(h, wrow);
                let mut p = scale * (dot - lse).exp();
                if (t as usize) == c0 + k {
                    p -= scale;
                }
                *slot = p;
                let dw = &mut dweight[(c0 + k) * dim..(c0 + k + 1) * dim];
                for d in 0..dim {
                    dh[d] += p * wrow[d];
                    dw[d] += p * h[d];
                }
            }
            c0 = end;
        }
    }
    FusedBwd { dhidden, dweight }
}

impl CustomOp3 for FusedLinearCrossEntropyOp {
    fn name(&self) -> &'static str {
        "unsloth_fused_linear_ce"
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
        let h_span = l1.contiguous_offsets().ok_or_else(|| {
            candle_core::Error::Msg("fused CE: hidden must be contiguous".into()).bt()
        })?;
        let w_span = l2.contiguous_offsets().ok_or_else(|| {
            candle_core::Error::Msg("fused CE: weight must be contiguous".into()).bt()
        })?;
        let t_span = l3.contiguous_offsets().ok_or_else(|| {
            candle_core::Error::Msg("fused CE: targets must be contiguous".into()).bt()
        })?;
        let dim = *l1
            .dims()
            .last()
            .ok_or_else(|| candle_core::Error::Msg("fused CE: empty hidden".into()).bt())?;
        let vocab = l2.dims()[0];
        let hidden = &s1.as_slice::<f32>()?[h_span.0..h_span.1];
        let weight = &s2.as_slice::<f32>()?[w_span.0..w_span.1];
        let targets = &s3.as_slice::<i64>()?[t_span.0..t_span.1];
        let rows = hidden.len() / dim;
        let geom = FusedGeom {
            rows,
            dim,
            vocab,
            ignore: self.ignore_index,
            chunk: self.chunk_size,
        };
        let fwd = cpu_fused_fwd(hidden, weight, targets, &geom)?;
        Ok((CpuStorage::F32(vec![fwd.mean]), Shape::from(())))
    }

    fn bwd(
        &self,
        hidden: &Tensor,
        weight: &Tensor,
        targets: &Tensor,
        _loss: &Tensor,
        gy: &Tensor,
    ) -> CandleResult<(Option<Tensor>, Option<Tensor>, Option<Tensor>)> {
        let dim = hidden.dim(D::Minus1)?;
        let vocab = weight.dim(0)?;
        let rows = hidden.elem_count() / dim;
        let valid = targets
            .ne(self.ignore_index)?
            .to_dtype(DType::F32)?
            .sum_all()?
            .to_scalar::<f32>()?;
        let scale = if valid == 0.0 {
            0.0
        } else {
            gy.to_dtype(DType::F32)?.to_scalar::<f32>()? / valid
        };
        if !hidden.device().is_cpu() {
            let (dh, dw) = device_fused_bwd(
                hidden,
                weight,
                targets,
                self.ignore_index,
                self.chunk_size,
                scale,
            )?;
            return Ok((Some(dh), Some(dw), None));
        }
        let h = hidden.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
        let w = weight.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
        let t = targets.contiguous()?.flatten_all()?.to_vec1::<i64>()?;
        let geom = FusedGeom {
            rows,
            dim,
            vocab,
            ignore: self.ignore_index,
            chunk: self.chunk_size,
        };
        let grads = cpu_fused_bwd(&h, &w, &t, &geom, scale);
        let dh = Tensor::from_vec(grads.dhidden, hidden.shape(), hidden.device())?;
        let dw = Tensor::from_vec(grads.dweight, weight.shape(), weight.device())?;
        Ok((Some(dh), Some(dw), None))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    fn ref_loss(hidden: &Tensor, weight: &Tensor, targets: &Tensor, ignore: i64) -> f32 {
        let n = hidden.dim(0).unwrap();
        let d = hidden.dim(1).unwrap();
        let logits = hidden
            .reshape((n, d))
            .unwrap()
            .matmul(&weight.t().unwrap())
            .unwrap();
        super::super::chunked_cross_entropy(&logits, targets, ignore, 32)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }

    #[test]
    fn matches_unfused_ce() {
        let d = Device::Cpu;
        let hidden = Tensor::randn(0.0f32, 0.5, (6, 8), &d).unwrap();
        let weight = Tensor::randn(0.0f32, 0.5, (20, 8), &d).unwrap();
        let targets = Tensor::from_vec(vec![0i64, 3, -100, 19, 7, 2], (6,), &d).unwrap();
        let y = fused_linear_cross_entropy(&hidden, &weight, &targets, -100, 7)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let r = ref_loss(&hidden, &weight, &targets, -100);
        let err = (y - r).abs();
        assert!(err < 1e-5, "fused {y} vs chunked {r} err={err}");
        assert!(fused_linear_ce_avoids_full_logits(true));
        assert!(fused_linear_ce_avoids_full_logits(false));
    }

    #[test]
    fn device_tile_path_matches_custom_op() {
        let d = Device::Cpu;
        let hidden = Tensor::randn(0.0f32, 0.5, (6, 8), &d).unwrap();
        let weight = Tensor::randn(0.0f32, 0.5, (20, 8), &d).unwrap();
        let targets = Tensor::from_vec(vec![0i64, 3, -100, 19, 7, 2], (6,), &d).unwrap();
        let via_op = fused_linear_cross_entropy(&hidden, &weight, &targets, -100, 7)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let via_tiles = device_fused_fwd(&hidden, &weight, &targets, -100, 7)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let err = (via_op - via_tiles).abs();
        assert!(err < 1e-5, "op {via_op} vs tiles {via_tiles} err={err}");
    }

    #[test]
    fn backward_finite() {
        let d = Device::Cpu;
        let hidden = Tensor::randn(0.0f32, 0.3, (4, 6), &d).unwrap();
        let weight = Tensor::randn(0.0f32, 0.3, (10, 6), &d).unwrap();
        let targets = Tensor::from_vec(vec![1i64, 2, 0, 9], (4,), &d).unwrap();
        let loss = fused_linear_cross_entropy(&hidden, &weight, &targets, -100, 4).unwrap();
        let gy = Tensor::ones(loss.shape(), DType::F32, &d).unwrap();
        let op = FusedLinearCrossEntropyOp {
            ignore_index: -100,
            chunk_size: 4,
        };
        let (dh, dw, dt) = op.bwd(&hidden, &weight, &targets, &loss, &gy).unwrap();
        assert!(dt.is_none());
        let hs: Vec<f32> = dh.unwrap().flatten_all().unwrap().to_vec1().unwrap();
        let ws: Vec<f32> = dw.unwrap().flatten_all().unwrap().to_vec1().unwrap();
        assert!(hs.iter().all(|v| v.is_finite()));
        assert!(ws.iter().all(|v| v.is_finite()));
    }
}
