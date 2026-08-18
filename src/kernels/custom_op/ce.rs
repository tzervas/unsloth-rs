// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! Chunked cross-entropy via Candle [`CustomOp2`].
//!
//! Forward never materializes a `[N, V]` softmax. Backward writes `dlogits`
//! in vocab chunks (still allocates `dlogits` so autograd can reach `lm_head`;
//! fused linear+CE is out of P1 — see `docs/P1_CUSTOMOP_PLAN.md`).

use candle_core::{CpuStorage, CustomOp2, DType, Layout, Result as CandleResult, Shape, Tensor, D};

/// Default vocab chunk. Peak extra besides `dlogits` is `O(chunk)`, not `O(V)`.
pub const DEFAULT_CE_CHUNK: usize = 4096;

/// Mean token CE. `logits` f32 `[..., V]`, `targets` i64 `[...]`.
#[derive(Clone, Debug)]
pub struct ChunkedCrossEntropyOp {
    /// Label value that is skipped (HF default `-100`).
    pub ignore_index: i64,
    /// Vocab chunk for logsumexp / softmax.
    pub chunk_size: usize,
}

impl ChunkedCrossEntropyOp {
    /// Typical SFT settings.
    #[must_use]
    pub fn new(ignore_index: i64, chunk_size: usize) -> Self {
        Self {
            ignore_index,
            chunk_size,
        }
    }
}

/// Mean CE over non-ignored tokens. f32 logits, i64 targets.
///
/// # Errors
///
/// Dtype/rank/shape mismatch, `chunk_size == 0`, or target out of range
/// (other than `ignore_index`).
pub fn chunked_cross_entropy(
    logits: &Tensor,
    targets: &Tensor,
    ignore_index: i64,
    chunk_size: usize,
) -> CandleResult<Tensor> {
    if chunk_size == 0 {
        candle_core::bail!("CE chunk_size must be > 0");
    }
    if logits.dtype() != DType::F32 {
        candle_core::bail!("CustomOp CE is f32-only (got {:?})", logits.dtype());
    }
    if targets.dtype() != DType::I64 {
        candle_core::bail!("CE targets must be i64 (got {:?})", targets.dtype());
    }
    let vocab = logits.dim(D::Minus1)?;
    let n = logits.elem_count() / vocab;
    if targets.elem_count() != n {
        candle_core::bail!(
            "CE targets numel {} != logits rows {n}",
            targets.elem_count()
        );
    }
    let logits = logits.contiguous()?;
    let targets = targets.contiguous()?;
    logits.apply_op2(
        &targets,
        ChunkedCrossEntropyOp {
            ignore_index,
            chunk_size,
        },
    )
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

fn row_lse(row: &[f32], chunk: usize) -> f32 {
    let mut lse = f32::NEG_INFINITY;
    let mut i = 0;
    while i < row.len() {
        let end = (i + chunk).min(row.len());
        lse = logaddexp(lse, logsumexp(&row[i..end]));
        i = end;
    }
    lse
}

struct CeFwd {
    mean: f32,
}

fn cpu_ce_fwd(
    logits: &[f32],
    targets: &[i64],
    vocab: usize,
    ignore: i64,
    chunk: usize,
) -> CandleResult<CeFwd> {
    let rows = logits.len() / vocab;
    let mut sum = 0.0f32;
    let mut n_valid = 0usize;
    for r in 0..rows {
        let t = targets[r];
        if t == ignore {
            continue;
        }
        if t < 0 || (t as usize) >= vocab {
            candle_core::bail!("CE target {t} out of range for vocab {vocab}");
        }
        let row = &logits[r * vocab..(r + 1) * vocab];
        let lse = row_lse(row, chunk);
        sum += lse - row[t as usize];
        n_valid += 1;
    }
    let mean = if n_valid == 0 {
        0.0
    } else {
        sum / n_valid as f32
    };
    Ok(CeFwd { mean })
}

fn cpu_ce_bwd(
    logits: &[f32],
    targets: &[i64],
    vocab: usize,
    ignore: i64,
    chunk: usize,
    scale: f32,
) -> Vec<f32> {
    let rows = logits.len() / vocab;
    let mut d = vec![0.0f32; logits.len()];
    if scale == 0.0 {
        return d;
    }
    for r in 0..rows {
        let t = targets[r];
        if t == ignore {
            continue;
        }
        let row = &logits[r * vocab..(r + 1) * vocab];
        let lse = row_lse(row, chunk);
        let base = r * vocab;
        let mut i = 0;
        while i < vocab {
            let end = (i + chunk).min(vocab);
            for v in i..end {
                d[base + v] = scale * (row[v] - lse).exp();
            }
            i = end;
        }
        if t >= 0 {
            d[base + t as usize] -= scale;
        }
    }
    d
}

impl CustomOp2 for ChunkedCrossEntropyOp {
    fn name(&self) -> &'static str {
        "unsloth_chunked_ce"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> CandleResult<(CpuStorage, Shape)> {
        let (a, b) = l1.contiguous_offsets().ok_or_else(|| {
            candle_core::Error::Msg("CE CustomOp: logits must be contiguous".into()).bt()
        })?;
        let (c, d) = l2.contiguous_offsets().ok_or_else(|| {
            candle_core::Error::Msg("CE CustomOp: targets must be contiguous".into()).bt()
        })?;
        let vocab = *l1.dims().last().ok_or_else(|| {
            candle_core::Error::Msg("CE CustomOp: empty logits shape".into()).bt()
        })?;
        let logits = &s1.as_slice::<f32>()?[a..b];
        let targets = &s2.as_slice::<i64>()?[c..d];
        let fwd = cpu_ce_fwd(logits, targets, vocab, self.ignore_index, self.chunk_size)?;
        Ok((CpuStorage::F32(vec![fwd.mean]), Shape::from(())))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle_core::CudaStorage,
        l1: &Layout,
        s2: &candle_core::CudaStorage,
        l2: &Layout,
    ) -> CandleResult<(candle_core::CudaStorage, Shape)> {
        cuda_ce_fwd(self.ignore_index, s1, l1, s2, l2)
    }

    fn bwd(
        &self,
        logits: &Tensor,
        targets: &Tensor,
        _loss: &Tensor,
        gy: &Tensor,
    ) -> CandleResult<(Option<Tensor>, Option<Tensor>)> {
        let vocab = logits.dim(D::Minus1)?;
        let n = logits.elem_count() / vocab;
        let valid = targets
            .flatten_all()?
            .to_vec1::<i64>()?
            .iter()
            .filter(|&&t| t != self.ignore_index)
            .count();
        // Note: counting valid on host is O(N) labels, not O(N*V). Acceptable.
        let scale = if valid == 0 {
            0.0
        } else {
            gy.to_dtype(DType::F32)?.to_scalar::<f32>()? / valid as f32
        };

        if logits.device().is_cpu() {
            let lg = logits.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
            let tg = targets.contiguous()?.flatten_all()?.to_vec1::<i64>()?;
            let d = cpu_ce_bwd(&lg, &tg, vocab, self.ignore_index, self.chunk_size, scale);
            let dx = Tensor::from_vec(d, logits.shape(), logits.device())?;
            return Ok((Some(dx), None));
        }

        #[cfg(feature = "cuda")]
        {
            if logits.device().is_cuda() {
                let dx = cuda_ce_bwd(logits, targets, self.ignore_index, scale)?;
                return Ok((Some(dx), None));
            }
        }

        let _ = n;
        candle_core::bail!("CE bwd: unsupported device {:?}", logits.device())
    }
}

#[cfg(feature = "cuda")]
fn cuda_ce_fwd(
    ignore: i64,
    sl: &candle_core::CudaStorage,
    ll: &Layout,
    st: &candle_core::CudaStorage,
    lt: &Layout,
) -> CandleResult<(candle_core::CudaStorage, Shape)> {
    use super::nvrtc::{alloc_f32, launch, launch_config, load_func, next_pow2};
    use candle_core::cuda::CudaStorage;
    use candle_core::cuda::cudarc::driver::PushKernelArg;

    let (a, b) = ll
        .contiguous_offsets()
        .ok_or_else(|| candle_core::Error::Msg("CE CUDA: logits must be contiguous".into()).bt())?;
    let (c, d) = lt.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg("CE CUDA: targets must be contiguous".into()).bt()
    })?;
    let vocab = *ll.dims().last().unwrap();
    let rows = (b - a) / vocab;
    let dev = sl.device.clone();
    let logits = sl.as_cuda_slice::<f32>()?.slice(a..b);
    let targets = st.as_cuda_slice::<i64>()?.slice(c..d);
    let mut sum = dev.alloc_zeros::<f32>(1)?;
    let mut cnt = dev.alloc_zeros::<i32>(1)?;
    let out = alloc_f32(&dev, 1)?;

    let vocab_i = i32::try_from(vocab)
        .map_err(|_| candle_core::Error::Msg(format!("CE vocab {vocab} exceeds i32")).bt())?;
    let stream = dev.cuda_stream();
    if rows > 0 {
        let func = load_func(&dev, "ce_fwd_f32", "unsloth_ce_fwd_f32", CE_FWD_SRC)?;
        let block = next_pow2(vocab.min(256)).max(32);
        let cfg = launch_config(rows, block, block * std::mem::size_of::<f32>() * 2)?;
        let mut builder = stream.launch_builder(&func);
        builder.arg(&logits);
        builder.arg(&targets);
        builder.arg(&sum);
        builder.arg(&cnt);
        builder.arg(&vocab_i);
        builder.arg(&ignore);
        launch(&mut builder, cfg)?;
    }
    let fin = load_func(&dev, "ce_finalize_f32", "unsloth_ce_fin_f32", CE_FIN_SRC)?;
    {
        let cfg = launch_config(1, 1, 0)?;
        let mut builder = stream.launch_builder(&fin);
        builder.arg(&sum);
        builder.arg(&cnt);
        builder.arg(&out);
        launch(&mut builder, cfg)?;
    }
    let _ = d;
    Ok((CudaStorage::wrap_cuda_slice(out, dev), Shape::from(())))
}

/// Writes `dlogits` on CUDA without pulling logits to host.
#[cfg(feature = "cuda")]
struct CeBwdOp {
    ignore: i64,
    scale: f32,
}

#[cfg(feature = "cuda")]
impl CustomOp2 for CeBwdOp {
    fn name(&self) -> &'static str {
        "unsloth_chunked_ce_bwd"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> CandleResult<(CpuStorage, Shape)> {
        let (a, b) = l1.contiguous_offsets().unwrap();
        let (c, d) = l2.contiguous_offsets().unwrap();
        let vocab = *l1.dims().last().unwrap();
        let logits = &s1.as_slice::<f32>()?[a..b];
        let targets = &s2.as_slice::<i64>()?[c..d];
        let out = cpu_ce_bwd(logits, targets, vocab, self.ignore, 4096, self.scale);
        Ok((CpuStorage::F32(out), l1.shape().clone()))
    }

    fn cuda_fwd(
        &self,
        sl: &candle_core::CudaStorage,
        ll: &Layout,
        st: &candle_core::CudaStorage,
        lt: &Layout,
    ) -> CandleResult<(candle_core::CudaStorage, Shape)> {
        use super::nvrtc::{alloc_f32, launch, launch_config, load_func, next_pow2};
        use candle_core::cuda::CudaStorage;
        use candle_core::cuda::cudarc::driver::PushKernelArg;

        let (a, b) = ll.contiguous_offsets().ok_or_else(|| {
            candle_core::Error::Msg("CE bwd CUDA: logits must be contiguous".into()).bt()
        })?;
        let (c, d) = lt.contiguous_offsets().ok_or_else(|| {
            candle_core::Error::Msg("CE bwd CUDA: targets must be contiguous".into()).bt()
        })?;
        let vocab = *ll.dims().last().unwrap();
        let rows = (b - a) / vocab;
        let dev = sl.device.clone();
        let logits = sl.as_cuda_slice::<f32>()?.slice(a..b);
        let targets = st.as_cuda_slice::<i64>()?.slice(c..d);
        let dx = alloc_f32(&dev, b - a)?;
        if rows == 0 {
            return Ok((CudaStorage::wrap_cuda_slice(dx, dev), ll.shape().clone()));
        }
        let func = load_func(&dev, "ce_bwd_f32", "unsloth_ce_bwd_f32", CE_BWD_SRC)?;
        let block = next_pow2(vocab.min(256)).max(32);
        let cfg = launch_config(rows, block, block * std::mem::size_of::<f32>() * 2)?;
        let vocab_i = i32::try_from(vocab).map_err(|_| {
            candle_core::Error::Msg(format!("CE bwd vocab {vocab} exceeds i32")).bt()
        })?;
        let stream = dev.cuda_stream();
        let mut builder = stream.launch_builder(&func);
        builder.arg(&logits);
        builder.arg(&targets);
        builder.arg(&dx);
        builder.arg(&vocab_i);
        builder.arg(&self.ignore);
        builder.arg(&self.scale);
        launch(&mut builder, cfg)?;
        let _ = (c, d);
        Ok((CudaStorage::wrap_cuda_slice(dx, dev), ll.shape().clone()))
    }
}

#[cfg(feature = "cuda")]
fn cuda_ce_bwd(logits: &Tensor, targets: &Tensor, ignore: i64, scale: f32) -> CandleResult<Tensor> {
    let logits = logits.contiguous()?;
    let targets = targets.contiguous()?;
    logits.apply_op2_no_bwd(&targets, &CeBwdOp { ignore, scale })
}

#[cfg(feature = "cuda")]
const CE_FWD_SRC: &str = r#"
extern "C" __global__ void ce_fwd_f32(
    const float* __restrict__ logits,
    const long long* __restrict__ targets,
    float* sum_loss,
    int* n_valid,
    int vocab,
    long long ignore
) {
    extern __shared__ float smem[];
    float* smax = smem;
    float* ssum = smem + blockDim.x;
    int row = (int)blockIdx.x;
    int tid = (int)threadIdx.x;
    const float* rowp = logits + (size_t)row * (size_t)vocab;

    float tmax = -1e30f;
    for (int v = tid; v < vocab; v += (int)blockDim.x) {
        tmax = fmaxf(tmax, rowp[v]);
    }
    smax[tid] = tmax;
    __syncthreads();
    for (int stride = (int)blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) smax[tid] = fmaxf(smax[tid], smax[tid + stride]);
        __syncthreads();
    }
    float mx = smax[0];
    float acc = 0.f;
    for (int v = tid; v < vocab; v += (int)blockDim.x) {
        acc += expf(rowp[v] - mx);
    }
    ssum[tid] = acc;
    __syncthreads();
    for (int stride = (int)blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) ssum[tid] += ssum[tid + stride];
        __syncthreads();
    }
    if (tid == 0) {
        long long t = targets[row];
        if (t != ignore && t >= 0 && t < (long long)vocab) {
            float lse = mx + logf(ssum[0]);
            atomicAdd(sum_loss, lse - rowp[(int)t]);
            atomicAdd(n_valid, 1);
        }
    }
}

extern "C" __global__ void ce_finalize_unused(const float* sum_loss, const int* n_valid, float* out) {
    (void)sum_loss; (void)n_valid; (void)out;
}
"#;

#[cfg(feature = "cuda")]
const CE_FIN_SRC: &str = r#"
extern "C" __global__ void ce_finalize_f32(const float* sum_loss, const int* n_valid, float* out) {
    int n = n_valid[0];
    out[0] = n > 0 ? sum_loss[0] / (float)n : 0.f;
}
"#;

#[cfg(feature = "cuda")]
const CE_BWD_SRC: &str = r#"
extern "C" __global__ void ce_bwd_f32(
    const float* __restrict__ logits,
    const long long* __restrict__ targets,
    float* __restrict__ dlogits,
    int vocab,
    long long ignore,
    float scale
) {
    extern __shared__ float smem[];
    float* smax = smem;
    float* ssum = smem + blockDim.x;
    int row = (int)blockIdx.x;
    int tid = (int)threadIdx.x;
    const float* rowp = logits + (size_t)row * (size_t)vocab;
    float* dout = dlogits + (size_t)row * (size_t)vocab;
    long long t = targets[row];
    bool valid = (t != ignore && t >= 0 && t < (long long)vocab);

    float tmax = -1e30f;
    for (int v = tid; v < vocab; v += (int)blockDim.x) {
        tmax = fmaxf(tmax, rowp[v]);
    }
    smax[tid] = tmax;
    __syncthreads();
    for (int stride = (int)blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) smax[tid] = fmaxf(smax[tid], smax[tid + stride]);
        __syncthreads();
    }
    float mx = smax[0];
    float acc = 0.f;
    for (int v = tid; v < vocab; v += (int)blockDim.x) {
        acc += expf(rowp[v] - mx);
    }
    ssum[tid] = acc;
    __syncthreads();
    for (int stride = (int)blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) ssum[tid] += ssum[tid + stride];
        __syncthreads();
    }
    float inv = valid ? (1.f / ssum[0]) : 0.f;
    for (int v = tid; v < vocab; v += (int)blockDim.x) {
        float g = valid ? scale * expf(rowp[v] - mx) * inv : 0.f;
        if (valid && v == (int)t) g -= scale;
        dout[v] = g;
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn naive_mean_ce(logits: &Tensor, targets: &[i64], ignore: i64) -> f32 {
        let vocab = logits.dim(D::Minus1).unwrap();
        let rows = logits.elem_count() / vocab;
        let flat = logits.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let mut sum = 0.0f32;
        let mut n = 0usize;
        for r in 0..rows {
            let t = targets[r];
            if t == ignore {
                continue;
            }
            let row = &flat[r * vocab..(r + 1) * vocab];
            let lse = logsumexp(row);
            sum += lse - row[t as usize];
            n += 1;
        }
        if n == 0 {
            0.0
        } else {
            sum / n as f32
        }
    }

    #[test]
    fn matches_naive_no_ignore() {
        let d = Device::Cpu;
        let logits = Tensor::randn(0.0f32, 1.0, (4, 13), &d).unwrap();
        let targets = Tensor::from_vec(vec![0i64, 3, 12, 7], (4,), &d).unwrap();
        let y = chunked_cross_entropy(&logits, &targets, -100, 5)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let r = naive_mean_ce(&logits, &[0, 3, 12, 7], -100);
        assert!((y - r).abs() < 1e-5, "y={y} r={r}");
    }

    #[test]
    fn ignore_index_drops_rows() {
        let d = Device::Cpu;
        let logits = Tensor::randn(0.0f32, 1.0, (3, 8), &d).unwrap();
        let targets = Tensor::from_vec(vec![-100i64, 2, -100], (3,), &d).unwrap();
        let y = chunked_cross_entropy(&logits, &targets, -100, 4)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let r = naive_mean_ce(&logits, &[-100, 2, -100], -100);
        assert!((y - r).abs() < 1e-5, "y={y} r={r}");
    }

    #[test]
    fn all_ignore_is_zero() {
        let d = Device::Cpu;
        let logits = Tensor::randn(0.0f32, 1.0, (2, 6), &d).unwrap();
        let targets = Tensor::from_vec(vec![-100i64, -100], (2,), &d).unwrap();
        let y = chunked_cross_entropy(&logits, &targets, -100, 6)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert_eq!(y, 0.0);
    }

    #[test]
    fn oob_target_errors() {
        let d = Device::Cpu;
        let logits = Tensor::zeros((2, 4), DType::F32, &d).unwrap();
        let targets = Tensor::from_vec(vec![0i64, 99], (2,), &d).unwrap();
        assert!(chunked_cross_entropy(&logits, &targets, -100, 4).is_err());
    }

    #[test]
    fn chunk_zero_errors() {
        let d = Device::Cpu;
        let logits = Tensor::zeros((2, 4), DType::F32, &d).unwrap();
        let targets = Tensor::from_vec(vec![0i64, 1], (2,), &d).unwrap();
        assert!(chunked_cross_entropy(&logits, &targets, -100, 0).is_err());
    }

    #[test]
    fn backward_one_hot_identity() {
        let d = Device::Cpu;
        let logits = Tensor::from_vec(vec![2.0f32, 0.0, 0.0, 0.5, 1.0, -1.0], (2, 3), &d).unwrap();
        let targets = Tensor::from_vec(vec![0i64, 1], (2,), &d).unwrap();
        let loss = chunked_cross_entropy(&logits, &targets, -100, 2).unwrap();
        let gy = Tensor::new(1.0f32, &d).unwrap();
        let op = ChunkedCrossEntropyOp::new(-100, 2);
        let (dx, dt) = op.bwd(&logits, &targets, &loss, &gy).unwrap();
        assert!(dt.is_none());
        let dx = dx.unwrap().to_vec2::<f32>().unwrap();
        // Each row of dlogits should sum to 0 (softmax - onehot).
        for row in &dx {
            let s: f32 = row.iter().sum();
            assert!(s.abs() < 1e-5, "row sum {s}");
        }
    }
}
