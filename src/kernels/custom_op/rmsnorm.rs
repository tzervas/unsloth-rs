// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! Device-resident RMSNorm via Candle [`CustomOp2`].
//!
//! Forward: `y = (x / sqrt(mean(x^2) + eps)) * weight` over the last dim.
//! CPU walks `CpuStorage`. CUDA launches a one-block-per-row kernel on
//! `CudaStorage` (NVRTC PTX, no CubeCL, no host copy).
//!
//! Backward uses Candle tensor ops on the same device (not fused). That is
//! still device-resident — it is not the G0 win; forward is.

#[cfg(any(test, feature = "cuda"))]
use candle_core::DType;
use candle_core::{CpuStorage, CustomOp2, Layout, Result as CandleResult, Shape, Tensor, D};

/// Binary CustomOp: `x` (any rank, last dim = `H`) × `weight` (`[H]`).
#[derive(Clone, Debug)]
pub struct RmsNormOp {
    /// Epsilon added under the square root.
    pub eps: f32,
}

impl RmsNormOp {
    /// Construct with the usual transformer `eps` (e.g. `1e-5` / `1e-6`).
    #[must_use]
    pub fn new(eps: f32) -> Self {
        Self { eps }
    }
}

/// Apply CustomOp RMSNorm. Makes `x` and `weight` contiguous first.
///
/// # Errors
///
/// Returns a candle error if dtypes are not f32/f16/bf16, ranks/shapes mismatch, or
/// the device backend rejects the op.
pub fn rmsnorm_custom_op(x: &Tensor, weight: &Tensor, eps: f32) -> CandleResult<Tensor> {
    if x.dtype() != weight.dtype() || !super::is_f32_or_f16(x.dtype()) {
        candle_core::bail!(
            "CustomOp RMSNorm is f32/f16/bf16 (got x={:?} w={:?})",
            x.dtype(),
            weight.dtype()
        );
    }
    let x = x.contiguous()?;
    let weight = weight.contiguous()?;
    x.apply_op2(&weight, RmsNormOp { eps })
}

fn last_dim(dims: &[usize]) -> CandleResult<usize> {
    dims.last()
        .copied()
        .ok_or_else(|| candle_core::Error::Msg("RMSNorm CustomOp: empty shape".into()).bt())
}

fn cpu_rmsnorm_f32(x: &[f32], w: &[f32], hidden: usize, eps: f32) -> Vec<f32> {
    let rows = x.len().checked_div(hidden).unwrap_or(0);
    let mut out = vec![0.0f32; x.len()];
    for row in 0..rows {
        let base = row * hidden;
        let mut sum_sq = 0.0f32;
        for i in 0..hidden {
            let v = x[base + i];
            sum_sq += v * v;
        }
        let inv = (sum_sq / hidden as f32 + eps).sqrt().recip();
        for i in 0..hidden {
            out[base + i] = x[base + i] * inv * w[i];
        }
    }
    out
}

impl CustomOp2 for RmsNormOp {
    fn name(&self) -> &'static str {
        "unsloth_rmsnorm"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> CandleResult<(CpuStorage, Shape)> {
        let (o1, o2) = l1.contiguous_offsets().ok_or_else(|| {
            candle_core::Error::Msg("RMSNorm CustomOp: x must be contiguous".into()).bt()
        })?;
        let (w1, w2) = l2.contiguous_offsets().ok_or_else(|| {
            candle_core::Error::Msg("RMSNorm CustomOp: weight must be contiguous".into()).bt()
        })?;
        let hidden = last_dim(l1.dims())?;
        if l2.dims() != [hidden] {
            candle_core::bail!("RMSNorm weight shape {:?} must be [{hidden}]", l2.dims());
        }
        match (s1, s2) {
            (CpuStorage::F32(_), CpuStorage::F32(_)) => {
                let x = &s1.as_slice::<f32>()?[o1..o2];
                let w = &s2.as_slice::<f32>()?[w1..w2];
                if w.len() != hidden {
                    candle_core::bail!("RMSNorm weight len {} != hidden {hidden}", w.len());
                }
                let out = cpu_rmsnorm_f32(x, w, hidden, self.eps);
                Ok((CpuStorage::F32(out), l1.shape().clone()))
            }
            (CpuStorage::F16(_), CpuStorage::F16(_)) => {
                let x = &s1.as_slice::<half::f16>()?[o1..o2];
                let w = &s2.as_slice::<half::f16>()?[w1..w2];
                if w.len() != hidden {
                    candle_core::bail!("RMSNorm weight len {} != hidden {hidden}", w.len());
                }
                let xf: Vec<f32> = x.iter().map(|v| v.to_f32()).collect();
                let wf: Vec<f32> = w.iter().map(|v| v.to_f32()).collect();
                let out = cpu_rmsnorm_f32(&xf, &wf, hidden, self.eps);
                let out16: Vec<half::f16> = out.into_iter().map(half::f16::from_f32).collect();
                Ok((CpuStorage::F16(out16), l1.shape().clone()))
            }
            (CpuStorage::BF16(_), CpuStorage::BF16(_)) => {
                let x = &s1.as_slice::<half::bf16>()?[o1..o2];
                let w = &s2.as_slice::<half::bf16>()?[w1..w2];
                if w.len() != hidden {
                    candle_core::bail!("RMSNorm weight len {} != hidden {hidden}", w.len());
                }
                let xf: Vec<f32> = x.iter().map(|v| v.to_f32()).collect();
                let wf: Vec<f32> = w.iter().map(|v| v.to_f32()).collect();
                let out = cpu_rmsnorm_f32(&xf, &wf, hidden, self.eps);
                let outb: Vec<half::bf16> = out.into_iter().map(half::bf16::from_f32).collect();
                Ok((CpuStorage::BF16(outb), l1.shape().clone()))
            }
            _ => candle_core::bail!("RMSNorm CPU dtype mismatch"),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle_core::CudaStorage,
        l1: &Layout,
        s2: &candle_core::CudaStorage,
        l2: &Layout,
    ) -> CandleResult<(candle_core::CudaStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        match s1.dtype() {
            DType::F32 => cuda_rmsnorm(self.eps, s1, l1, s2, l2),
            DType::F16 => cuda_rmsnorm_f16(self.eps, s1, l1, s2, l2),
            DType::BF16 => cuda_rmsnorm_bf16(self.eps, s1, l1, s2, l2),
            other => candle_core::bail!("RMSNorm CUDA dtype {other:?}"),
        }
    }

    fn bwd(
        &self,
        x: &Tensor,
        weight: &Tensor,
        _y: &Tensor,
        gy: &Tensor,
    ) -> CandleResult<(Option<Tensor>, Option<Tensor>)> {
        // Standard RMSNorm grad on the same device (unfused Candle ops).
        let hidden = x.dim(D::Minus1)? as f64;
        let x2 = x.sqr()?;
        let mean_sq = x2.mean_keepdim(D::Minus1)?;
        let inv = (mean_sq + f64::from(self.eps))?.sqrt()?.recip()?;
        let gy_w = gy.broadcast_mul(weight)?;
        let term1 = gy_w.broadcast_mul(&inv)?;
        let inner = gy_w.mul(x)?.sum_keepdim(D::Minus1)?;
        let inv3 = inv.sqr()?.mul(&inv)?;
        let scale = Tensor::new((hidden.recip()) as f32, x.device())?.to_dtype(x.dtype())?;
        let term2 = x
            .broadcast_mul(&inv3)?
            .broadcast_mul(&inner)?
            .broadcast_mul(&scale)?;
        let gx = (term1 - term2)?;

        // dL/dw = sum_{all but last} (gy * x * inv)
        let gx_w_src = gy.mul(x)?.broadcast_mul(&inv)?;
        let gw = if gx_w_src.rank() == 1 {
            gx_w_src
        } else {
            let mut acc = gx_w_src;
            while acc.rank() > 1 {
                acc = acc.sum(0)?;
            }
            acc
        };
        Ok((Some(gx), Some(gw)))
    }
}

#[cfg(feature = "cuda")]
fn cuda_rmsnorm(
    eps: f32,
    sx: &candle_core::CudaStorage,
    lx: &Layout,
    sw: &candle_core::CudaStorage,
    lw: &Layout,
) -> CandleResult<(candle_core::CudaStorage, Shape)> {
    use super::nvrtc::{alloc_f32, launch, launch_config, load_func, next_pow2};
    use candle_core::cuda::cudarc::driver::PushKernelArg;
    use candle_core::cuda::CudaStorage;

    let (o1, o2) = lx.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg("RMSNorm CustomOp CUDA: x must be contiguous".into()).bt()
    })?;
    let (w1, w2) = lw.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg("RMSNorm CustomOp CUDA: weight must be contiguous".into()).bt()
    })?;
    let hidden = last_dim(lx.dims())?;
    if lw.dims() != [hidden] {
        candle_core::bail!("RMSNorm weight shape {:?} must be [{hidden}]", lw.dims());
    }
    let rows = lx.shape().elem_count() / hidden;
    let dev = sx.device.clone();
    let x = sx.as_cuda_slice::<f32>()?;
    let w = sw.as_cuda_slice::<f32>()?;
    let x = x.slice(o1..o2);
    let w = w.slice(w1..w2);

    let y = alloc_f32(&dev, rows * hidden)?;
    if rows == 0 || hidden == 0 {
        return Ok((CudaStorage::wrap_cuda_slice(y, dev), lx.shape().clone()));
    }
    let func = load_func(&dev, "rmsnorm_f32", "unsloth_rmsnorm_f32", RMSNORM_SRC)?;
    let block = next_pow2(hidden.min(1024)).max(1);
    let cfg = launch_config(rows, block, block * std::mem::size_of::<f32>())?;
    let hidden_i = i32::try_from(hidden).map_err(|_| {
        candle_core::Error::Msg(format!("RMSNorm hidden {hidden} exceeds i32")).bt()
    })?;
    let stream = dev.cuda_stream();
    let mut builder = stream.launch_builder(&func);
    builder.arg(&x);
    builder.arg(&w);
    builder.arg(&y);
    builder.arg(&hidden_i);
    builder.arg(&eps);
    launch(&mut builder, cfg)?;

    let storage = CudaStorage::wrap_cuda_slice(y, dev);
    Ok((storage, lx.shape().clone()))
}

#[cfg(feature = "cuda")]
const RMSNORM_SRC: &str = r#"
extern "C" __global__ void rmsnorm_f32(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ y,
    int hidden,
    float eps
) {
    extern __shared__ float smem[];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int bdx = blockDim.x;
    const float* xr = x + (size_t)row * (size_t)hidden;
    float* yr = y + (size_t)row * (size_t)hidden;

    float local = 0.f;
    for (int i = tid; i < hidden; i += bdx) {
        float v = xr[i];
        local += v * v;
    }
    smem[tid] = local;
    __syncthreads();
    for (int stride = bdx >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }
    float inv = rsqrtf(smem[0] / (float)hidden + eps);
    for (int i = tid; i < hidden; i += bdx) {
        yr[i] = xr[i] * inv * w[i];
    }
}
"#;

#[cfg(feature = "cuda")]
fn cuda_rmsnorm_f16(
    eps: f32,
    sx: &candle_core::CudaStorage,
    lx: &Layout,
    sw: &candle_core::CudaStorage,
    lw: &Layout,
) -> CandleResult<(candle_core::CudaStorage, Shape)> {
    use super::nvrtc::{alloc_f16, launch, launch_config, load_func, next_pow2};
    use candle_core::cuda::cudarc::driver::PushKernelArg;
    use candle_core::cuda::CudaStorage;

    let (o1, o2) = lx.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg("RMSNorm CustomOp CUDA: x must be contiguous".into()).bt()
    })?;
    let (w1, w2) = lw.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg("RMSNorm CustomOp CUDA: weight must be contiguous".into()).bt()
    })?;
    let hidden = last_dim(lx.dims())?;
    if lw.dims() != [hidden] {
        candle_core::bail!("RMSNorm weight shape {:?} must be [{hidden}]", lw.dims());
    }
    let rows = lx.shape().elem_count() / hidden;
    let dev = sx.device.clone();
    let x = sx.as_cuda_slice::<half::f16>()?.slice(o1..o2);
    let w = sw.as_cuda_slice::<half::f16>()?.slice(w1..w2);
    let y = alloc_f16(&dev, rows * hidden)?;
    if rows == 0 || hidden == 0 {
        return Ok((CudaStorage::wrap_cuda_slice(y, dev), lx.shape().clone()));
    }
    let src = format!("{}{}", super::nvrtc::F16_CONV_SRC, RMSNORM_F16_SRC);
    let func = load_func(&dev, "rmsnorm_f16", "unsloth_rmsnorm_f16_bits", &src)?;
    let block = next_pow2(hidden.min(1024)).max(1);
    let cfg = launch_config(rows, block, block * std::mem::size_of::<f32>())?;
    let hidden_i = i32::try_from(hidden).map_err(|_| {
        candle_core::Error::Msg(format!("RMSNorm hidden {hidden} exceeds i32")).bt()
    })?;
    let stream = dev.cuda_stream();
    let mut builder = stream.launch_builder(&func);
    builder.arg(&x);
    builder.arg(&w);
    builder.arg(&y);
    builder.arg(&hidden_i);
    builder.arg(&eps);
    launch(&mut builder, cfg)?;
    Ok((CudaStorage::wrap_cuda_slice(y, dev), lx.shape().clone()))
}

#[cfg(feature = "cuda")]
fn cuda_rmsnorm_bf16(
    eps: f32,
    sx: &candle_core::CudaStorage,
    lx: &Layout,
    sw: &candle_core::CudaStorage,
    lw: &Layout,
) -> CandleResult<(candle_core::CudaStorage, Shape)> {
    use super::nvrtc::{alloc_bf16, launch, launch_config, load_func, next_pow2};
    use candle_core::cuda::cudarc::driver::PushKernelArg;
    use candle_core::cuda::CudaStorage;

    let (o1, o2) = lx.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg("RMSNorm CustomOp CUDA: x must be contiguous".into()).bt()
    })?;
    let (w1, w2) = lw.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg("RMSNorm CustomOp CUDA: weight must be contiguous".into()).bt()
    })?;
    let hidden = last_dim(lx.dims())?;
    if lw.dims() != [hidden] {
        candle_core::bail!("RMSNorm weight shape {:?} must be [{hidden}]", lw.dims());
    }
    let rows = lx.shape().elem_count() / hidden;
    let dev = sx.device.clone();
    let x = sx.as_cuda_slice::<half::bf16>()?.slice(o1..o2);
    let w = sw.as_cuda_slice::<half::bf16>()?.slice(w1..w2);
    let y = alloc_bf16(&dev, rows * hidden)?;
    if rows == 0 || hidden == 0 {
        return Ok((CudaStorage::wrap_cuda_slice(y, dev), lx.shape().clone()));
    }
    let src = format!("{}{}", super::nvrtc::BF16_CONV_SRC, RMSNORM_F16_SRC);
    let func = load_func(&dev, "rmsnorm_f16", "unsloth_rmsnorm_bf16_bits", &src)?;
    let block = next_pow2(hidden.min(1024)).max(1);
    let cfg = launch_config(rows, block, block * std::mem::size_of::<f32>())?;
    let hidden_i = i32::try_from(hidden).map_err(|_| {
        candle_core::Error::Msg(format!("RMSNorm hidden {hidden} exceeds i32")).bt()
    })?;
    let stream = dev.cuda_stream();
    let mut builder = stream.launch_builder(&func);
    builder.arg(&x);
    builder.arg(&w);
    builder.arg(&y);
    builder.arg(&hidden_i);
    builder.arg(&eps);
    launch(&mut builder, cfg)?;
    Ok((CudaStorage::wrap_cuda_slice(y, dev), lx.shape().clone()))
}

#[cfg(feature = "cuda")]
const RMSNORM_F16_SRC: &str = r#"
extern "C" __global__ void rmsnorm_f16(
    const unsigned short* __restrict__ x,
    const unsigned short* __restrict__ w,
    unsigned short* __restrict__ y,
    int hidden,
    float eps
) {
    extern __shared__ float smem[];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int bdx = blockDim.x;
    const unsigned short* xr = x + (size_t)row * (size_t)hidden;
    unsigned short* yr = y + (size_t)row * (size_t)hidden;

    float local = 0.f;
    for (int i = tid; i < hidden; i += bdx) {
        float v = u16_as_f32(xr[i]);
        local += v * v;
    }
    smem[tid] = local;
    __syncthreads();
    for (int stride = bdx >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }
    float inv = rsqrtf(smem[0] / (float)hidden + eps);
    for (int i = tid; i < hidden; i += bdx) {
        float v = u16_as_f32(xr[i]) * inv * u16_as_f32(w[i]);
        yr[i] = f32_as_u16(v);
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    fn reference(x: &Tensor, w: &Tensor, eps: f32) -> Tensor {
        let x_sq = x.sqr().unwrap();
        let mean_sq = x_sq.mean_keepdim(x.rank() - 1).unwrap();
        let rms = (mean_sq + f64::from(eps)).unwrap().sqrt().unwrap();
        x.broadcast_div(&rms).unwrap().broadcast_mul(w).unwrap()
    }

    #[test]
    fn custom_op_matches_reference() {
        let device = Device::Cpu;
        let x = Tensor::randn(0.0f32, 1.0, (2, 8, 32), &device).unwrap();
        let w = Tensor::ones((32,), DType::F32, &device).unwrap();
        let y = rmsnorm_custom_op(&x, &w, 1e-5).unwrap();
        let r = reference(&x, &w, 1e-5);
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
    fn f16_cpu_mae_vs_f32_ref() {
        let device = Device::Cpu;
        let x = Tensor::randn(0.0f32, 1.0, (2, 8, 32), &device).unwrap();
        let w = Tensor::ones((32,), DType::F32, &device).unwrap();
        let y = rmsnorm_custom_op(
            &x.to_dtype(DType::F16).unwrap(),
            &w.to_dtype(DType::F16).unwrap(),
            1e-5,
        )
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
        let r = reference(&x, &w, 1e-5);
        let mae = (y - r)
            .unwrap()
            .abs()
            .unwrap()
            .mean_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(mae < 2e-3, "f16 cpu mae={mae}");
    }

    #[test]
    fn bf16_cpu_mae_vs_f32_ref() {
        let device = Device::Cpu;
        let x = Tensor::randn(0.0f32, 1.0, (2, 8, 32), &device).unwrap();
        let w = Tensor::ones((32,), DType::F32, &device).unwrap();
        let y = rmsnorm_custom_op(
            &x.to_dtype(DType::BF16).unwrap(),
            &w.to_dtype(DType::BF16).unwrap(),
            1e-5,
        )
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
        let r = reference(&x, &w, 1e-5);
        let mae = (y - r)
            .unwrap()
            .abs()
            .unwrap()
            .mean_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(mae < 2e-3, "bf16 cpu mae={mae}");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn f16_cuda_mae_vs_f32_ref() {
        let device = Device::new_cuda(0).unwrap_or_else(|e| {
            eprintln!("FAIL_ENV: no CUDA device ({e})");
            std::process::exit(2);
        });
        let x = Tensor::randn(0.0f32, 1.0, (2, 8, 32), &device).unwrap();
        let w = Tensor::ones((32,), DType::F32, &device).unwrap();
        let y = rmsnorm_custom_op(
            &x.to_dtype(DType::F16).unwrap(),
            &w.to_dtype(DType::F16).unwrap(),
            1e-5,
        )
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
        let r = reference(&x, &w, 1e-5);
        let mae = (y - r)
            .unwrap()
            .abs()
            .unwrap()
            .mean_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(mae < 2e-3, "f16 cuda mae={mae}");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn bf16_cuda_mae_vs_f32_ref() {
        let device = Device::new_cuda(0).unwrap_or_else(|e| {
            eprintln!("FAIL_ENV: no CUDA device ({e})");
            std::process::exit(2);
        });
        let x = Tensor::randn(0.0f32, 1.0, (2, 8, 32), &device).unwrap();
        let w = Tensor::ones((32,), DType::F32, &device).unwrap();
        let y = rmsnorm_custom_op(
            &x.to_dtype(DType::BF16).unwrap(),
            &w.to_dtype(DType::BF16).unwrap(),
            1e-5,
        )
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
        let r = reference(&x, &w, 1e-5);
        let mae = (y - r)
            .unwrap()
            .abs()
            .unwrap()
            .mean_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(mae < 2e-3, "bf16 cuda mae={mae}");
    }

    #[test]
    fn custom_op_rejects_non_f32() {
        let device = Device::Cpu;
        let x = Tensor::zeros((2, 4), DType::F64, &device).unwrap();
        let w = Tensor::ones((4,), DType::F64, &device).unwrap();
        assert!(rmsnorm_custom_op(&x, &w, 1e-5).is_err());
    }

    #[test]
    fn custom_op_weight_must_match_hidden() {
        let device = Device::Cpu;
        let x = Tensor::randn(0.0f32, 1.0, (2, 16), &device).unwrap();
        let w = Tensor::ones((8,), DType::F32, &device).unwrap();
        assert!(rmsnorm_custom_op(&x, &w, 1e-5).is_err());
    }

    #[test]
    fn custom_op_backward_finite() {
        let device = Device::Cpu;
        let x = Tensor::randn(0.0f32, 1.0, (3, 16), &device)
            .unwrap()
            .contiguous()
            .unwrap();
        // Need a Var for autograd? apply_op2 registers bwd if inputs require grad.
        // Smoke: call bwd directly.
        let w = Tensor::ones((16,), DType::F32, &device).unwrap();
        let y = rmsnorm_custom_op(&x, &w, 1e-5).unwrap();
        let gy = Tensor::ones(y.shape(), DType::F32, &device).unwrap();
        let op = RmsNormOp::new(1e-5);
        let (gx, gw) = op.bwd(&x, &w, &y, &gy).unwrap();
        let gx = gx.unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let gw = gw.unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(gx.iter().all(|v| v.is_finite()));
        assert!(gw.iter().all(|v| v.is_finite()));
        assert_eq!(gw.len(), 16);
    }
}
