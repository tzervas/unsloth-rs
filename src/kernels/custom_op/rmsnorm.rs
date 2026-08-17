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

use candle_core::{CpuStorage, CustomOp2, DType, Layout, Result as CandleResult, Shape, Tensor, D};

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
/// Returns a candle error if dtypes are not f32, ranks/shapes mismatch, or
/// the device backend rejects the op.
pub fn rmsnorm_custom_op(x: &Tensor, weight: &Tensor, eps: f32) -> CandleResult<Tensor> {
    if x.dtype() != DType::F32 || weight.dtype() != DType::F32 {
        candle_core::bail!(
            "CustomOp RMSNorm is f32-only (got x={:?} w={:?})",
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
        let x = &s1.as_slice::<f32>()?[o1..o2];
        let w = &s2.as_slice::<f32>()?[w1..w2];
        if w.len() != hidden {
            candle_core::bail!("RMSNorm weight len {} != hidden {hidden}", w.len());
        }
        let out = cpu_rmsnorm_f32(x, w, hidden, self.eps);
        Ok((CpuStorage::F32(out), l1.shape().clone()))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle_core::CudaStorage,
        l1: &Layout,
        s2: &candle_core::CudaStorage,
        l2: &Layout,
    ) -> CandleResult<(candle_core::CudaStorage, Shape)> {
        cuda_rmsnorm(self.eps, s1, l1, s2, l2)
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
    use candle_core::cuda::{cudarc, CudaStorage, WrapErr};
    use cudarc::driver::{LaunchConfig, PushKernelArg};

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

    let mut y = unsafe { dev.alloc::<f32>(rows * hidden) }?;
    let func = load_rmsnorm_func(&dev)?;
    let block = next_pow2(hidden.min(1024)).max(1);
    let cfg = LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (block as u32, 1, 1),
        shared_mem_bytes: (block * std::mem::size_of::<f32>()) as u32,
    };
    let hidden_i = hidden as i32;
    let stream = dev.cuda_stream();
    let mut builder = stream.launch_builder(&func);
    builder.arg(&x);
    builder.arg(&w);
    builder.arg(&y);
    builder.arg(&hidden_i);
    builder.arg(&eps);
    unsafe { builder.launch(cfg) }.w()?;

    let storage = CudaStorage::wrap_cuda_slice(y, dev);
    Ok((storage, lx.shape().clone()))
}

#[cfg(feature = "cuda")]
fn next_pow2(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    n.next_power_of_two()
}

#[cfg(feature = "cuda")]
fn load_rmsnorm_func(
    dev: &candle_core::CudaDevice,
) -> CandleResult<impl core::ops::Deref<Target = candle_core::cuda::cudarc::driver::CudaFunction>> {
    let ptx = rmsnorm_ptx()?;
    dev.get_or_load_custom_func("rmsnorm_f32", "unsloth_rmsnorm_f32", &ptx)
}

#[cfg(feature = "cuda")]
fn rmsnorm_ptx() -> CandleResult<String> {
    use candle_core::cuda::{cudarc, WrapErr};
    let src = r#"
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
    let opts = cudarc::nvrtc::CompileOptions {
        use_fast_math: Some(true),
        ..Default::default()
    };
    let ptx = cudarc::nvrtc::safe::compile_ptx_with_opts(src, opts).w()?;
    Ok(ptx.to_src().to_string())
}

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
