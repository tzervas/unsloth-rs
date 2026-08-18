// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! Device-resident LayerNorm via [`CustomOp3`]: `(x-mean)/rms * w + b`.
//!
//! Matches Unsloth `fast_layernorm` (Apache-2.0): last-dim mean/var, affine
//! weight and bias. f32. Not RMSNorm.

use candle_core::{CpuStorage, CustomOp3, DType, Layout, Result as CandleResult, Shape, Tensor, D};

/// Binary affine LayerNorm. `eps` is added under the variance sqrt.
#[derive(Clone, Debug)]
pub struct LayerNormOp {
    /// Epsilon under the variance square root.
    pub eps: f32,
}

/// `y = (x - mean) / sqrt(var + eps) * weight + bias` over the last dim.
///
/// `weight` and `bias` are `[H]`. f32.
///
/// # Errors
///
/// Dtype/shape mismatch or backend error.
pub fn layernorm_custom_op(
    x: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    eps: f32,
) -> CandleResult<Tensor> {
    if x.dtype() != DType::F32 || weight.dtype() != DType::F32 || bias.dtype() != DType::F32 {
        candle_core::bail!(
            "CustomOp LayerNorm is f32-only (got x={:?} w={:?} b={:?})",
            x.dtype(),
            weight.dtype(),
            bias.dtype()
        );
    }
    let hidden = x.dim(D::Minus1)?;
    if weight.dims() != [hidden] || bias.dims() != [hidden] {
        candle_core::bail!(
            "LayerNorm weight/bias must be [{hidden}], got {:?} / {:?}",
            weight.shape(),
            bias.shape()
        );
    }
    let x = x.contiguous()?;
    let weight = weight.contiguous()?;
    let bias = bias.contiguous()?;
    x.apply_op3(&weight, &bias, LayerNormOp { eps })
}

fn cpu_layernorm_f32(x: &[f32], w: &[f32], b: &[f32], hidden: usize, eps: f32) -> Vec<f32> {
    let rows = x.len().checked_div(hidden).unwrap_or(0);
    let mut out = vec![0.0f32; x.len()];
    let hidden_f = hidden as f32;
    for row in 0..rows {
        let base = row * hidden;
        let mut sum = 0.0f32;
        for i in 0..hidden {
            sum += x[base + i];
        }
        let mean = sum / hidden_f;
        let mut var = 0.0f32;
        for i in 0..hidden {
            let d = x[base + i] - mean;
            var += d * d;
        }
        let inv = (var / hidden_f + eps).sqrt().recip();
        for i in 0..hidden {
            out[base + i] = (x[base + i] - mean) * inv * w[i] + b[i];
        }
    }
    out
}

impl CustomOp3 for LayerNormOp {
    fn name(&self) -> &'static str {
        "unsloth_layernorm"
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
            candle_core::Error::Msg("LayerNorm: x must be contiguous".into()).bt()
        })?;
        let w_span = l2.contiguous_offsets().ok_or_else(|| {
            candle_core::Error::Msg("LayerNorm: weight must be contiguous".into()).bt()
        })?;
        let b_span = l3.contiguous_offsets().ok_or_else(|| {
            candle_core::Error::Msg("LayerNorm: bias must be contiguous".into()).bt()
        })?;
        let hidden = *l1
            .dims()
            .last()
            .ok_or_else(|| candle_core::Error::Msg("LayerNorm: empty shape".into()).bt())?;
        if l2.dims() != [hidden] || l3.dims() != [hidden] {
            candle_core::bail!(
                "LayerNorm affine {:?} / {:?} must be [{hidden}]",
                l2.dims(),
                l3.dims()
            );
        }
        let x = &s1.as_slice::<f32>()?[x_span.0..x_span.1];
        let w = &s2.as_slice::<f32>()?[w_span.0..w_span.1];
        let b = &s3.as_slice::<f32>()?[b_span.0..b_span.1];
        let out = cpu_layernorm_f32(x, w, b, hidden, self.eps);
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
        cuda_layernorm(self.eps, s1, l1, s2, l2, s3, l3)
    }
}

#[cfg(feature = "cuda")]
fn cuda_layernorm(
    eps: f32,
    sx: &candle_core::CudaStorage,
    lx: &Layout,
    sw: &candle_core::CudaStorage,
    lw: &Layout,
    sb: &candle_core::CudaStorage,
    lb: &Layout,
) -> CandleResult<(candle_core::CudaStorage, Shape)> {
    use super::nvrtc::{alloc_f32, launch, launch_config, load_func, next_pow2};
    use candle_core::cuda::cudarc::driver::PushKernelArg;
    use candle_core::cuda::CudaStorage;

    let x_span = lx.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg("LayerNorm CUDA: x must be contiguous".into()).bt()
    })?;
    let w_span = lw.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg("LayerNorm CUDA: weight must be contiguous".into()).bt()
    })?;
    let b_span = lb.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg("LayerNorm CUDA: bias must be contiguous".into()).bt()
    })?;
    let hidden = *lx
        .dims()
        .last()
        .ok_or_else(|| candle_core::Error::Msg("LayerNorm CUDA: empty shape".into()).bt())?;
    let rows = lx.shape().elem_count() / hidden;
    let dev = sx.device.clone();
    let x = sx.as_cuda_slice::<f32>()?.slice(x_span.0..x_span.1);
    let w = sw.as_cuda_slice::<f32>()?.slice(w_span.0..w_span.1);
    let b = sb.as_cuda_slice::<f32>()?.slice(b_span.0..b_span.1);
    let y = alloc_f32(&dev, rows * hidden)?;
    if rows == 0 || hidden == 0 {
        return Ok((CudaStorage::wrap_cuda_slice(y, dev), lx.shape().clone()));
    }
    let func = load_func(
        &dev,
        "layernorm_f32",
        "unsloth_layernorm_f32",
        LAYERNORM_SRC,
    )?;
    let block = next_pow2(hidden.min(1024)).max(1);
    let cfg = launch_config(rows, block, block * std::mem::size_of::<f32>())?;
    let hidden_i = i32::try_from(hidden).map_err(|_| {
        candle_core::Error::Msg(format!("LayerNorm hidden {hidden} exceeds i32")).bt()
    })?;
    let stream = dev.cuda_stream();
    let mut builder = stream.launch_builder(&func);
    builder.arg(&x);
    builder.arg(&w);
    builder.arg(&b);
    builder.arg(&y);
    builder.arg(&hidden_i);
    builder.arg(&eps);
    launch(&mut builder, cfg)?;
    Ok((CudaStorage::wrap_cuda_slice(y, dev), lx.shape().clone()))
}

#[cfg(feature = "cuda")]
const LAYERNORM_SRC: &str = r#"
extern "C" __global__ void layernorm_f32(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ y,
    int hidden,
    float eps
) {
    extern __shared__ float smem[];
    const int row = (int)blockIdx.x;
    const int tid = (int)threadIdx.x;
    const int bdx = (int)blockDim.x;
    const float* xr = x + (size_t)row * (size_t)hidden;
    float* yr = y + (size_t)row * (size_t)hidden;

    float local = 0.f;
    for (int i = tid; i < hidden; i += bdx) {
        local += xr[i];
    }
    smem[tid] = local;
    __syncthreads();
    for (int stride = bdx >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }
    float mean = smem[0] / (float)hidden;
    __syncthreads();

    local = 0.f;
    for (int i = tid; i < hidden; i += bdx) {
        float d = xr[i] - mean;
        local += d * d;
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
        yr[i] = (xr[i] - mean) * inv * w[i] + b[i];
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn reference(x: &Tensor, w: &Tensor, b: &Tensor, eps: f32) -> Tensor {
        let mean = x.mean_keepdim(x.rank() - 1).unwrap();
        let xc = x.broadcast_sub(&mean).unwrap();
        let var = xc.sqr().unwrap().mean_keepdim(x.rank() - 1).unwrap();
        let inv = (var + f64::from(eps))
            .unwrap()
            .sqrt()
            .unwrap()
            .recip()
            .unwrap();
        xc.broadcast_mul(&inv)
            .unwrap()
            .broadcast_mul(w)
            .unwrap()
            .broadcast_add(b)
            .unwrap()
    }

    #[test]
    fn matches_mean_var_affine() {
        let d = Device::Cpu;
        let x = Tensor::randn(0.0f32, 1.0, (2, 8, 32), &d).unwrap();
        let w = Tensor::randn(0.0f32, 1.0, 32, &d).unwrap();
        let b = Tensor::randn(0.0f32, 1.0, 32, &d).unwrap();
        let y = layernorm_custom_op(&x, &w, &b, 1e-5).unwrap();
        let r = reference(&x, &w, &b, 1e-5);
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

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_matches_cpu() {
        let Ok(gpu) = Device::new_cuda(0) else {
            return;
        };
        let cpu = Device::Cpu;
        let x = Tensor::randn(0.0f32, 1.0, (2, 5, 17), &cpu).unwrap();
        let w = Tensor::randn(0.0f32, 1.0, 17, &cpu).unwrap();
        let b = Tensor::randn(0.0f32, 1.0, 17, &cpu).unwrap();
        let y_cpu = layernorm_custom_op(&x, &w, &b, 1e-5).unwrap();
        let y_gpu = layernorm_custom_op(
            &x.to_device(&gpu).unwrap(),
            &w.to_device(&gpu).unwrap(),
            &b.to_device(&gpu).unwrap(),
            1e-5,
        )
        .unwrap()
        .to_device(&cpu)
        .unwrap();
        let mae = (y_cpu - y_gpu)
            .unwrap()
            .abs()
            .unwrap()
            .mean_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(mae < 1e-5, "cuda mae={mae}");
    }

    #[test]
    fn rejects_affine_mismatch() {
        let d = Device::Cpu;
        let x = Tensor::zeros((2, 8), DType::F32, &d).unwrap();
        let w = Tensor::zeros((4,), DType::F32, &d).unwrap();
        let b = Tensor::zeros((8,), DType::F32, &d).unwrap();
        assert!(layernorm_custom_op(&x, &w, &b, 1e-5).is_err());
    }
}
