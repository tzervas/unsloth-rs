// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! Device-resident `silu(gate) * up` via Candle [`CustomOp2`].
//!
//! This is the elementwise fuse after the two FFN GEMMs. It does **not**
//! fuse the GEMMs (cuBLAS already owns those).

use candle_core::{CpuStorage, CustomOp2, DType, Layout, Result as CandleResult, Shape, Tensor};

/// `y = silu(gate) * up`, `silu(g) = g * sigmoid(g)`.
#[derive(Clone, Debug, Default)]
pub struct SwiGluOp;

/// Apply CustomOp SwiGLU elementwise. Both tensors f32, same shape, contiguous.
///
/// # Errors
///
/// Non-f32, shape mismatch, or backend error.
pub fn swiglu_custom_op(gate: &Tensor, up: &Tensor) -> CandleResult<Tensor> {
    if gate.dtype() != DType::F32 || up.dtype() != DType::F32 {
        candle_core::bail!(
            "CustomOp SwiGLU is f32-only (got gate={:?} up={:?})",
            gate.dtype(),
            up.dtype()
        );
    }
    if gate.shape() != up.shape() {
        candle_core::bail!(
            "SwiGLU gate/up shapes must match: {:?} vs {:?}",
            gate.shape(),
            up.shape()
        );
    }
    let gate = gate.contiguous()?;
    let up = up.contiguous()?;
    gate.apply_op2(&up, SwiGluOp)
}

fn silu(g: f32) -> f32 {
    g / (1.0 + (-g).exp())
}

#[cfg(test)]
fn silu_grad(g: f32) -> f32 {
    // d/dg [g * σ(g)] = σ(g) * (1 + g * (1 - σ(g)))
    let sig = 1.0 / (1.0 + (-g).exp());
    sig * (1.0 + g * (1.0 - sig))
}

impl CustomOp2 for SwiGluOp {
    fn name(&self) -> &'static str {
        "unsloth_swiglu"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> CandleResult<(CpuStorage, Shape)> {
        let (a, b) = l1.contiguous_offsets().ok_or_else(|| {
            candle_core::Error::Msg("SwiGLU CustomOp: gate must be contiguous".into()).bt()
        })?;
        let (c, d) = l2.contiguous_offsets().ok_or_else(|| {
            candle_core::Error::Msg("SwiGLU CustomOp: up must be contiguous".into()).bt()
        })?;
        let gate = &s1.as_slice::<f32>()?[a..b];
        let up = &s2.as_slice::<f32>()?[c..d];
        if gate.len() != up.len() {
            candle_core::bail!("SwiGLU numel mismatch {} vs {}", gate.len(), up.len());
        }
        let out: Vec<f32> = gate
            .iter()
            .zip(up.iter())
            .map(|(&g, &u)| silu(g) * u)
            .collect();
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
        cuda_swiglu(s1, l1, s2, l2)
    }

    fn bwd(
        &self,
        gate: &Tensor,
        up: &Tensor,
        _y: &Tensor,
        gy: &Tensor,
    ) -> CandleResult<(Option<Tensor>, Option<Tensor>)> {
        let sig = candle_nn::ops::sigmoid(gate)?;
        let silu_g = gate.mul(&sig)?;
        let d_up = gy.mul(&silu_g)?;
        // d silu/dg = σ(g) * (1 + g * (1 - σ(g)))
        let one_minus_sig = (sig.neg()? + 1.0)?;
        let dsilu = sig.mul(&(one_minus_sig.mul(gate)? + 1.0)?)?;
        let d_gate = gy.mul(up)?.mul(&dsilu)?;
        Ok((Some(d_gate), Some(d_up)))
    }
}

#[cfg(feature = "cuda")]
fn cuda_swiglu(
    sg: &candle_core::CudaStorage,
    lg: &Layout,
    su: &candle_core::CudaStorage,
    lu: &Layout,
) -> CandleResult<(candle_core::CudaStorage, Shape)> {
    use super::nvrtc::{alloc_f32, launch, launch_config, load_func, next_pow2};
    use candle_core::cuda::cudarc::driver::PushKernelArg;
    use candle_core::cuda::CudaStorage;

    let (a, b) = lg.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg("SwiGLU CUDA: gate must be contiguous".into()).bt()
    })?;
    let (c, d) = lu
        .contiguous_offsets()
        .ok_or_else(|| candle_core::Error::Msg("SwiGLU CUDA: up must be contiguous".into()).bt())?;
    let n = b - a;
    if d - c != n {
        candle_core::bail!("SwiGLU CUDA numel mismatch");
    }
    let dev = sg.device.clone();
    let g = sg.as_cuda_slice::<f32>()?.slice(a..b);
    let u = su.as_cuda_slice::<f32>()?.slice(c..d);
    let y = alloc_f32(&dev, n)?;
    if n == 0 {
        return Ok((CudaStorage::wrap_cuda_slice(y, dev), lg.shape().clone()));
    }
    let func = load_func(&dev, "swiglu_f32", "unsloth_swiglu_f32", SWIGLU_SRC)?;
    let block = next_pow2(n.min(256)).max(32);
    let grid = n.div_ceil(block);
    let cfg = launch_config(grid, block, 0)?;
    let n_i = i32::try_from(n)
        .map_err(|_| candle_core::Error::Msg(format!("SwiGLU n {n} exceeds i32")).bt())?;
    let stream = dev.cuda_stream();
    let mut builder = stream.launch_builder(&func);
    builder.arg(&g);
    builder.arg(&u);
    builder.arg(&y);
    builder.arg(&n_i);
    launch(&mut builder, cfg)?;
    Ok((CudaStorage::wrap_cuda_slice(y, dev), lg.shape().clone()))
}

#[cfg(feature = "cuda")]
const SWIGLU_SRC: &str = r#"
extern "C" __global__ void swiglu_f32(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ y,
    int n
) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;
    float g = gate[i];
    float sig = 1.f / (1.f + expf(-g));
    y[i] = g * sig * up[i];
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn matches_silu_mul() {
        let d = Device::Cpu;
        let gate = Tensor::randn(0.0f32, 1.0, (2, 17), &d).unwrap();
        let up = Tensor::randn(0.0f32, 1.0, (2, 17), &d).unwrap();
        let y = swiglu_custom_op(&gate, &up).unwrap();
        let r = candle_nn::ops::silu(&gate).unwrap().mul(&up).unwrap();
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
    fn rejects_shape_mismatch() {
        let d = Device::Cpu;
        let gate = Tensor::zeros((2, 4), DType::F32, &d).unwrap();
        let up = Tensor::zeros((2, 3), DType::F32, &d).unwrap();
        assert!(swiglu_custom_op(&gate, &up).is_err());
    }

    #[test]
    fn backward_finite() {
        let d = Device::Cpu;
        let gate = Tensor::randn(0.0f32, 1.0, (4, 8), &d).unwrap();
        let up = Tensor::randn(0.0f32, 1.0, (4, 8), &d).unwrap();
        let y = swiglu_custom_op(&gate, &up).unwrap();
        let gy = Tensor::ones(y.shape(), DType::F32, &d).unwrap();
        let (dg, du) = SwiGluOp.bwd(&gate, &up, &y, &gy).unwrap();
        for t in [dg.unwrap(), du.unwrap()] {
            let v = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            assert!(v.iter().all(|x| x.is_finite()));
        }
    }

    #[test]
    fn silu_grad_unit() {
        let g = 0.5f32;
        let eps = 1e-3f32;
        let num = (silu(g + eps) - silu(g - eps)) / (2.0 * eps);
        assert!((num - silu_grad(g)).abs() < 1e-4);
    }
}
