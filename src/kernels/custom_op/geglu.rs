// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! Device-resident exact GeGLU: `gelu(gate) * up` via [`CustomOp2`].
//!
//! Matches Unsloth `geglu_exact_forward_kernel` (Apache-2.0):
//! `gelu(e) = 0.5 * e * (1 + erf(e / sqrt(2)))`. Not the tanh approximation.

use candle_core::{CpuStorage, CustomOp2, DType, Layout, Result as CandleResult, Shape, Tensor};

/// `y = gelu_exact(gate) * up`.
#[derive(Clone, Debug, Default)]
pub struct GeGluOp;

/// Exact GeGLU. Both tensors f32, same shape, contiguous.
///
/// # Errors
///
/// Non-f32, shape mismatch, or backend error.
pub fn geglu_custom_op(gate: &Tensor, up: &Tensor) -> CandleResult<Tensor> {
    if gate.dtype() != DType::F32 || up.dtype() != DType::F32 {
        candle_core::bail!(
            "CustomOp GeGLU is f32-only (got gate={:?} up={:?})",
            gate.dtype(),
            up.dtype()
        );
    }
    if gate.shape() != up.shape() {
        candle_core::bail!(
            "GeGLU gate/up shapes must match: {:?} vs {:?}",
            gate.shape(),
            up.shape()
        );
    }
    let gate = gate.contiguous()?;
    let up = up.contiguous()?;
    gate.apply_op2(&up, GeGluOp)
}

/// Abramowitz–Stegun 7.1.26. Good to ~1e-7; CUDA uses `erff`.
#[allow(clippy::excessive_precision)]
fn erf_f32(x: f32) -> f32 {
    let ax = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * ax);
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    let y = 1.0 - poly * (-ax * ax).exp();
    y.copysign(x)
}

fn gelu_exact(e: f32) -> f32 {
    0.5 * e * (1.0 + erf_f32(e * std::f32::consts::FRAC_1_SQRT_2))
}

fn tensor_erf_as(x: &Tensor) -> CandleResult<Tensor> {
    let zeros = Tensor::zeros_like(x)?;
    let ax = x.abs()?;
    let t = ((&ax * 0.327_591_1)? + 1.0)?.recip()?;
    // Horner for a1 + t*(a2 + t*(a3 + t*(a4 + t*a5))), then poly = t * that.
    let p = t.affine(1.061_405_429, -1.453_152_027)?;
    let p = t.mul(&p)?.affine(1.0, 1.421_413_741)?;
    let p = t.mul(&p)?.affine(1.0, -0.284_496_736)?;
    let p = t.mul(&p)?.affine(1.0, 0.254_829_592)?;
    let poly = t.mul(&p)?;
    let mag = poly.mul(&ax.sqr()?.neg()?.exp()?)?.affine(-1.0, 1.0)?;
    x.ge(&zeros)?.where_cond(&mag, &mag.neg()?)
}

impl CustomOp2 for GeGluOp {
    fn name(&self) -> &'static str {
        "unsloth_geglu"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> CandleResult<(CpuStorage, Shape)> {
        let (a, b) = l1.contiguous_offsets().ok_or_else(|| {
            candle_core::Error::Msg("GeGLU CustomOp: gate must be contiguous".into()).bt()
        })?;
        let (c, d) = l2.contiguous_offsets().ok_or_else(|| {
            candle_core::Error::Msg("GeGLU CustomOp: up must be contiguous".into()).bt()
        })?;
        let gate = &s1.as_slice::<f32>()?[a..b];
        let up = &s2.as_slice::<f32>()?[c..d];
        if gate.len() != up.len() {
            candle_core::bail!("GeGLU numel mismatch {} vs {}", gate.len(), up.len());
        }
        let out: Vec<f32> = gate
            .iter()
            .zip(up.iter())
            .map(|(&g, &u)| gelu_exact(g) * u)
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
        cuda_geglu(s1, l1, s2, l2)
    }

    fn bwd(
        &self,
        gate: &Tensor,
        up: &Tensor,
        _y: &Tensor,
        gy: &Tensor,
    ) -> CandleResult<(Option<Tensor>, Option<Tensor>)> {
        // d gelu/de = 0.5*(1+erf(z)) + e * ϕ(e), z = e/√2, ϕ = N(0,1) pdf
        let z = (gate * std::f64::consts::FRAC_1_SQRT_2)?;
        let erf_z = tensor_erf_as(&z)?;
        let one = Tensor::ones_like(gate)?;
        let gelu = (gate * 0.5)?.mul(&erf_z.add(&one)?)?;
        let inv_sqrt_2pi = (2.0 * std::f64::consts::PI).sqrt().recip() as f32;
        let pdf = (gate.sqr()?.neg()? * 0.5)?
            .exp()?
            .affine(f64::from(inv_sqrt_2pi), 0.0)?;
        let dgelu = (erf_z.add(&one)? * 0.5)?.add(&(gate * pdf)?)?;
        let d_gate = gy.mul(up)?.mul(&dgelu)?;
        let d_up = gy.mul(&gelu)?;
        Ok((Some(d_gate), Some(d_up)))
    }
}

#[cfg(feature = "cuda")]
fn cuda_geglu(
    sg: &candle_core::CudaStorage,
    lg: &Layout,
    su: &candle_core::CudaStorage,
    lu: &Layout,
) -> CandleResult<(candle_core::CudaStorage, Shape)> {
    use super::nvrtc::{alloc_f32, launch, launch_config, load_func, next_pow2};
    use candle_core::cuda::cudarc::driver::PushKernelArg;
    use candle_core::cuda::CudaStorage;

    let (a, b) = lg.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg("GeGLU CUDA: gate must be contiguous".into()).bt()
    })?;
    let (c, d) = lu
        .contiguous_offsets()
        .ok_or_else(|| candle_core::Error::Msg("GeGLU CUDA: up must be contiguous".into()).bt())?;
    let n = b - a;
    if d - c != n {
        candle_core::bail!("GeGLU CUDA numel mismatch");
    }
    let dev = sg.device.clone();
    let g = sg.as_cuda_slice::<f32>()?.slice(a..b);
    let u = su.as_cuda_slice::<f32>()?.slice(c..d);
    let y = alloc_f32(&dev, n)?;
    if n == 0 {
        return Ok((CudaStorage::wrap_cuda_slice(y, dev), lg.shape().clone()));
    }
    let func = load_func(&dev, "geglu_f32", "unsloth_geglu_f32", GEGLU_SRC)?;
    let block = next_pow2(n.min(256)).max(32);
    let grid = n.div_ceil(block);
    let cfg = launch_config(grid, block, 0)?;
    let n_i = i32::try_from(n)
        .map_err(|_| candle_core::Error::Msg(format!("GeGLU n {n} exceeds i32")).bt())?;
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
const GEGLU_SRC: &str = r#"
extern "C" __global__ void geglu_f32(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ y,
    int n
) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;
    float e = gate[i];
    float gelu = 0.5f * e * (1.f + erff(e * 0.7071067811865476f));
    y[i] = gelu * up[i];
}
"#;

#[cfg(test)]
#[allow(clippy::similar_names, clippy::many_single_char_names)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn reference(gate: &Tensor, up: &Tensor) -> Tensor {
        let g = gate.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let u = up.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let y: Vec<f32> = g
            .iter()
            .zip(u.iter())
            .map(|(&e, &v)| gelu_exact(e) * v)
            .collect();
        Tensor::from_vec(y, gate.dims(), gate.device()).unwrap()
    }

    #[test]
    fn gelu_one_matches_known() {
        assert!((gelu_exact(1.0) - 0.841_344_746).abs() < 2e-6);
        assert!(gelu_exact(0.0).abs() < 1e-7);
    }

    #[test]
    fn matches_exact_gelu_mul() {
        let d = Device::Cpu;
        let gate = Tensor::randn(0.0f32, 1.0, (2, 17), &d).unwrap();
        let up = Tensor::randn(0.0f32, 1.0, (2, 17), &d).unwrap();
        let y = geglu_custom_op(&gate, &up).unwrap();
        let r = reference(&gate, &up);
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
        assert!(geglu_custom_op(&gate, &up).is_err());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_matches_cpu() {
        let gpu = Device::new_cuda(0).unwrap_or_else(|e| {
            eprintln!("FAIL_ENV: no CUDA device ({e})");
            std::process::exit(2);
        });
        let cpu = Device::Cpu;
        let gate = Tensor::randn(0.0f32, 1.0, (3, 19), &cpu).unwrap();
        let up = Tensor::randn(0.0f32, 1.0, (3, 19), &cpu).unwrap();
        let y_cpu = geglu_custom_op(&gate, &up).unwrap();
        let y_gpu = geglu_custom_op(&gate.to_device(&gpu).unwrap(), &up.to_device(&gpu).unwrap())
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
    fn backward_matches_central_diff() {
        let d = Device::Cpu;
        let h = 1e-3f32;
        for &e in &[0.0f32, 1.0, -1.0, 2.0, -2.0] {
            let gate = Tensor::new(&[e], &d).unwrap();
            let up = Tensor::new(&[1.25f32], &d).unwrap();
            let y = geglu_custom_op(&gate, &up).unwrap();
            let gy = Tensor::new(&[1.0f32], &d).unwrap();
            let (dg, du) = GeGluOp.bwd(&gate, &up, &y, &gy).unwrap();
            let dg = dg.unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap()[0];
            let du = du.unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap()[0];
            let y_plus = geglu_custom_op(&Tensor::new(&[e + h], &d).unwrap(), &up).unwrap();
            let y_minus = geglu_custom_op(&Tensor::new(&[e - h], &d).unwrap(), &up).unwrap();
            let y_plus = y_plus.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0];
            let y_minus = y_minus.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0];
            let want_dg = (y_plus - y_minus) / (2.0 * h); // central diff vs CustomOp fwd
            let want_du = gelu_exact(e);
            assert!((dg - want_dg).abs() < 2e-3, "e={e} dg={dg} fd={want_dg}");
            assert!((du - want_du).abs() < 2e-5, "e={e} du={du} want={want_du}");
        }
    }

    #[test]
    fn backward_finite() {
        let d = Device::Cpu;
        let gate = Tensor::randn(0.0f32, 1.0, (4, 8), &d).unwrap();
        let up = Tensor::randn(0.0f32, 1.0, (4, 8), &d).unwrap();
        let y = geglu_custom_op(&gate, &up).unwrap();
        let gy = Tensor::ones(y.shape(), DType::F32, &d).unwrap();
        let (dg, du) = GeGluOp.bwd(&gate, &up, &y, &gy).unwrap();
        for t in [dg.unwrap(), du.unwrap()] {
            let v = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            assert!(v.iter().all(|x| x.is_finite()));
        }
    }
}
