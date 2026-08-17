// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! NVRTC compile + `CudaDevice::get_or_load_custom_func` helper.
//!
//! Only compiled with `--features cuda`. Used by every CustomOp CUDA path so
//! we do not copy-paste `compile_ptx_with_opts` four times.

use candle_core::cuda::{cudarc, WrapErr};
use candle_core::Result;

/// Compile CUDA C to PTX (fast-math on).
pub fn compile_ptx(src: &str) -> Result<String> {
    let opts = cudarc::nvrtc::CompileOptions {
        use_fast_math: Some(true),
        ..Default::default()
    };
    let ptx = cudarc::nvrtc::safe::compile_ptx_with_opts(src, opts).w()?;
    Ok(ptx.to_src().to_string())
}

/// Compile `src` and load `fn_name` from module `module_name` (cached on device).
pub fn load_func(
    dev: &candle_core::CudaDevice,
    fn_name: &str,
    module_name: &str,
    src: &str,
) -> Result<impl core::ops::Deref<Target = cudarc::driver::CudaFunction>> {
    let ptx = compile_ptx(src)?;
    dev.get_or_load_custom_func(fn_name, module_name, &ptx)
}

/// Next power of two, minimum 1.
#[must_use]
pub fn next_pow2(n: usize) -> usize {
    if n <= 1 {
        1
    } else {
        n.next_power_of_two()
    }
}
