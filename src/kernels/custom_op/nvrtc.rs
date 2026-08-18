// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! NVRTC compile + `CudaDevice::get_or_load_custom_func` helper.
//!
//! Only compiled with `--features cuda`. Used by every CustomOp CUDA path so
//! we do not copy-paste `compile_ptx_with_opts` four times.
//!
//! `load_func` caches C→PTX by `module_name`. `get_or_load_custom_func` already
//! caches the device function; compiling on every call was the launch-tax leak.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use candle_core::cuda::{cudarc, WrapErr};
use candle_core::Result;

static PTX_CACHE: OnceLock<Mutex<HashMap<String, String>>> = OnceLock::new();

/// Compile CUDA C to PTX (fast-math on).
pub fn compile_ptx(src: &str) -> Result<String> {
    let opts = cudarc::nvrtc::CompileOptions {
        use_fast_math: Some(true),
        ..Default::default()
    };
    let ptx = cudarc::nvrtc::safe::compile_ptx_with_opts(src, opts).w()?;
    Ok(ptx.to_src().to_string())
}

fn cached_ptx(module_name: &str, src: &str) -> Result<String> {
    let cache = PTX_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    {
        let guard = match cache.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        if let Some(ptx) = guard.get(module_name) {
            return Ok(ptx.clone());
        }
    }
    let ptx = compile_ptx(src)?;
    let mut guard = match cache.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    Ok(guard.entry(module_name.to_string()).or_insert(ptx).clone())
}

/// Compile `src` (cached by `module_name`) and load `fn_name` (cached on device).
pub fn load_func(
    dev: &candle_core::CudaDevice,
    fn_name: &str,
    module_name: &str,
    src: &str,
) -> Result<impl core::ops::Deref<Target = cudarc::driver::CudaFunction>> {
    let ptx = cached_ptx(module_name, src)?;
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
