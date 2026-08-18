// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! NVRTC compile + `CudaDevice::get_or_load_custom_func` helper.
//!
//! Only compiled with `--features cuda`. Used by every CustomOp CUDA path so
//! we do not copy-paste `compile_ptx_with_opts` four times.
//!
//! `load_func` caches C→PTX by `module_name` as `Arc<str>`.
//! `get_or_load_custom_func` already caches the device function; compiling
//! on every call was the launch-tax leak. A Mutex plus Candle dispatch still
//! run on every launch — `compile_cached` is **not** a launch-tax close.

use std::sync::Arc;

use candle_core::cuda::{cudarc, WrapErr};
use candle_core::Result;

pub use super::ptx_cache::{next_pow2, ptx_compile_count, sorted_percentile};

/// Compile CUDA C to PTX (fast-math on).
pub fn compile_ptx(src: &str) -> Result<String> {
    let opts = cudarc::nvrtc::CompileOptions {
        use_fast_math: Some(true),
        ..Default::default()
    };
    let ptx = cudarc::nvrtc::safe::compile_ptx_with_opts(src, opts).w()?;
    Ok(ptx.to_src().to_string())
}

/// Cached C→PTX. Hits clone an `Arc`, not the PTX text.
pub fn cached_ptx(module_name: &str, src: &str) -> Result<Arc<str>> {
    super::ptx_cache::global_ptx_cache().get_or_insert(module_name, || compile_ptx(src))
}

/// Compile `src` (cached by `module_name`) and load `fn_name` (cached on device).
pub fn load_func(
    dev: &candle_core::CudaDevice,
    fn_name: &str,
    module_name: &str,
    src: &str,
) -> Result<impl core::ops::Deref<Target = cudarc::driver::CudaFunction>> {
    let ptx = cached_ptx(module_name, src)?;
    dev.get_or_load_custom_func(fn_name, module_name, ptx.as_ref())
}
