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

/// Device buffer for a CustomOp output. Zeroed — Candle's `alloc` is `unsafe`
/// only because it skips init. We do not need uninitialized memory here.
pub fn alloc_f32(
    dev: &candle_core::CudaDevice,
    n: usize,
) -> Result<cudarc::driver::CudaSlice<f32>> {
    dev.alloc_zeros::<f32>(n)
}

/// 1-D launch config after range checks. Call sites should not build raw
/// `LaunchConfig` with unchecked `as u32` casts.
pub fn launch_config(
    grid_x: usize,
    block_x: usize,
    shared_mem_bytes: usize,
) -> Result<cudarc::driver::LaunchConfig> {
    if block_x == 0 || block_x > 1024 {
        candle_core::bail!("CustomOp CUDA block_x {block_x} must be in 1..=1024");
    }
    if grid_x == 0 {
        candle_core::bail!("CustomOp CUDA grid_x must be > 0");
    }
    if shared_mem_bytes > 48 * 1024 {
        candle_core::bail!("CustomOp CUDA shared mem {shared_mem_bytes} exceeds 48KiB");
    }
    let grid = u32::try_from(grid_x).map_err(|_| {
        candle_core::Error::Msg(format!("CustomOp CUDA grid_x {grid_x} exceeds u32")).bt()
    })?;
    let block = u32::try_from(block_x).map_err(|_| {
        candle_core::Error::Msg(format!("CustomOp CUDA block_x {block_x} exceeds u32")).bt()
    })?;
    let smem = u32::try_from(shared_mem_bytes).map_err(|_| {
        candle_core::Error::Msg(format!(
            "CustomOp CUDA shared mem {shared_mem_bytes} exceeds u32"
        ))
        .bt()
    })?;
    Ok(cudarc::driver::LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: smem,
    })
}

/// Submit a CustomOp kernel. This is the **only** `unsafe` on the CustomOp
/// CUDA path.
///
/// cudarc's `LaunchArgs::launch` is `unsafe` because it cannot check kernel
/// ABI, argument mutability, or that the PTX stays in-bounds. Use-after-free
/// and cross-stream races are already handled by cudarc's event tracking.
///
/// Callers must have pushed arguments that match the loaded PTX (`fn_name`
/// from [`load_func`]) in order and type. Output slices come from
/// [`alloc_f32`]. `cfg` comes from [`launch_config`].
pub fn launch(
    builder: &mut cudarc::driver::LaunchArgs<'_>,
    cfg: cudarc::driver::LaunchConfig,
) -> Result<()> {
    // SAFETY: LaunchConfig is range-checked. Slices pushed as `arg` are live
    // `CudaSlice`/`CudaView` so cudarc records read/write events. Remaining
    // unsafety is the C kernel ABI, which Rust cannot express; it lives here
    // instead of at every call site.
    unsafe { builder.launch(cfg) }.w()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::launch_config;

    #[test]
    fn launch_config_rejects_empty_block() {
        assert!(launch_config(1, 0, 0).is_err());
    }

    #[test]
    fn launch_config_rejects_huge_smem() {
        assert!(launch_config(1, 32, 49 * 1024).is_err());
    }

    #[test]
    fn launch_config_accepts_one_block() {
        let cfg = launch_config(4, 128, 256).unwrap();
        assert_eq!(cfg.grid_dim, (4, 1, 1));
        assert_eq!(cfg.block_dim, (128, 1, 1));
        assert_eq!(cfg.shared_mem_bytes, 256);
    }
}
