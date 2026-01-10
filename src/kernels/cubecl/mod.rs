// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! `CubeCL` GPU kernel implementations for Flash Attention.
//!
//! This module provides memory-efficient GPU kernels using `CubeCL` v0.8.1.
//! The implementation follows the Flash Attention 2 algorithm with online softmax
//! for O(N) memory complexity instead of O(N²).
//!
//! ## Module Structure
//!
//! - [`config`] - Kernel configuration (tile sizes, launch parameters)
//! - [`interop`] - Candle ↔ `CubeCL` tensor conversion utilities
//! - [`kernel`] - Actual Flash Attention `CubeCL` kernel implementation
//!
//! ## Hardware Targets
//!
//! - **Phase 1**: `GeForce` RTX 5080 (primary development)
//! - **Phase 2**: `GeForce` RTX 3090 Ti (validation and tuning)
//! - **Future**: A100/H100, AMD MI series, WGPU/CPU backends
//!
//! ## Usage
//!
//! ```rust,ignore
//! use unsloth_rs::kernels::cubecl::{flash_attention_kernel, FlashAttentionConfig};
//!
//! let config = FlashAttentionConfig::default();
//! let output = flash_attention_kernel(&q, &k, &v, scale, mask, &config)?;
//! ```
//!
//! ## Implementation Status
//!
//! See [`FLASH_ATTENTION_IMPLEMENTATION_STATUS.md`] for current progress.

pub mod config;
pub mod interop;
pub mod kernel;

pub use config::FlashAttentionConfig;
pub use interop::{
    candle_to_cubecl_handle, cubecl_bytes_to_u32_plane, cubecl_to_candle_tensor,
    has_cubecl_cuda_support, ternary_tensor_to_cubecl_handles, u32_planes_to_cubecl_bytes,
};
pub use kernel::flash_attention_kernel;
