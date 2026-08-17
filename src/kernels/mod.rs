// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! Transformer kernels.
//!
//! ## G0 CustomOp (device-resident, no CubeCL host copy)
//! - [`custom_op`] — Candle `CustomOp*` on `CpuStorage` / `CudaStorage`
//!
//! ## CubeCL (optional, host D2H/H2D)
//! - [`cubecl`] — Flash Attention; [`cubecl::interop_requires_host_roundtrip`] is `true`
//! - [`fused_rmsnorm_rope`] / [`fused_swiglu`] — CubeCL drafts + CPU fallback
//!
//! ## Candle reference layers
//! - [`attention`] — Multi-head attention with GQA
//! - [`rmsnorm`] — RMSNorm layer (forwards through CustomOp)
//! - [`rope`] — Rotary Position Embedding
//! - [`swiglu`] — SwiGLU
//!
//! ## Specialized
//! - [`ternary`] — Ternary bitsliced matmul (CPU)

pub mod attention;
pub mod attention_cubecl;
pub mod cubecl;
pub mod custom_op;
pub mod fused_rmsnorm_rope;
pub mod fused_swiglu;
pub mod rmsnorm;
pub mod rope;
pub mod swiglu;
pub mod ternary;

// Core attention exports
pub use attention::{FusedAttention, FusedAttentionConfig};
pub use attention_cubecl::{flash_attention_cubecl, has_cubecl_support};
pub use cubecl::{flash_attention_kernel, FlashAttentionConfig};

// G0 CustomOp
pub use custom_op::{
    chunked_cross_entropy, custom_op_device_resident, custom_op_f32_only, rmsnorm_custom_op,
    rope_custom_op, swiglu_custom_op, ChunkedCrossEntropyOp, RmsNormOp, RopeOp, SwiGluOp,
    DEFAULT_CE_CHUNK,
};

// Legacy layer exports
pub use rmsnorm::RmsNorm;
pub use rope::RotaryEmbedding;
pub use swiglu::SwiGLU;

// Fused CubeCL kernel exports
#[cfg(feature = "cuda")]
pub use fused_rmsnorm_rope::{fused_rmsnorm_rope, rmsnorm as rmsnorm_cubecl, rope as rope_cubecl};
#[cfg(feature = "cuda")]
pub use fused_swiglu::{fused_ffn_swiglu, swiglu as swiglu_cubecl, swiglu_backward};

// Non-CUDA fallback exports (always available)
#[cfg(not(feature = "cuda"))]
pub use fused_rmsnorm_rope::{fused_rmsnorm_rope, rmsnorm as rmsnorm_cubecl, rope as rope_cubecl};
#[cfg(not(feature = "cuda"))]
pub use fused_swiglu::{fused_ffn_swiglu, swiglu as swiglu_cubecl, swiglu_backward};

// Ternary bitsliced operations
pub use ternary::{
    CalibrationMethod, SparsityMetadata, TernaryConfig, TernaryLinear, TernaryPlanes, TernaryTensor,
};
