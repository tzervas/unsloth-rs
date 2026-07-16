// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! Optimized GPU kernels.
//!
//! This module provides CubeCL-accelerated GPU kernels for transformer operations:
//!
//! ## Core Operations
//! - [`cubecl`] - Flash Attention with online softmax (O(N) memory)
//! - [`fused_rmsnorm_rope`] - Fused RMSNorm + Rotary Position Embedding
//! - [`fused_swiglu`] - Fused SwiGLU activation for FFN blocks
//!
//! ## Legacy Operations (Candle-based)
//! - [`attention`] - Multi-head attention with GQA support
//! - [`rmsnorm`] - Standalone RMSNorm
//! - [`rope`] - Standalone Rotary Position Embedding
//! - [`swiglu`] - Standalone SwiGLU activation
//!
//! ## Specialized Operations
//! - [`ternary`] - Ternary bitsliced matrix multiplication

pub mod attention;
pub mod attention_cubecl;
pub mod cubecl;
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
