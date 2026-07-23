// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! Ternary bitsliced operations for memory-efficient inference.
//!
//! This module implements single-trit {-1, 0, +1} bitsliced operations
//! based on TWN (Ternary Weight Networks) and HDC/VSA literature.
//!
//! ## Key Features
//!
//! - **Memory Efficiency**: 16-32x weight reduction via bitsliced representation
//! - **Popcount-Based Matmul**: Exact dot products via hardware popcount intrinsics
//! - **Sparsity Acceleration**: 95%+ zero weights enable plane/dim skipping
//!
//! ## Mathematical Foundation
//!
//! Ternary dot product via +plane/-plane bitsliced representation:
//!
//! ```text
//! dot(A, B) = popcount(A+ & B+) + popcount(A- & B-)
//!           - popcount(A+ & B-) - popcount(A- & B+)
//! scaled_dot = dot * scale_a * scale_b
//! ```
//!
//! ## Module Structure
//!
//! - [`config`] - Configuration for ternary kernels (thresholds, tile sizes)
//! - [`types`] - Core tensor types (`TernaryTensor`, `TernaryPlanes`)
//! - [`quantize`] - FP → ternary quantization with scale calibration
//! - [`matmul`] - Bitsliced matrix multiplication kernel
//! - [`linear`] - Drop-in `TernaryLinear` layer
//! - [`attention`] - Ternary attention scoring with online softmax
//!
//! ## Usage
//!
//! ```rust,ignore
//! use unsloth_rs::kernels::ternary::{TernaryTensor, TernaryLinear, quantize_weights};
//!
//! // Quantize FP32 weights to ternary
//! let (ternary_weights, scale) = quantize_weights(&fp_weights, TernaryConfig::default())?;
//!
//! // Create ternary linear layer
//! let layer = TernaryLinear::new(ternary_weights, scale, bias)?;
//!
//! // Forward pass (FP16 activations, ternary weights)
//! let output = layer.forward(&activations)?;
//! ```
//!
//! ## Scope (honest)
//!
//! - **In scope**: CPU ternary quantize / matmul / linear / attention experiments.
//! - **Out of scope**: GPU ternary CubeCL kernels (archived; see `archive/ternary_cubecl/`).
//! - Compression ratios are measurable on CPU; do **not** claim training speedups
//!   or Unsloth-style GPU wins from this module.

// ---------------------------------------------------------------------------
// UNS-P2-01 — Ternary CubeCL GPU modules: **NON-GOAL / archived**
//
// Historical drafts lived at `src/kernels/ternary/{attention,matmul}_cubecl.rs`.
// They were never exported on default or `cuda` builds and are not maintained
// against CubeCL 0.9. Source moved to `archive/ternary_cubecl/` (excluded from
// crates.io package). CPU ternary path below remains supported/experimental.
//
// Feature `_ternary_cubecl_todo` is an empty placeholder only — it does **not**
// compile archived GPU modules back into the crate.
// ---------------------------------------------------------------------------

pub mod attention;
pub mod config;
pub mod linear;
pub mod matmul;
pub mod model;
pub mod quantize;
pub mod types;

pub use attention::{
    should_use_ternary_attention, ternary_attention_cpu, TernaryAttentionConfig,
    TernaryAttentionWeights,
};
pub use config::TernaryConfig;
pub use linear::TernaryLinear;
pub use matmul::{ternary_matmul, ternary_matmul_cpu};
pub use model::{
    quantize_linear_layer, quantize_weights_collection, ModelQuantizationConfig, QuantizationStats,
    TernaryModel,
};
pub use quantize::{quantize_tensor, CalibrationMethod};
pub use types::{SparsityMetadata, TernaryPlanes, TernaryTensor};
