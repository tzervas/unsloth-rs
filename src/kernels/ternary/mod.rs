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
//! ## Performance Targets
//!
//! - **Speedup**: ≥5x vs FP16 matmul on sparse pruned models
//! - **Memory**: ≥10x weight reduction (targeting 20-30x with sparsity)
//! - **Accuracy**: <2% perplexity degradation post-calibration

pub mod attention;
// TODO: Re-enable once CubeCL API compatibility is fixed
// #[cfg(feature = "cuda")]
// pub mod attention_cubecl;
pub mod config;
pub mod linear;
pub mod matmul;
// TODO: Re-enable once CubeCL API compatibility is fixed
// #[cfg(feature = "cuda")]
// pub mod matmul_cubecl;
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
