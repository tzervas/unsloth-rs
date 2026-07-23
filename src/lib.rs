// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! # unsloth-rs
//!
//! Candle/CubeCL **transformer kernel building blocks** for LLM inference experiments.
//!
//! **Not** a product port of Python Unsloth: no LoRA/QLoRA trainer, model zoo, or
//! proven 2× / 70% VRAM training claims. Sister crates handle PEFT and orchestration.
//!
//! ## What This Crate Provides
//!
//! Common transformer operations built on [Candle](https://github.com/huggingface/candle):
//!
//! - **Multi-head attention**: Core attention with grouped-query attention (GQA) support
//! - **Rotary position embeddings (`RoPE`)**: Position encoding used in modern LLMs
//! - **RMS normalization**: Efficient normalization used in LLaMA-style models
//! - **`SwiGLU` activation**: Gated activation for transformer MLPs
//! - **Memory estimation utilities**: Activation / checkpoint *estimates* (not a trainer)
//!
//! ## Why This Crate?
//!
//! Rust-native reference implementations with optional CubeCL CUDA kernels.
//! Prefer correctness and honest scope over Unsloth product parity claims.
//!
//! ## Current Status
//!
//! CPU paths are the default, well-tested surface. GPU Flash Attention and fused
//! kernels exist under the `cuda` feature but require a healthy device/toolkit
//! (`CUDA_COMPUTE_CAP` pin may be required; see crate README / GPU_SETUP.md).
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use unsloth_rs::kernels::{FusedAttention, FusedAttentionConfig};
//! use candle_core::Device;
//!
//! let config = FusedAttentionConfig {
//!     hidden_size: 768,
//!     num_heads: 12,
//!     head_dim: 64,
//!     ..Default::default()
//! };
//! let attention = FusedAttention::new(config, &Device::Cpu)?;
//! ```

#![warn(missing_docs)]
#![warn(clippy::pedantic)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::implicit_hasher)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::struct_field_names)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::doc_overindented_list_items)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::type_complexity)]

pub mod error;
pub mod kernels;
pub mod memory;
pub mod training;

pub use error::{Result, UnslothError};
