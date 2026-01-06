//! # unsloth-rs
//!
//! Rust implementations of transformer building blocks for LLM inference and fine-tuning.
//!
//! This crate provides common transformer operations built on [Candle](https://github.com/huggingface/candle):
//!
//! - Multi-head attention with grouped-query attention (GQA) support
//! - Rotary position embeddings (RoPE)
//! - RMS normalization
//! - SwiGLU activation
//! - Memory estimation utilities
//!
//! ## Status
//!
//! Current implementations are CPU reference implementations with GPU dispatch
//! via Candle's CUDA backend. Fused GPU kernels are planned for future versions.
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

pub mod error;
pub mod kernels;
pub mod memory;
pub mod training;

pub use error::{Result, UnslothError};
