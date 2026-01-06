//! # unsloth-rs
//!
//! Rust implementations of transformer building blocks for LLM inference and fine-tuning.
//!
//! ## What This Crate Provides
//!
//! This crate provides common transformer operations built on [Candle](https://github.com/huggingface/candle):
//!
//! - **Multi-head attention**: Core attention mechanism with grouped-query attention (GQA) support
//! - **Rotary position embeddings (RoPE)**: Position encoding used in modern LLMs
//! - **RMS normalization**: Efficient normalization layer used in LLaMA-style models
//! - **SwiGLU activation**: Gated activation function for transformer MLPs
//! - **Memory estimation utilities**: Tools for tracking and estimating memory usage
//!
//! ## Why This Crate?
//!
//! This crate provides a Rust-native implementation of transformer components,
//! offering type safety and memory safety guarantees. The implementations are
//! designed to be clear and maintainable, serving as reference implementations
//! that can be extended with optimized GPU kernels.
//!
//! ## Current Status
//!
//! Current implementations are CPU reference implementations with GPU dispatch
//! via Candle's CUDA backend. Fused GPU kernels using CubeCL are planned for
//! future versions.
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
