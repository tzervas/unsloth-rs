//! # unsloth-rs
//!
//! Memory-optimized LLM fine-tuning with custom GPU kernels.
//!
//! This crate provides optimized implementations that achieve:
//! - 2-5x training speedups
//! - 70-80% VRAM reduction
//! - Cross-platform GPU support via CubeCL
//!
//! ## Features
//!
//! - **Fused Attention** - Combined QKV projection + attention + output in single kernel
//! - **Gradient Checkpointing** - Recompute vs store activations
//! - **Memory-Efficient Backward** - Chunked gradient computation
//! - **Mixed Precision** - Automatic bf16/f16 handling
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use unsloth_rs::kernels::FusedAttention;
//! use candle_core::Device;
//!
//! let attention = FusedAttention::new(
//!     768,  // hidden_size
//!     12,   // num_heads
//!     64,   // head_dim
//!     &Device::Cuda(0),
//! )?;
//! ```

#![warn(missing_docs)]
#![warn(clippy::pedantic)]

pub mod error;
pub mod kernels;
pub mod memory;
pub mod training;

pub use error::{Result, UnslothError};
