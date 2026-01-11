// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! Error types for unsloth-rs.

use thiserror::Error;

/// Result type alias for unsloth-rs operations.
pub type Result<T> = std::result::Result<T, UnslothError>;

/// Errors that can occur in unsloth-rs operations.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum UnslothError {
    /// GPU kernel error.
    #[error("kernel error: {0}")]
    Kernel(String),

    /// Out of memory.
    #[error("out of memory: required {required} bytes, available {available} bytes")]
    OutOfMemory {
        /// Required memory in bytes
        required: usize,
        /// Available memory in bytes
        available: usize,
    },

    /// Device not available.
    #[error("device not available: {0}")]
    DeviceNotAvailable(String),

    /// Shape mismatch.
    #[error("shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        /// Expected shape
        expected: Vec<usize>,
        /// Actual shape
        actual: Vec<usize>,
    },

    /// Invalid configuration.
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    /// Quantization error.
    #[error("quantization error: {0}")]
    Quantization(String),

    /// Ternary operation error.
    #[error("ternary operation error: {0}")]
    Ternary(String),

    /// Candle error.
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),
}
