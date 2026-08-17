// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! Candle `CustomOp` kernels that stay on Candle storage.
//!
//! G0 path: `CpuStorage` / `CudaStorage` only — never `to_vec1` into CubeCL.
//! CubeCL Flash Attention still host-roundtrips
//! ([`crate::kernels::cubecl::interop_requires_host_roundtrip`]).
//!
//! Scope: f32. RMSNorm, SwiGLU `silu⊙up`, RoPE apply, chunked CE, online attention.

pub mod attention;
pub mod ce;
#[cfg(feature = "cuda")]
pub mod nvrtc;
pub mod rmsnorm;
pub mod rope;
pub mod swiglu;

pub use attention::{attention_custom_op, attention_device, AttentionOp};
pub use ce::{chunked_cross_entropy, ChunkedCrossEntropyOp, DEFAULT_CE_CHUNK};
pub use rmsnorm::{rmsnorm_custom_op, RmsNormOp};
pub use rope::{rope_custom_op, RopeOp};
pub use swiglu::{swiglu_custom_op, SwiGluOp};

/// CustomOp forward paths stay on Candle storage (no CubeCL handle, no D2H).
#[must_use]
pub const fn custom_op_device_resident() -> bool {
    true
}

/// Dtype scope for CustomOp kernels.
#[must_use]
pub const fn custom_op_f32_only() -> bool {
    true
}

#[cfg(test)]
mod tests {
    #[test]
    fn honesty_flags() {
        assert!(super::custom_op_device_resident());
        assert!(super::custom_op_f32_only());
        assert!(crate::kernels::cubecl::interop_requires_host_roundtrip());
    }
}
