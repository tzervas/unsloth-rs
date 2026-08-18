// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! Candle `CustomOp` kernels that stay on Candle storage.
//!
//! G0 path: `CpuStorage` / `CudaStorage` only — never `to_vec1` into CubeCL.
//! CubeCL Flash Attention still host-roundtrips
//! ([`crate::kernels::cubecl::interop_requires_host_roundtrip`]).
//!
//! Scope: f32. RMSNorm, SwiGLU `silu⊙up`, RoPE (+ position_ids), chunked CE,
//! fused linear+CE (CPU), online attention.

pub mod attention;
pub mod ce;
mod cpu_isa;
pub mod fused_ce;
#[cfg(feature = "cuda")]
pub mod nvrtc;
mod ptx_cache;
pub mod rmsnorm;
pub mod rope;
pub mod swiglu;

pub use attention::{attention_custom_op, attention_device, AttentionOp};
pub use ce::{chunked_cross_entropy, ChunkedCrossEntropyOp, DEFAULT_CE_CHUNK};
pub use fused_ce::{
    fused_linear_ce_avoids_full_logits, fused_linear_cross_entropy, FusedLinearCrossEntropyOp,
};
pub use ptx_cache::{next_pow2, ptx_compile_count, sorted_percentile};
pub use rmsnorm::{rmsnorm_custom_op, RmsNormOp};
pub use rope::{rope_custom_op, rope_with_position_ids, RopeOp};
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
