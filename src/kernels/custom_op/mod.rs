// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! Candle `CustomOp` kernels that stay on Candle storage.
//!
//! G0 path: `CpuStorage` / `CudaStorage` only — never `to_vec1` into CubeCL.
//! CubeCL Flash Attention still host-roundtrips
//! ([`crate::kernels::cubecl::interop_requires_host_roundtrip`]).
//!
//! Scope: f32 everywhere; RMSNorm / SwiGLU / attention also accept f16 / bf16
//! (float accumulate). LayerNorm, GeGLU, RoPE (+ position_ids), chunked CE,
//! fused linear+CE stay f32. CUDA attention is tiled SRAM FA for dim≤128.

pub mod attention;
pub mod ce;
mod cpu_isa;
pub mod fused_ce;
pub mod geglu;
pub mod layernorm;
#[cfg(feature = "cuda")]
pub mod nvrtc;
mod ptx_cache;
pub mod rmsnorm;
pub mod rope;
pub mod swiglu;

pub use attention::{
    attention_custom_op, attention_custom_op_softcap, attention_custom_op_window, attention_device,
    attention_device_softcap, attention_device_window, AttentionOp,
};
pub use ce::{chunked_cross_entropy, ChunkedCrossEntropyOp, DEFAULT_CE_CHUNK};
pub use fused_ce::{
    fused_linear_ce_avoids_full_logits, fused_linear_cross_entropy, FusedLinearCrossEntropyOp,
};
pub use geglu::{geglu_custom_op, GeGluOp};
pub use layernorm::{layernorm_custom_op, LayerNormOp};
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
///
/// RMSNorm, SwiGLU, and attention accept f32, f16, and bf16. Other ops stay f32-only.
#[must_use]
pub const fn custom_op_f32_only() -> bool {
    false
}

/// f32, f16, and bf16 are the supported CustomOp dtypes for rmsnorm/swiglu/attention.
#[must_use]
pub fn is_f32_or_f16(dtype: candle_core::DType) -> bool {
    matches!(
        dtype,
        candle_core::DType::F32 | candle_core::DType::F16 | candle_core::DType::BF16
    )
}

#[cfg(test)]
mod tests {
    #[test]
    fn honesty_flags() {
        assert!(super::custom_op_device_resident());
        assert!(!super::custom_op_f32_only());
        assert!(super::is_f32_or_f16(candle_core::DType::F16));
        assert!(super::is_f32_or_f16(candle_core::DType::BF16));
        assert!(crate::kernels::cubecl::interop_requires_host_roundtrip());
    }
}
