// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! Candle `CustomOp` kernels that stay on Candle storage.
//!
//! This is the **G0** path from `docs/ECOSYSTEM_GAPS.md`: operate on
//! `CpuStorage` / `CudaStorage` inside Candle so we never `to_vec1` through
//! host to reach CubeCL.
//!
//! CubeCL Flash Attention still requires a host round-trip
//! ([`crate::kernels::cubecl::interop_requires_host_roundtrip`]). These
//! CustomOps do **not**.
//!
//! Scope for 1.0.4: f32 RMSNorm only. Fused CE / RoPE / SwiGLU follow the same
//! pattern after this lands.

pub mod rmsnorm;

pub use rmsnorm::{rmsnorm_custom_op, RmsNormOp};

/// Returns `true` if the CustomOp RMSNorm forward path stays on Candle storage
/// (no CubeCL handle, no `Tensor::to_vec1` D2H).
///
/// Always `true` once this module is compiled. CUDA still needs `--features
/// cuda` and a device; without that the CPU CustomOp is used.
#[must_use]
pub const fn custom_op_device_resident() -> bool {
    true
}

/// Dtype scope for CustomOp kernels in 1.0.4.
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
        // CubeCL path is a different plane — still host-roundtrip.
        assert!(crate::kernels::cubecl::interop_requires_host_roundtrip());
    }
}
