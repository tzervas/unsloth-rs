// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! Hook for [`triton-bridge`](https://github.com/tzervas/triton-bridge-rs) `v0.2`.
//!
//! **No Cargo dependency yet.** 0.2 can load PTX on a real GPU (`--features
//! cuda`), but CPU publishes of unsloth-rs must not hard-depend on a crate
//! that is `FAIL_ENV` without `libcuda`. The reserved feature `triton-bridge`
//! exists so callers can `cfg` now; [`triton_bridge_ready`] is **always
//! `false`** in this tree until:
//!
//! 1. a 5080 run launches FA on a device pointer (triton-bridge GPU_HANDOFF)
//! 2. that crate is tagged / on crates.io as a ready 0.2.x
//! 3. this module calls `triton_bridge::bridge_ready()` instead of `false`
//!
//! Default attention stays [`super::attention_device`] (no host CubeCL copy).
//! RMS / RoPE / SwiGLU / CE stay CustomOp — do not FFI those.

/// Feature `triton-bridge` is compiled in. Does **not** mean a loader exists.
#[must_use]
pub const fn triton_bridge_feature_enabled() -> bool {
    cfg!(feature = "triton-bridge")
}

/// Mirror of `triton_bridge::bridge_ready()`. Hard-coded `false` in this tree.
#[must_use]
pub const fn triton_bridge_ready() -> bool {
    false
}

/// Why [`triton_bridge_ready`] is false (stable for logs).
#[must_use]
pub const fn triton_bridge_not_ready_reason() -> &'static str {
    "unsloth-rs: triton-bridge 0.2 loader exists in that crate; this tree has no Cargo dep until FA CUBIN launches on a device. See tzervas/triton-bridge-rs GPU_HANDOFF"
}

/// Dispatch helper: use the foreign bridge only when it is actually ready.
///
/// Today this always returns `false` so [`crate::kernels::attention_device`]
/// remains the path.
#[must_use]
pub fn should_dispatch_triton_bridge() -> bool {
    triton_bridge_feature_enabled() && triton_bridge_ready()
}

#[cfg(test)]
mod tests {
    #[test]
    fn never_dispatch_until_fa_payload() {
        assert!(!super::triton_bridge_ready());
        assert!(!super::should_dispatch_triton_bridge());
        assert!(super::triton_bridge_not_ready_reason().contains("triton-bridge"));
    }
}
