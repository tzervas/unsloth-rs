// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! Hook for [`triton-bridge`](https://github.com/tzervas/triton-bridge-rs) `v0.1.0`.
//!
//! **No Cargo dependency yet.** A `NotReady` crate on crates.io would break
//! nothing but would pin unsloth-rs publishes to a stub. The reserved feature
//! `triton-bridge` exists so callers can `cfg` now; [`triton_bridge_ready`] is
//! **always `false`** until:
//!
//! 1. triton-bridge-rs Phase 1 sets `bridge_ready()` (device-pointer launch)
//! 2. that crate is on crates.io (or we accept a git dep on a private build)
//! 3. this module calls through instead of returning `false`
//!
//! Default attention stays [`super::attention_device`] (no host CubeCL copy).

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
    "unsloth-rs: triton-bridge v0.1.0 is a contract only; no Cargo dep, no launch. See tzervas/triton-bridge-rs#3"
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
    fn never_dispatch_until_phase1() {
        assert!(!super::triton_bridge_ready());
        assert!(!super::should_dispatch_triton_bridge());
        assert!(super::triton_bridge_not_ready_reason().contains("triton-bridge"));
    }
}
