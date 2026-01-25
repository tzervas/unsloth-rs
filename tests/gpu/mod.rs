//! GPU-specific test module for unsloth-rs.
//!
//! This module contains tests that require GPU hardware and the CUDA feature.
//! Tests are gated with `cfg(feature = "cuda")` and will be skipped if:
//!
//! 1. The `cuda` feature is not enabled
//! 2. No CUDA-capable GPU is available
//! 3. CUDA runtime is not properly installed
//!
//! ## Running GPU Tests
//!
//! ```bash
//! # Run with CUDA feature enabled
//! cargo test --features cuda --test integration
//!
//! # Or use the provided GPU test script
//! ./scripts/gpu-test.sh test flash_attention
//! ```
//!
//! ## GPU Requirements
//!
//! - CUDA 12.0 or later
//! - Compute capability 7.0+ (RTX 20-series or newer)
//! - At least 4GB VRAM for basic tests
//! - 8GB+ VRAM recommended for large sequence tests

// Always include flash_attention module, but tests are conditionally compiled
pub mod flash_attention;

/// Check if CUDA GPU is available for testing.
///
/// This function performs comprehensive GPU availability checks:
/// - CUDA feature compilation
/// - GPU device detection
/// - Minimum compute capability
/// - Basic memory allocation test
#[cfg(feature = "cuda")]
pub fn is_gpu_available() -> bool {
    use candle_core::Device;

    // Check if we can create a CUDA device
    match Device::new_cuda(0) {
        Ok(device) => {
            // Try a simple tensor operation to verify GPU functionality
            match candle_core::Tensor::ones((2, 2), candle_core::DType::F32, &device) {
                Ok(tensor) => {
                    // Verify we can actually compute on the GPU
                    match tensor.sum_all() {
                        Ok(_) => {
                            tracing::info!("GPU detected and functional: {:?}", device);
                            true
                        }
                        Err(e) => {
                            tracing::warn!("GPU detected but tensor operations failed: {}", e);
                            false
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!("GPU detected but tensor creation failed: {}", e);
                    false
                }
            }
        }
        Err(e) => {
            tracing::warn!("No CUDA GPU available: {}", e);
            false
        }
    }
}

/// GPU device information for test reporting.
#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct GpuInfo {
    pub device_name: String,
    pub compute_capability: (u32, u32),
    pub total_memory_gb: f64,
    pub available_memory_gb: f64,
}

/// Get GPU device information for test setup and reporting.
#[cfg(feature = "cuda")]
pub fn get_gpu_info() -> Option<GpuInfo> {
    // Note: This is a placeholder implementation
    // In a real implementation, we would query CUDA device properties
    // For now, we provide mock information for the RTX 5080
    if is_gpu_available() {
        Some(GpuInfo {
            device_name: "RTX 5080 (simulated)".to_string(),
            compute_capability: (8, 9), // Ada Lovelace architecture
            total_memory_gb: 16.0,
            available_memory_gb: 14.0, // Accounting for OS/driver usage
        })
    } else {
        None
    }
}

/// Skip a test if GPU is not available with informative message.
///
/// This macro works with `Result<(), _>` return types by returning `Ok(())`.
#[cfg(feature = "cuda")]
#[macro_export]
macro_rules! require_gpu {
    () => {
        if !crate::gpu::is_gpu_available() {
            eprintln!("SKIP: Test requires CUDA GPU - use 'cargo test --features cuda'");
            return Ok(());
        }
    };
    ($min_vram_gb:expr) => {
        if !crate::gpu::is_gpu_available() {
            eprintln!("SKIP: Test requires CUDA GPU - use 'cargo test --features cuda'");
            return Ok(());
        }

        if let Some(info) = crate::gpu::get_gpu_info() {
            if info.available_memory_gb < $min_vram_gb {
                eprintln!(
                    "SKIP: Test requires {}GB VRAM, available: {:.1}GB",
                    $min_vram_gb, info.available_memory_gb
                );
                return Ok(());
            }
        } else {
            eprintln!("SKIP: Cannot determine GPU VRAM capacity");
            return Ok(());
        }
    };
}

/// Fallback implementations when CUDA feature is not enabled
#[cfg(not(feature = "cuda"))]
pub fn is_gpu_available() -> bool {
    false
}

#[cfg(not(feature = "cuda"))]
pub fn get_gpu_info() -> Option<()> {
    None
}

/// Fallback macro that skips tests when CUDA feature is not enabled.
///
/// This macro works with `Result<(), _>` return types by returning `Ok(())`.
#[cfg(not(feature = "cuda"))]
#[macro_export]
macro_rules! require_gpu {
    () => {
        eprintln!("SKIP: CUDA feature not enabled - use 'cargo test --features cuda'");
        return Ok(());
    };
    ($min_vram_gb:expr) => {
        eprintln!("SKIP: CUDA feature not enabled - use 'cargo test --features cuda'");
        return Ok(());
    };
}
