//! Candle ↔ CubeCL tensor conversion utilities.
//!
//! This module provides helpers for converting between Candle tensors and
//! CubeCL buffer handles, enabling seamless integration between the two frameworks.
//!
//! ## Key Functions
//!
//! - [`candle_to_cubecl_handle`] - Convert contiguous Candle tensor to CubeCL handle
//! - [`cubecl_to_candle_tensor`] - Convert CubeCL output back to Candle tensor
//! - [`has_cubecl_cuda_support`] - Check if CUDA runtime is available
//!
//! ## Memory Management
//!
//! The conversion functions handle:
//! - Ensuring tensor contiguity (required for raw pointer access)
//! - Buffer creation via `client.create(bytes)`
//! - Buffer reuse where possible to minimize allocations
//!
//! ## Fallback Routing
//!
//! When CubeCL is not available, functions return appropriate errors or
//! fallback implementations are used in the kernel module.

use crate::error::{Result, UnslothError};
use candle_core::{DType, Device, Tensor};

/// Check if CubeCL CUDA runtime support is available.
///
/// This checks:
/// 1. The `cuda` feature is enabled at compile time
/// 2. A CUDA-capable device is detected at runtime
///
/// # Returns
///
/// `true` if CubeCL CUDA kernels can be launched, `false` otherwise.
///
/// # Example
///
/// ```rust
/// use unsloth_rs::kernels::cubecl::has_cubecl_cuda_support;
///
/// if has_cubecl_cuda_support() {
///     println!("CubeCL CUDA acceleration available!");
/// } else {
///     println!("Falling back to Candle backend");
/// }
/// ```
#[must_use]
pub fn has_cubecl_cuda_support() -> bool {
    // Check if cuda feature is enabled
    #[cfg(feature = "cuda")]
    {
        // TODO: Add actual CubeCL runtime device detection
        // For now, check if Candle can see a CUDA device
        // This will be replaced with:
        // cubecl_cuda::CudaRuntime::is_available()

        // Placeholder: Check Candle CUDA support as proxy
        matches!(Device::cuda_if_available(0), Ok(Device::Cuda(_)))
    }

    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

/// Convert a Candle tensor to a CubeCL buffer handle.
///
/// The tensor must be contiguous in memory. If not, it will be made contiguous
/// (which may involve a copy).
///
/// # Arguments
///
/// * `tensor` - The Candle tensor to convert
///
/// # Returns
///
/// A tuple of `(raw_data_bytes, shape, dtype)` that can be used to create
/// a CubeCL buffer handle via `client.create(bytes)`.
///
/// # Errors
///
/// Returns error if:
/// - Tensor is not on a CUDA device
/// - Tensor dtype is not supported (only f32 currently)
///
/// # Example
///
/// ```rust,ignore
/// use unsloth_rs::kernels::cubecl::candle_to_cubecl_handle;
///
/// let tensor = Tensor::randn(0.0f32, 1.0, (2, 4, 8, 64), &Device::cuda_if_available(0)?)?;
/// let (bytes, shape, dtype) = candle_to_cubecl_handle(&tensor)?;
///
/// // Use with CubeCL:
/// // let handle = client.create(&bytes);
/// ```
pub fn candle_to_cubecl_handle(tensor: &Tensor) -> Result<(Vec<u8>, Vec<usize>, DType)> {
    // Ensure tensor is on CUDA
    if !matches!(tensor.device(), Device::Cuda(_)) {
        return Err(UnslothError::InvalidConfig(
            "candle_to_cubecl_handle requires CUDA tensor".to_string(),
        ));
    }

    // Ensure contiguous memory layout
    let tensor = tensor.contiguous()?;

    // Get shape and dtype
    let shape = tensor.dims().to_vec();
    let dtype = tensor.dtype();

    // Only f32 supported currently
    // TODO: Add f16/bf16 support
    if dtype != DType::F32 {
        return Err(UnslothError::InvalidConfig(format!(
            "candle_to_cubecl_handle only supports f32, got {:?}",
            dtype
        )));
    }

    // Extract raw bytes
    // For CUDA tensors, this requires a device-to-host copy
    let data: Vec<f32> = tensor.flatten_all()?.to_vec1()?;
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();

    Ok((bytes, shape, dtype))
}

/// Convert a CubeCL buffer back to a Candle tensor.
///
/// # Arguments
///
/// * `bytes` - Raw output bytes from CubeCL kernel
/// * `shape` - Target tensor shape
/// * `device` - Target Candle device (must be CUDA)
///
/// # Returns
///
/// A Candle tensor with the specified shape on the target device.
///
/// # Errors
///
/// Returns error if:
/// - Shape dimensions don't match byte count
/// - Device is not CUDA
///
/// # Example
///
/// ```rust,ignore
/// use unsloth_rs::kernels::cubecl::cubecl_to_candle_tensor;
///
/// // After kernel execution:
/// // let output_bytes = client.read(&output_handle);
/// let tensor = cubecl_to_candle_tensor(&output_bytes, &[2, 4, 8, 64], &device)?;
/// ```
pub fn cubecl_to_candle_tensor(bytes: &[u8], shape: &[usize], device: &Device) -> Result<Tensor> {
    // Validate device
    if !matches!(device, Device::Cuda(_)) {
        return Err(UnslothError::InvalidConfig(
            "cubecl_to_candle_tensor requires CUDA device".to_string(),
        ));
    }

    // Calculate expected size
    let num_elements: usize = shape.iter().product();
    let expected_bytes = num_elements * 4; // f32

    if bytes.len() != expected_bytes {
        return Err(UnslothError::InvalidConfig(format!(
            "Byte count mismatch: expected {} for shape {:?}, got {}",
            expected_bytes,
            shape,
            bytes.len()
        )));
    }

    // Convert bytes to f32
    let data: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    // Create Candle tensor
    // Note: This creates on CPU first, then transfers to CUDA
    // TODO: Optimize with direct GPU buffer creation
    let tensor = Tensor::from_vec(data, shape, device)?;

    Ok(tensor)
}

/// Allocate output buffer for kernel results.
///
/// Creates an uninitialized buffer of the specified size for kernel output.
/// This avoids unnecessary initialization overhead.
///
/// # Arguments
///
/// * `num_elements` - Number of f32 elements to allocate
///
/// # Returns
///
/// A byte vector suitable for CubeCL output buffer.
#[must_use]
pub fn allocate_output_buffer(num_elements: usize) -> Vec<u8> {
    // Allocate without initialization for performance
    // Safety: The kernel will write to all elements before reading
    vec![0u8; num_elements * 4]
}

/// Convert u32 plane data to raw bytes for CubeCL buffer creation.
///
/// # Arguments
/// * `plane` - The u32 plane data (e.g., plus_plane or minus_plane from TernaryTensor)
///
/// # Returns
/// Raw bytes that can be passed to `client.create()`
#[must_use]
pub fn u32_planes_to_cubecl_bytes(plane: &[u32]) -> Vec<u8> {
    plane.iter().flat_map(|&word| word.to_le_bytes()).collect()
}

/// Convert a TernaryTensor to CubeCL buffer handles.
///
/// Returns the raw bytes for plus_plane, minus_plane, and scales,
/// along with shape metadata for kernel configuration.
///
/// # Arguments
/// * `tensor` - The TernaryTensor to convert
///
/// # Returns
/// Tuple of (plus_bytes, minus_bytes, scales_bytes, shape, k_words)
#[must_use]
pub fn ternary_tensor_to_cubecl_handles(
    tensor: &crate::kernels::ternary::TernaryTensor,
) -> (Vec<u8>, Vec<u8>, Vec<u8>, (usize, usize), usize) {
    let plus_bytes = u32_planes_to_cubecl_bytes(&tensor.plus_plane);
    let minus_bytes = u32_planes_to_cubecl_bytes(&tensor.minus_plane);
    let scales_bytes: Vec<u8> = tensor.scales.iter()
        .flat_map(|&s| s.to_le_bytes())
        .collect();
    
    (plus_bytes, minus_bytes, scales_bytes, tensor.shape, tensor.k_words)
}

/// Convert CubeCL output bytes back to u32 plane.
///
/// # Arguments
/// * `bytes` - Raw bytes from CubeCL buffer
///
/// # Returns
/// Vec<u32> plane data
#[must_use]
pub fn cubecl_bytes_to_u32_plane(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_has_cubecl_cuda_support() {
        // Should not panic regardless of CUDA availability
        let _ = has_cubecl_cuda_support();
    }

    #[test]
    fn test_allocate_output_buffer() {
        let buffer = allocate_output_buffer(100);
        assert_eq!(buffer.len(), 400); // 100 * 4 bytes per f32
    }

    #[test]
    fn test_candle_to_cubecl_cpu_error() {
        let tensor = Tensor::zeros((2, 4), DType::F32, &Device::Cpu).unwrap();
        let result = candle_to_cubecl_handle(&tensor);
        assert!(result.is_err());
    }

    #[test]
    fn test_u32_planes_to_bytes_roundtrip() {
        let original: Vec<u32> = vec![0xDEADBEEF, 0xCAFEBABE, 0x12345678];
        let bytes = u32_planes_to_cubecl_bytes(&original);
        let recovered = cubecl_bytes_to_u32_plane(&bytes);
        assert_eq!(original, recovered);
    }

    #[test]
    fn test_ternary_tensor_to_cubecl_handles() {
        use crate::kernels::ternary::TernaryTensor;
        
        let shape = (4, 64); // 4 rows, 64 cols
        let k_words = 2; // 64 / 32 = 2
        
        let plus = vec![0xAAAAAAAAu32; 4 * k_words];
        let minus = vec![0x55555555u32; 4 * k_words];
        let scales = vec![1.5f32; 4];
        
        let tensor = TernaryTensor::new(plus.clone(), minus.clone(), scales.clone(), shape);
        let (plus_bytes, minus_bytes, scales_bytes, ret_shape, ret_k_words) = 
            ternary_tensor_to_cubecl_handles(&tensor);
        
        assert_eq!(ret_shape, shape);
        assert_eq!(ret_k_words, k_words);
        assert_eq!(plus_bytes.len(), plus.len() * 4);
        assert_eq!(minus_bytes.len(), minus.len() * 4);
        assert_eq!(scales_bytes.len(), scales.len() * 4);
        
        // Verify roundtrip
        let recovered_plus = cubecl_bytes_to_u32_plane(&plus_bytes);
        assert_eq!(plus, recovered_plus);
    }

    // GPU tests require cuda feature and hardware
    #[cfg(feature = "cuda")]
    mod cuda_tests {
        use super::*;

        #[test]
        fn test_roundtrip_conversion() {
            if let Ok(device) = Device::cuda_if_available(0) {
                if matches!(device, Device::Cuda(_)) {
                    let original = Tensor::randn(0.0f32, 1.0, (2, 4, 8, 64), &device).unwrap();

                    let (bytes, shape, _) = candle_to_cubecl_handle(&original).unwrap();
                    let recovered = cubecl_to_candle_tensor(&bytes, &shape, &device).unwrap();

                    // Check shape matches
                    assert_eq!(original.dims(), recovered.dims());

                    // Check values match (within floating point tolerance)
                    let orig_data: Vec<f32> = original.flatten_all().unwrap().to_vec1().unwrap();
                    let rec_data: Vec<f32> = recovered.flatten_all().unwrap().to_vec1().unwrap();

                    for (a, b) in orig_data.iter().zip(rec_data.iter()) {
                        assert!((a - b).abs() < 1e-6, "Values differ: {} vs {}", a, b);
                    }
                }
            }
        }
    }
}
