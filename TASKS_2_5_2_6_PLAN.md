# Phase 2 Tasks 2.5 & 2.6 Implementation Plan

**Date**: 2026-01-07  
**Status**: Awaiting Approval  
**Previous Tasks**: 2.1-2.4 Complete ✅

## Overview

This document outlines the detailed implementation plan for the final two tasks of Phase 2:
- **Task 2.5**: Plane Skipping with SparsityMetadata (2-3 days)
- **Task 2.6**: GPU Dispatch Integration (1 day)

Upon completion, Phase 2 will be fully implemented with all ternary matmul GPU kernels ready for validation.

---

## Task 2.5: Plane Skipping with SparsityMetadata

**Objective**: Optimize kernel performance on sparse models (95%+ sparsity) by skipping computation for all-zero weight planes.

**Expected Impact**:
- 2-4x speedup on 95%+ sparse models
- Minimal overhead on dense models (<5%)
- Dynamic adaptation based on runtime sparsity patterns

### Current State Analysis

**Already Implemented** ✅:
- `SparsityMetadata` struct in `src/kernels/ternary/types.rs`
- `from_planes()` - Creates metadata from TernaryPlanes
- `is_chunk_active()` - Checks if chunk has non-zero values
- `chunk_sparsity()` - Calculates effective sparsity
- Active chunk bitmap (u64 words, 64 chunks per word)

**What's Missing** ❌:
- Integration of SparsityMetadata into GPU kernels
- Kernel dispatch logic based on sparsity threshold
- GPU-side chunk checking and skipping
- Performance validation on sparse vs dense inputs

### Implementation Strategy

#### 5.1: Add Sparsity-Aware Configuration (1-2 hours)

**File**: `src/kernels/ternary/matmul_cubecl.rs`

Add configuration flag to enable/disable plane skipping:

```rust
/// Configuration for sparse-optimized kernels
#[derive(Clone, Copy, Debug)]
pub struct SparseOptimizedConfig {
    /// Base vectorized config
    pub base: VectorizedTernaryMatmulConfig,
    /// Enable plane skipping optimization
    pub enable_plane_skipping: bool,
    /// Minimum sparsity to enable optimization (default: 0.90)
    pub sparsity_threshold: f32,
    /// Chunk size for sparsity checking (default: 64 dims = 2 u32 words)
    pub chunk_size: u32,
}

impl SparseOptimizedConfig {
    /// Create from sparsity level
    pub fn from_sparsity(
        base: VectorizedTernaryMatmulConfig,
        sparsity: f32,
    ) -> Self {
        Self {
            base,
            enable_plane_skipping: sparsity >= 0.90,
            sparsity_threshold: 0.90,
            chunk_size: 64, // 2 u32 words
        }
    }

    /// RTX 5080 preset with sparsity optimization
    pub fn rtx_5080_sparse(
        m: u32, n: u32, k_words: u32, in_features: u32, sparsity: f32
    ) -> Self {
        let base = VectorizedTernaryMatmulConfig::rtx_5080_preset(
            m, n, k_words, in_features
        );
        Self::from_sparsity(base, sparsity)
    }
}
```

**Testing**:
- Verify config creation with various sparsity levels
- Test threshold logic (enable at 90%, disable below)

#### 5.2: Implement Sparse Kernel with Plane Skipping (1 day)

**File**: `src/kernels/ternary/matmul_cubecl.rs`

Create optimized kernel that skips inactive chunks:

```rust
/// Sparse-optimized ternary matmul kernel with plane skipping.
///
/// Uses sparsity metadata to skip computation for all-zero weight chunks.
/// Most effective on models with 95%+ sparsity.
///
/// Algorithm:
/// ```text
/// For each K tile:
///   - Load sparsity bitmap for this tile
///   - For each chunk in tile:
///     - Check if chunk is active (non-zero)
///     - If inactive: skip load, quantization, and popcount
///     - If active: perform normal computation
///   - Accumulate only active chunks
/// ```
#[cube(launch_unchecked)]
pub fn ternary_matmul_kernel_sparse<F: Float>(
    // Input activations [batch, in_features] as f32
    input: &Array<F>,
    // Weight positive plane [out_features, k_words] as u32
    w_plus: &Array<F>,
    // Weight negative plane [out_features, k_words] as u32
    w_minus: &Array<F>,
    // Per-row scales [out_features]
    scales: &Array<F>,
    // Sparsity metadata: active chunk bitmap [out_features, num_chunks_words]
    // Each u64 represents 64 chunks
    sparsity_bitmap: &Array<u64>,
    // Output [batch, out_features]
    output: &mut Array<F>,
    // Compile-time configuration
    #[comptime] config: SparseOptimizedConfig,
) {
    let batch_idx = CUBE_POS_X;
    let out_block_idx = CUBE_POS_Y;
    let thread_idx = UNIT_POS_X;

    if batch_idx >= config.base.m {
        return;
    }

    // Shared memory for input tile
    let tile_k_words = config.base.tile_k();
    let mut input_plus_tile = SharedMemory::<u32>::new(tile_k_words);
    let mut input_minus_tile = SharedMemory::<u32>::new(tile_k_words);

    // Each thread computes outputs_per_thread outputs
    for out_local in 0..config.base.outputs_per_thread {
        let out_idx = out_block_idx * config.base.block_size * config.base.outputs_per_thread
            + thread_idx * config.base.outputs_per_thread
            + out_local;

        if out_idx >= config.base.n {
            continue;
        }

        let input_offset = batch_idx * config.base.in_features;
        let weight_offset = out_idx * config.base.k_words;

        let mut pos_sum = 0u32;
        let mut neg_sum = 0u32;

        // Calculate chunk parameters
        let words_per_chunk = config.chunk_size / 32;  // 64 dims = 2 words
        let num_chunks = (config.base.k_words + words_per_chunk - 1) / words_per_chunk;
        let chunks_per_u64 = 64u32;
        
        // Sparsity bitmap offset for this output feature
        let bitmap_words = (num_chunks + chunks_per_u64 - 1) / chunks_per_u64;
        let bitmap_offset = out_idx * bitmap_words;

        // Number of K tiles
        let num_k_tiles = (config.base.k_words + tile_k_words - 1) / tile_k_words;

        // Process K dimension in tiles
        for k_tile in 0..num_k_tiles {
            let k_start = k_tile * tile_k_words;
            let k_end = u32::min(k_start + tile_k_words, config.base.k_words);
            let tile_size = k_end - k_start;

            // Cooperatively load input tile (same as vectorized kernel)
            // ... [omitted for brevity - same as vectorized kernel]

            sync_units();

            // Process tile with plane skipping
            let chunk_start = k_start / words_per_chunk;
            let chunk_end = (k_end + words_per_chunk - 1) / words_per_chunk;

            for chunk_idx in chunk_start..chunk_end {
                // Check if this chunk is active
                let bitmap_word_idx = chunk_idx / chunks_per_u64;
                let bit_idx = chunk_idx % chunks_per_u64;
                let bitmap = sparsity_bitmap[bitmap_offset + bitmap_word_idx];
                let is_active = (bitmap & (1u64 << bit_idx)) != 0u64;

                if !is_active {
                    // Skip this chunk entirely
                    continue;
                }

                // Process active chunk
                let chunk_word_start = chunk_idx * words_per_chunk;
                let chunk_word_end = u32::min(
                    chunk_word_start + words_per_chunk,
                    k_end
                );

                for k_word in chunk_word_start..chunk_word_end {
                    if k_word < k_start || k_word >= k_end {
                        continue;
                    }

                    let k_local = k_word - k_start;

                    // Load weight planes
                    let wp_f32 = w_plus[weight_offset + k_word];
                    let wm_f32 = w_minus[weight_offset + k_word];
                    let wp_bits = u32::reinterpret(wp_f32);
                    let wm_bits = u32::reinterpret(wm_f32);

                    // Load input planes from shared memory
                    let ip = input_plus_tile[k_local];
                    let im = input_minus_tile[k_local];

                    // Popcount-based ternary dot product
                    pos_sum = pos_sum + (wp_bits & ip).count_ones();
                    pos_sum = pos_sum + (wm_bits & im).count_ones();

                    neg_sum = neg_sum + (wp_bits & im).count_ones();
                    neg_sum = neg_sum + (wm_bits & ip).count_ones();
                }
            }

            sync_units();
        }

        // Convert popcount result to float and apply scale
        let dot = F::cast_from(pos_sum) - F::cast_from(neg_sum);
        let scale = scales[out_idx];
        output[batch_idx * config.base.n + out_idx] = dot * scale;
    }
}

/// Launch configuration for sparse kernel (same as vectorized)
pub fn get_sparse_launch_config(
    config: &SparseOptimizedConfig,
) -> (CubeCount, CubeDim) {
    get_vectorized_launch_config(&config.base)
}
```

**Key Features**:
- Comptime check for `enable_plane_skipping` (can be compiled out)
- Runtime chunk activity checking via bitmap
- Skips load, quantization, and popcount for inactive chunks
- Minimal overhead when all chunks are active (dense case)

**Testing**:
- Test with 50%, 75%, 90%, 95%, 99% sparsity
- Verify correctness matches non-sparse kernel
- Measure skip rate matches actual sparsity

#### 5.3: Add Helper Functions for Sparsity Bitmap Conversion (2-3 hours)

**File**: `src/kernels/cubecl/interop.rs`

Add conversion for sparsity metadata:

```rust
/// Convert SparsityMetadata active_chunks to CubeCL u64 array bytes.
///
/// # Arguments
/// * `metadata` - The SparsityMetadata from TernaryTensor
///
/// # Returns
/// Raw bytes for CubeCL u64 array
pub fn sparsity_metadata_to_cubecl_bytes(
    metadata: &crate::kernels::ternary::SparsityMetadata
) -> Vec<u8> {
    metadata.active_chunks
        .iter()
        .flat_map(|&word| word.to_le_bytes())
        .collect()
}

/// Create sparsity bitmap for entire tensor.
///
/// For each output feature (row), creates a chunk activity bitmap.
///
/// # Arguments
/// * `tensor` - The TernaryTensor with sparsity metadata
/// * `chunk_size` - Size of chunks in dimensions (default: 64)
///
/// # Returns
/// Flattened bitmap [out_features, bitmap_words] as bytes
pub fn create_sparsity_bitmap_for_tensor(
    tensor: &crate::kernels::ternary::TernaryTensor,
    chunk_size: usize,
) -> Vec<u8> {
    let (out_features, in_features) = tensor.shape;
    let k_words = tensor.k_words;
    let words_per_chunk = chunk_size / 32;
    let num_chunks = (k_words + words_per_chunk - 1) / words_per_chunk;
    let bitmap_words = (num_chunks + 63) / 64;
    
    let mut bitmap = vec![0u64; out_features * bitmap_words];
    
    // Build bitmap for each output feature
    for row in 0..out_features {
        for chunk_idx in 0..num_chunks {
            let word_start = row * k_words + chunk_idx * words_per_chunk;
            let word_end = std::cmp::min(word_start + words_per_chunk, (row + 1) * k_words);
            
            // Check if chunk has any non-zero bits
            let mut is_active = false;
            for word_idx in word_start..word_end {
                if tensor.plus_plane[word_idx] != 0 || tensor.minus_plane[word_idx] != 0 {
                    is_active = true;
                    break;
                }
            }
            
            if is_active {
                let bitmap_idx = row * bitmap_words + chunk_idx / 64;
                let bit_idx = chunk_idx % 64;
                bitmap[bitmap_idx] |= 1u64 << bit_idx;
            }
        }
    }
    
    // Convert to bytes
    bitmap.iter().flat_map(|&word| word.to_le_bytes()).collect()
}
```

**Testing**:
- Test bitmap creation for known sparse patterns
- Verify bitmap correctly identifies active/inactive chunks
- Test with various chunk sizes

#### 5.4: Add Comprehensive Tests (1 day)

**File**: `src/kernels/ternary/matmul_cubecl.rs`

Add tests for sparse kernel:

```rust
#[test]
fn test_sparse_config_creation() {
    let base = VectorizedTernaryMatmulConfig::rtx_5080_preset(8, 512, 32, 1024);
    
    // High sparsity: enable skipping
    let sparse = SparseOptimizedConfig::from_sparsity(base, 0.95);
    assert!(sparse.enable_plane_skipping);
    assert_eq!(sparse.chunk_size, 64);
    
    // Low sparsity: disable skipping
    let dense = SparseOptimizedConfig::from_sparsity(base, 0.50);
    assert!(!dense.enable_plane_skipping);
}

#[test]
fn test_sparse_kernel_correctness_high_sparsity() {
    // Test with 95% sparse weights (only first 5% are non-zero)
    let batch_size = 2;
    let in_features = 128;
    let out_features = 4;
    let k_words = 4;
    
    // Create 95% sparse weights: only first chunk is active
    let mut w_plus = vec![0u32; out_features * k_words];
    let mut w_minus = vec![0u32; out_features * k_words];
    
    // Only first word of each output has weights (6.25% of total)
    for out_idx in 0..out_features {
        w_plus[out_idx * k_words] = 0xFFFFFFFF;
    }
    
    // Create sparsity bitmap (chunk_size = 64 dims = 2 words)
    // Chunk 0 (words 0-1): active
    // Chunk 1 (words 2-3): inactive
    let bitmap = vec![0x1u64; out_features]; // Only bit 0 set
    
    // ... rest of test validates correctness and measures skip rate
}

#[test]
fn test_sparse_kernel_vs_dense() {
    // Compare sparse kernel output with dense kernel
    // Should produce identical results
}

#[test]
fn test_sparse_kernel_skip_rate() {
    // Verify that skip rate matches actual sparsity
    // For 95% sparse: should skip ~95% of chunks
}
```

### Success Criteria for Task 2.5

- [ ] Sparse kernel compiles and runs without errors
- [ ] Numerical equivalence with non-sparse kernels (MAE < 1e-5)
- [ ] Skip rate matches actual sparsity (within 5%)
- [ ] Overhead on dense models <5%
- [ ] Expected speedup on 95% sparse models (measure on GPU when available)
- [ ] All tests pass (expect 5+ new tests)

---

## Task 2.6: GPU Dispatch Integration

**Objective**: Wire up all kernel implementations to `ternary_matmul()` with intelligent dispatch based on device, sparsity, and configuration.

**Expected Impact**:
- Seamless CPU/GPU dispatch
- Automatic kernel selection based on sparsity
- Fallback handling for unsupported configurations

### Implementation Strategy

#### 6.1: Update Dispatch Function (3-4 hours)

**File**: `src/kernels/ternary/matmul.rs`

Enhance main dispatch function:

```rust
/// Ternary matrix multiplication with automatic kernel selection.
///
/// Automatically selects the best kernel based on:
/// - Device (CPU vs GPU)
/// - Sparsity level (sparse-optimized vs dense)
/// - Hardware capabilities (RTX 5080 vs RTX 3090 Ti)
///
/// # Arguments
/// * `input` - FP activations [batch, in_features]
/// * `weights` - Ternary weights with optional sparsity metadata
/// * `bias` - Optional bias [out_features]
///
/// # Returns
/// Output tensor [batch, out_features]
pub fn ternary_matmul(
    input: &Tensor,
    weights: &TernaryTensor,
    bias: Option<&Tensor>,
) -> Result<Tensor> {
    match input.device() {
        Device::Cpu => ternary_matmul_cpu(input, weights, bias),
        
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => {
            if has_cubecl_cuda_support() {
                ternary_matmul_cubecl_dispatch(input, weights, bias)
            } else {
                // Fallback to CPU if CubeCL not available
                ternary_matmul_cpu(input, weights, bias)
            }
        }
        
        _ => Err(Error::UnsupportedDevice(format!(
            "Ternary matmul not supported on device: {:?}",
            input.device()
        ))),
    }
}

#[cfg(feature = "cuda")]
fn ternary_matmul_cubecl_dispatch(
    input: &Tensor,
    weights: &TernaryTensor,
    bias: Option<&Tensor>,
) -> Result<Tensor> {
    use cubecl::Runtime;
    use cubecl_cuda::CudaRuntime;
    
    // Get dimensions
    let input_shape = input.dims();
    let (out_features, in_features) = weights.shape;
    let batch_size = input_shape[0];
    let k_words = weights.k_words;
    
    // Calculate sparsity
    let sparsity = weights.sparsity();
    
    // Select kernel based on sparsity
    let use_sparse_kernel = sparsity >= 0.90;
    
    // Detect GPU and select preset
    let device_id = input.device().as_cuda_device()?;
    let gpu_name = detect_gpu_name(device_id)?;
    
    // Initialize CubeCL runtime
    let client = CudaRuntime::client(device_id);
    
    // Convert tensors to CubeCL handles
    let input_handle = candle_to_cubecl_handle(input, &client)?;
    let w_plus_handle = u32_planes_to_cubecl_handle(
        &weights.plus_plane_tensor()?, &client
    )?;
    let w_minus_handle = u32_planes_to_cubecl_handle(
        &weights.minus_plane_tensor()?, &client
    )?;
    let scales_handle = candle_to_cubecl_handle(
        &weights.scale_tensor()?, &client
    )?;
    
    // Allocate output
    let output_shape = [batch_size, out_features];
    let output_handle = allocate_cubecl_tensor(&output_shape, &client)?;
    
    // Dispatch to appropriate kernel
    let output = if use_sparse_kernel {
        // Use sparse-optimized kernel
        let config = if gpu_name.contains("5080") {
            SparseOptimizedConfig::rtx_5080_sparse(
                batch_size as u32,
                out_features as u32,
                k_words as u32,
                in_features as u32,
                sparsity,
            )
        } else {
            SparseOptimizedConfig::rtx_3090ti_sparse(
                batch_size as u32,
                out_features as u32,
                k_words as u32,
                in_features as u32,
                sparsity,
            )
        };
        
        // Create sparsity bitmap
        let bitmap_bytes = create_sparsity_bitmap_for_tensor(weights, 64);
        let bitmap_handle = client.create(&bitmap_bytes);
        
        let (cube_count, cube_dim) = get_sparse_launch_config(&config);
        
        ternary_matmul_kernel_sparse::launch_unchecked::<F32, CudaRuntime>(
            &client,
            cube_count,
            cube_dim,
            input_handle,
            w_plus_handle,
            w_minus_handle,
            scales_handle,
            bitmap_handle,
            output_handle,
            config,
        );
        
        cubecl_to_candle_tensor(output_handle, &output_shape, input.device())?
    } else {
        // Use vectorized kernel (best for dense)
        let config = if gpu_name.contains("5080") {
            VectorizedTernaryMatmulConfig::rtx_5080_preset(
                batch_size as u32,
                out_features as u32,
                k_words as u32,
                in_features as u32,
            )
        } else {
            VectorizedTernaryMatmulConfig::rtx_3090ti_preset(
                batch_size as u32,
                out_features as u32,
                k_words as u32,
                in_features as u32,
            )
        };
        
        let (cube_count, cube_dim) = get_vectorized_launch_config(&config);
        
        ternary_matmul_kernel_vectorized::launch_unchecked::<F32, CudaRuntime>(
            &client,
            cube_count,
            cube_dim,
            input_handle,
            w_plus_handle,
            w_minus_handle,
            scales_handle,
            output_handle,
            config,
        );
        
        cubecl_to_candle_tensor(output_handle, &output_shape, input.device())?
    };
    
    // Add bias if provided
    if let Some(b) = bias {
        output.broadcast_add(b)
    } else {
        Ok(output)
    }
}

/// Detect GPU name from device ID
#[cfg(feature = "cuda")]
fn detect_gpu_name(device_id: usize) -> Result<String> {
    // Use CUDA device properties to detect GPU
    // For now, return a placeholder
    // TODO: Implement actual GPU detection
    Ok("RTX 5080".to_string())
}
```

**Key Features**:
- Automatic CPU/GPU dispatch
- Sparsity-based kernel selection (sparse vs vectorized)
- GPU-specific configuration presets
- Fallback handling for missing CubeCL support
- Bias addition after kernel execution

#### 6.2: Add Integration Tests (2-3 hours)

**File**: `tests/integration/ternary_matmul_gpu.rs` (new)

Create end-to-end integration tests:

```rust
use unsloth_rs::kernels::ternary::{TernaryTensor, ternary_matmul};
use candle_core::{Tensor, Device, DType};

#[test]
fn test_ternary_matmul_dispatch_cpu() {
    let device = Device::Cpu;
    
    // Create test data
    let input = Tensor::randn(0.0f32, 1.0, (4, 128), &device).unwrap();
    let weights = create_test_ternary_weights(256, 128, 0.5);
    
    // Should automatically dispatch to CPU
    let output = ternary_matmul(&input, &weights, None).unwrap();
    
    assert_eq!(output.dims(), &[4, 256]);
}

#[test]
#[cfg(feature = "cuda")]
fn test_ternary_matmul_dispatch_gpu() {
    if let Ok(device) = Device::cuda_if_available(0) {
        if !matches!(device, Device::Cuda(_)) {
            return; // Skip if no GPU
        }
        
        let input = Tensor::randn(0.0f32, 1.0, (4, 128), &device).unwrap();
        let weights = create_test_ternary_weights(256, 128, 0.5);
        
        // Should automatically dispatch to GPU
        let output = ternary_matmul(&input, &weights, None).unwrap();
        
        assert_eq!(output.dims(), &[4, 256]);
    }
}

#[test]
#[cfg(feature = "cuda")]
fn test_ternary_matmul_sparse_vs_dense() {
    if let Ok(device) = Device::cuda_if_available(0) {
        if !matches!(device, Device::Cuda(_)) {
            return;
        }
        
        let input = Tensor::randn(0.0f32, 1.0, (4, 128), &device).unwrap();
        
        // Test with dense weights (should use vectorized kernel)
        let dense_weights = create_test_ternary_weights(256, 128, 0.5);
        let dense_output = ternary_matmul(&input, &dense_weights, None).unwrap();
        
        // Test with sparse weights (should use sparse kernel)
        let sparse_weights = create_test_ternary_weights(256, 128, 0.95);
        let sparse_output = ternary_matmul(&input, &sparse_weights, None).unwrap();
        
        // Both should produce valid outputs
        assert_eq!(dense_output.dims(), &[4, 256]);
        assert_eq!(sparse_output.dims(), &[4, 256]);
    }
}

#[test]
fn test_ternary_matmul_with_bias() {
    let device = Device::Cpu;
    let input = Tensor::randn(0.0f32, 1.0, (4, 128), &device).unwrap();
    let weights = create_test_ternary_weights(256, 128, 0.5);
    let bias = Tensor::randn(0.0f32, 1.0, (256,), &device).unwrap();
    
    let output = ternary_matmul(&input, &weights, Some(&bias)).unwrap();
    
    assert_eq!(output.dims(), &[4, 256]);
}

fn create_test_ternary_weights(
    out_features: usize,
    in_features: usize,
    sparsity: f32,
) -> TernaryTensor {
    // Helper to create test weights with specified sparsity
    // ... implementation
}
```

#### 6.3: Update TernaryLinear to Use Dispatch (1 hour)

**File**: `src/kernels/ternary/linear.rs`

Ensure TernaryLinear uses the dispatch:

```rust
impl candle_nn::Module for TernaryLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Use dispatch function (automatically selects CPU/GPU)
        ternary_matmul(x, &self.weights, self.bias.as_ref())
    }
}
```

### Success Criteria for Task 2.6

- [ ] Dispatch function correctly routes to CPU/GPU
- [ ] Sparse kernel selected for 90%+ sparsity
- [ ] Vectorized kernel selected for <90% sparsity
- [ ] Fallback to CPU works when CubeCL unavailable
- [ ] All integration tests pass
- [ ] TernaryLinear transparently uses GPU when available

---

## Testing Strategy

### Unit Tests
- Config creation and validation
- Sparsity bitmap generation
- Kernel numerical correctness
- Skip rate validation

### Integration Tests
- CPU/GPU dispatch
- Sparse vs dense kernel selection
- End-to-end workflows
- TernaryLinear layer usage

### Performance Tests (When GPU Available)
- Benchmark sparse kernel on 95% sparse models
- Measure overhead on dense models
- Profile skip rates
- Validate speedup claims

---

## Implementation Timeline

**Task 2.5: Plane Skipping** (2-3 days)
- Day 1: Config + Sparse kernel implementation
- Day 2: Helper functions + Comprehensive testing
- Day 3: Performance validation and tuning

**Task 2.6: GPU Dispatch** (1 day)
- Morning: Dispatch function implementation
- Afternoon: Integration tests + TernaryLinear update
- Evening: Documentation and final validation

**Total**: 3-4 days for both tasks

---

## Risk Assessment

### Technical Risks

**Sparsity Bitmap Overhead** 🟡 MEDIUM
- **Risk**: Bitmap checking may add overhead on dense models
- **Mitigation**: Comptime flag to compile out checks, measure overhead
- **Impact**: Acceptable if overhead <5%

**Kernel Selection Logic** 🟢 LOW
- **Risk**: May select suboptimal kernel
- **Mitigation**: Clear threshold (90%), allow override via config
- **Impact**: Users can manually override if needed

**GPU Detection** 🟡 MEDIUM
- **Risk**: May not correctly detect GPU model
- **Mitigation**: Conservative defaults (RTX 3090 Ti settings), allow manual config
- **Impact**: May not use optimal settings but will work correctly

### Process Risks

**GPU Validation** 🔴 HIGH
- **Risk**: Cannot validate GPU kernels without hardware
- **Mitigation**: Comprehensive CPU simulation tests, defer GPU testing
- **Impact**: Implementation will be correct but performance unvalidated

---

## Success Metrics

### Code Quality
- [ ] All tests pass (expect 120+ tests total)
- [ ] Zero clippy warnings
- [ ] Documentation coverage maintained
- [ ] No security vulnerabilities

### Feature Completeness
- [ ] Plane skipping kernel implemented
- [ ] Dispatch logic implemented
- [ ] Integration tests passing
- [ ] TernaryLinear using dispatch

### Performance (when GPU available)
- [ ] 2-4x speedup on 95% sparse models
- [ ] <5% overhead on dense models
- [ ] Correct kernel selection based on sparsity

---

## Deliverables

Upon completion of Tasks 2.5 and 2.6:

1. **Sparse-optimized kernel** with plane skipping
2. **Intelligent dispatch function** for CPU/GPU selection
3. **Sparsity bitmap utilities** for metadata conversion
4. **Comprehensive test suite** (unit + integration)
5. **Updated TernaryLinear** using GPU dispatch
6. **Documentation** for kernel selection and usage

---

## Next Steps After Phase 2

Once Tasks 2.5 and 2.6 are complete, Phase 2 will be finished. Next steps:

1. **GPU Validation**: Test all kernels on actual hardware
2. **Phase 3**: Ternary Attention GPU implementation
3. **Phase 4**: Advanced sparsity optimizations
4. **Phase 5**: End-to-end integration and benchmarking

---

## Approval Checklist

Before proceeding with implementation, please review:

- [ ] Implementation approach for plane skipping
- [ ] Sparse kernel design and chunk checking logic
- [ ] Dispatch function structure and kernel selection
- [ ] Testing strategy and coverage
- [ ] Timeline estimate (3-4 days)
- [ ] Risk mitigation strategies

**Ready to proceed after approval** ✅
