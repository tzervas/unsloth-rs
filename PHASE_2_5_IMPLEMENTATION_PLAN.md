# Phase 2-5 Kernel Implementation Plan

**Date**: 2026-01-07  
**Status**: Ready for Implementation  
**Base Branch**: `main` (includes PR #16 ternary kernel foundation)  
**CubeCL Version**: v0.8.1

## Executive Summary

This document provides a detailed implementation plan to complete Phases 2-5 of the unsloth-rs kernel optimization roadmap. The repository currently has a solid foundation with:
- ✅ 114 passing tests
- ✅ CPU reference implementations for all kernels
- ✅ Ternary tensor infrastructure (types, quantization, CPU matmul)
- ✅ Flash Attention CubeCL kernel (Phase 1 complete, awaiting GPU validation)
- ✅ Comprehensive documentation and benchmarking infrastructure

**Critical Path**: The main blockers are:
1. GPU hardware access for validation (RTX 5080, RTX 3090 Ti)
2. CubeCL kernel implementations for ternary operations
3. Integration testing across GPU architectures

## Current State Analysis

### Implemented ✅

**Core Infrastructure:**
- Ternary tensor types (`TernaryTensor`, `TernaryPlanes`, `SparsityMetadata`)
- FP32 → ternary quantization with TWN-style calibration
- CPU popcount-based matmul reference
- TernaryLinear layer (drop-in replacement for candle_nn::Linear)
- Flash Attention CubeCL kernel with causal masking
- Candle ↔ CubeCL tensor interop
- GPU configuration presets (RTX 5080, RTX 3090 Ti)

**Testing:**
- 114 unit tests passing (100% pass rate)
- Numerical equivalence tests for all CPU implementations
- Shape validation tests
- Memory estimation tests

**Documentation:**
- Comprehensive ROADMAP.md (strategic plan)
- TASKS.md (tactical execution)
- TERNARY_GPU_IMPLEMENTATION.md (implementation guide)
- FLASH_ATTENTION_PLAN.md (Flash Attention phases)
- BENCHMARKING.md (profiling guide)

### Partially Complete 🚧

**Flash Attention GPU:**
- ✅ Kernel implementation complete
- ✅ Causal masking support
- ✅ Numerical tests passing
- ⏸️ GPU profiling blocked (requires CUDA hardware)
- ⏸️ Cross-GPU validation blocked (requires hardware)

**Ternary Matmul:**
- ✅ CPU implementation complete
- ✅ `matmul_cubecl.rs` file exists (scaffold)
- ❌ GPU kernel not implemented
- ❌ No GPU benchmarks

**Mixed Precision:**
- ✅ Configuration structures exist
- ✅ Precision conversion utilities implemented
- ⚠️ Not fully integrated with all kernels
- ⚠️ No automatic mixed precision (AMP) support

### Not Started ❌

**Phase 2: GPU Ternary Matmul**
- CubeCL popcount-based kernel
- Tiled implementation with shared memory
- Vectorized Line<u32> loads
- Plane skipping optimization
- u32 tensor interop for CubeCL

**Phase 3: Ternary Attention GPU**
- Q·K^T ternary scoring kernel
- Online softmax with popcount
- Integration with Flash Attention
- Causal masking via plane operations
- Hybrid FP/ternary dispatch

**Phase 4: Advanced Sparsity**
- Dynamic plane skipping in kernels
- Chunk-based sparsity optimization
- Runtime sparsity detection
- CSR-like storage for ultra-sparse

**Phase 5: End-to-End Integration**
- Model quantization pipeline
- Full transformer layer tests
- GPU benchmarking suite
- Performance validation
- Documentation updates

## Implementation Phases

---

## Phase 2: GPU Ternary Matmul Kernel (2-3 weeks)

**Priority**: 🔴 HIGHEST  
**Dependencies**: None (foundation complete)  
**Branch**: `feature/ternary-matmul-gpu`

### Objectives
1. Implement CubeCL popcount-based matmul kernel
2. Achieve ≥5x speedup vs FP16 matmul on sparse models
3. Validate numerical equivalence with CPU reference
4. Support arbitrary batch dimensions

### Implementation Tasks

#### Task 2.1: u32 Tensor Interop (2-3 hours)
**File**: `src/kernels/cubecl/interop.rs`

Add functions to convert u32 plane tensors to CubeCL handles:

```rust
/// Convert Candle u32 tensor to CubeCL handle for bitsliced planes
pub fn u32_planes_to_cubecl_handle<R: Runtime>(
    tensor: &Tensor,
    client: &ComputeClient<R::Server, R::Channel>,
) -> Result<TensorHandle<R>> {
    // Similar to candle_to_cubecl_handle but for u32 dtype
    // Handle device transfer (CPU → GPU)
    // Validate shape and contiguity
}

/// Convert CubeCL u32 handle back to Candle tensor
pub fn cubecl_to_u32_candle_tensor<R: Runtime>(
    handle: TensorHandle<R>,
    shape: &[usize],
    device: &Device,
) -> Result<Tensor> {
    // Reverse conversion
}
```

**Testing**:
- Round-trip conversion: u32 Tensor → CubeCL → u32 Tensor
- Verify bitwise accuracy (no data loss)
- Test with various shapes and devices

#### Task 2.2: Basic Popcount Kernel (1-2 days)
**File**: `src/kernels/ternary/matmul_cubecl.rs`

Implement basic kernel without tiling:

```rust
#[cube(launch_unchecked)]
pub fn ternary_matmul_kernel<F: Float>(
    input: &Array<F>,           // [batch, in_features] - FP activations
    w_plus: &Array<u32>,        // [out_features, k_words] - +1 bits
    w_minus: &Array<u32>,       // [out_features, k_words] - -1 bits
    scales: &Array<F>,          // [out_features] - per-output scale
    output: &mut Array<F>,      // [batch, out_features]
    #[comptime] batch: u32,
    #[comptime] in_features: u32,
    #[comptime] out_features: u32,
    #[comptime] k_words: u32,
) {
    let batch_idx = CUBE_POS_X;
    let out_idx = CUBE_POS_Y;
    
    // Quantize input row to bitsliced format
    let mut input_plus = Array::<u32>::new(k_words);
    let mut input_minus = Array::<u32>::new(k_words);
    quantize_to_planes(&input[batch_idx], &mut input_plus, &mut input_minus, k_words);
    
    // Popcount-based dot product
    let mut pos_sum = 0u32;
    let mut neg_sum = 0u32;
    
    for k in 0..k_words {
        let wp = w_plus[out_idx * k_words + k];
        let wm = w_minus[out_idx * k_words + k];
        let ip = input_plus[k];
        let im = input_minus[k];
        
        // Native GPU popcount (maps to __popc on CUDA)
        pos_sum += (wp & ip).count_ones() + (wm & im).count_ones();
        neg_sum += (wp & im).count_ones() + (wm & ip).count_ones();
    }
    
    let dot = F::cast_from((pos_sum as i32) - (neg_sum as i32));
    output[batch_idx * out_features + out_idx] = dot * scales[out_idx];
}

#[cube]
fn quantize_to_planes<F: Float>(
    input: &[F],
    plus: &mut Array<u32>,
    minus: &mut Array<u32>,
    k_words: u32,
) {
    // Quantize FP values to +1/-1/0 and pack into u32 words
    for word_idx in 0..k_words {
        let mut plus_word = 0u32;
        let mut minus_word = 0u32;
        
        for bit_idx in 0..32 {
            let feature_idx = word_idx * 32 + bit_idx;
            if feature_idx < input.len() {
                let val = input[feature_idx];
                if val > F::new(0.5) {
                    plus_word |= 1u32 << bit_idx;
                } else if val < F::new(-0.5) {
                    minus_word |= 1u32 << bit_idx;
                }
            }
        }
        
        plus[word_idx] = plus_word;
        minus[word_idx] = minus_word;
    }
}
```

**Testing**:
- Compare against CPU reference `ternary_matmul_cpu()`
- Test various matrix sizes (small to large)
- Verify numerical accuracy (MAE < 1e-5)

#### Task 2.3: Tiled Kernel with Shared Memory (2-3 days)
**File**: `src/kernels/ternary/matmul_cubecl.rs`

Optimize with tiling and shared memory:

```rust
#[cube(launch_unchecked)]
pub fn ternary_matmul_tiled<F: Float>(
    input: &Array<F>,
    w_plus: &Array<u32>,
    w_minus: &Array<u32>,
    scales: &Array<F>,
    output: &mut Array<F>,
    #[comptime] config: TernaryMatmulConfig,
) {
    // Shared memory for input tile
    let mut input_tile = SharedMemory::<u32>::new(config.tile_k);
    
    let batch_idx = CUBE_POS_X;
    let out_block_idx = CUBE_POS_Y;
    let thread_idx = UNIT_POS_X;
    
    // Each thread computes tile_n / block_size outputs
    for out_local in 0..config.outputs_per_thread {
        let out_idx = out_block_idx * config.tile_n + thread_idx + out_local * config.block_size;
        
        if out_idx >= config.out_features {
            continue;
        }
        
        let mut pos_sum = 0u32;
        let mut neg_sum = 0u32;
        
        // Load input tile cooperatively
        for k_tile in 0..config.num_k_tiles {
            let k_start = k_tile * config.tile_k;
            
            // Cooperative load to shared memory
            if thread_idx < config.tile_k {
                input_tile[thread_idx] = load_quantized_word(
                    &input[batch_idx],
                    k_start + thread_idx,
                    config.in_features,
                );
            }
            sync_units(); // Barrier
            
            // Compute partial dot product
            for k_local in 0..config.tile_k {
                let k_word = k_start + k_local;
                if k_word >= config.k_words {
                    break;
                }
                
                let wp = w_plus[out_idx * config.k_words + k_word];
                let wm = w_minus[out_idx * config.k_words + k_word];
                let ip = input_tile[k_local];
                
                // Plane skipping: Skip if weight is all zeros
                if (wp | wm) == 0 {
                    continue;
                }
                
                // Extract minus plane (assume stored separately or in high bits)
                let im = input_tile[k_local + config.tile_k]; // Adjust based on layout
                
                pos_sum += (wp & ip).count_ones() + (wm & im).count_ones();
                neg_sum += (wp & im).count_ones() + (wm & ip).count_ones();
            }
            sync_units();
        }
        
        let dot = F::cast_from((pos_sum as i32) - (neg_sum as i32));
        output[batch_idx * config.out_features + out_idx] = dot * scales[out_idx];
    }
}
```

**Configuration**:
```rust
pub struct TernaryMatmulConfig {
    pub tile_k: usize,      // K-dimension tile size (128-256)
    pub tile_n: usize,      // N-dimension tile size
    pub block_size: usize,  // Threads per block (256)
    pub in_features: usize,
    pub out_features: usize,
    pub k_words: usize,     // in_features / 32
    pub outputs_per_thread: usize,
    pub num_k_tiles: usize,
    pub enable_plane_skipping: bool,
}
```

**Testing**:
- Benchmark vs basic kernel (expect 1.5-2x improvement)
- Test with various tile sizes
- Verify correctness across configurations

#### Task 2.4: Vectorized Line<u32> Loads (1-2 days)
**File**: `src/kernels/ternary/matmul_cubecl.rs`

Use CubeCL's `Line<u32>` for vectorized 4-element loads:

```rust
#[cube(launch_unchecked)]
pub fn ternary_matmul_vectorized<F: Float>(
    // ... same parameters
) {
    // Use Array<Line<u32>> for 4x throughput
    let w_plus_vec: &Array<Line<u32>> = /* cast or load */;
    
    for k_vec in 0..config.k_words / 4 {
        let wp_line = w_plus_vec[out_idx * (config.k_words / 4) + k_vec];
        
        // Process 4 u32 words in one instruction
        for i in 0..4 {
            let wp = wp_line[i];
            // ... popcount operations
        }
    }
}
```

**Testing**:
- Verify bitwise correctness
- Benchmark memory bandwidth improvement
- Compare occupancy vs scalar version

#### Task 2.5: Plane Skipping with SparsityMetadata (2-3 days)
**File**: `src/kernels/ternary/matmul_cubecl.rs`

Integrate sparsity metadata for dynamic skipping:

```rust
#[cube(launch_unchecked)]
pub fn ternary_matmul_sparse<F: Float>(
    input: &Array<F>,
    w_plus: &Array<u32>,
    w_minus: &Array<u32>,
    scales: &Array<F>,
    sparsity_bitmap: &Array<u64>, // Chunk activity bitmap
    output: &mut Array<F>,
    #[comptime] config: TernaryMatmulConfig,
) {
    // ... tiled computation
    
    for chunk_idx in 0..config.num_chunks {
        // Check if chunk is active (has non-zero weights)
        let chunk_word = chunk_idx / 64;
        let chunk_bit = chunk_idx % 64;
        let is_active = (sparsity_bitmap[chunk_word] & (1u64 << chunk_bit)) != 0;
        
        if !is_active {
            continue; // Skip entire chunk (32-64 dimensions)
        }
        
        // Process active chunk
        for k_word in chunk_idx * CHUNK_SIZE..(chunk_idx + 1) * CHUNK_SIZE {
            // ... popcount operations
        }
    }
}
```

**Testing**:
- Generate tensors with varying sparsity (50%, 90%, 95%, 99%)
- Verify speedup correlates with sparsity
- Ensure correctness with sparse patterns

#### Task 2.6: GPU Dispatch Integration (1 day)
**File**: `src/kernels/ternary/matmul.rs`

Wire up GPU kernel dispatch:

```rust
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
                ternary_matmul_cubecl(input, weights, bias)
            } else {
                ternary_matmul_cpu(input, weights, bias)
            }
        }
        _ => Err(Error::UnsupportedDevice),
    }
}

#[cfg(feature = "cuda")]
fn ternary_matmul_cubecl(
    input: &Tensor,
    weights: &TernaryTensor,
    bias: Option<&Tensor>,
) -> Result<Tensor> {
    use cubecl::Runtime;
    use cubecl_cuda::CudaRuntime;
    
    // Initialize CubeCL runtime
    let device_id = input.device().as_cuda_device()?;
    let client = CudaRuntime::client(device_id);
    
    // Convert tensors to CubeCL handles
    let input_handle = candle_to_cubecl_handle(input, &client)?;
    let w_plus_handle = u32_planes_to_cubecl_handle(&weights.plus_plane_tensor()?, &client)?;
    let w_minus_handle = u32_planes_to_cubecl_handle(&weights.minus_plane_tensor()?, &client)?;
    let scales_handle = candle_to_cubecl_handle(&weights.scale_tensor()?, &client)?;
    
    // Allocate output
    let output_shape = compute_output_shape(input, weights)?;
    let output_handle = allocate_cubecl_tensor(&output_shape, &client)?;
    
    // Configure kernel launch
    let config = TernaryMatmulConfig::from_shapes(
        input.dims(),
        weights.shape(),
        weights.sparsity_metadata(),
    )?;
    
    let cube_count = CubeCount::Static(
        config.batch_size as u32,
        (config.out_features / config.tile_n) as u32,
        1,
    );
    let cube_dim = CubeDim::new(config.block_size as u32, 1, 1);
    
    // Launch kernel
    if config.sparsity > 0.90 && config.enable_plane_skipping {
        ternary_matmul_sparse::launch_unchecked::<F32, CudaRuntime>(
            &client,
            cube_count,
            cube_dim,
            input_handle,
            w_plus_handle,
            w_minus_handle,
            scales_handle,
            sparsity_bitmap_handle,
            output_handle,
            config,
        );
    } else {
        ternary_matmul_tiled::launch_unchecked::<F32, CudaRuntime>(
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
    }
    
    // Convert result back to Candle
    let output = cubecl_to_candle_tensor(output_handle, &output_shape, input.device())?;
    
    // Add bias if provided
    if let Some(b) = bias {
        output.broadcast_add(b)
    } else {
        Ok(output)
    }
}
```

**Testing**:
- End-to-end test: Create TernaryLinear layer, run forward pass on GPU
- Compare against CPU reference
- Test with and without bias
- Test various batch sizes and dimensions

### Success Criteria

- [ ] All kernel implementations compile without errors
- [ ] GPU kernels produce identical results to CPU (MAE < 1e-5)
- [ ] Achieves ≥5x speedup vs FP16 matmul on 95% sparse models
- [ ] Memory usage reduced by ≥10x vs FP16 weights
- [ ] Tests pass on both RTX 5080 and RTX 3090 Ti (when hardware available)
- [ ] Benchmarks documented in BENCHMARKS.md

---

## Phase 3: Ternary Attention GPU Integration (2-3 weeks)

**Priority**: 🟠 HIGH  
**Dependencies**: Phase 2 (ternary matmul kernel)  
**Branch**: `feature/ternary-attention-gpu`

### Objectives
1. Integrate ternary matmul with Flash Attention for Q·K^T scoring
2. Implement online softmax with popcount-based scores
3. Add hybrid FP/ternary dispatch based on sparsity
4. Support causal masking via plane operations

### Implementation Tasks

#### Task 3.1: Q·K^T Ternary Scoring Kernel (3-4 days)
**File**: `src/kernels/ternary/attention.rs`

Extend attention to use ternary Q·K^T:

```rust
/// Compute attention scores using ternary Q·K^T
pub fn ternary_attention_scores_cubecl(
    q: &TernaryTensor,     // [batch, heads, seq_q, head_dim]
    k: &TernaryTensor,     // [batch, kv_heads, seq_k, head_dim]
    scale: f32,            // 1/sqrt(head_dim)
    causal_mask: bool,
) -> Result<Tensor> {
    // Launch kernel to compute popcount-based scores
    // Output: [batch, heads, seq_q, seq_k]
}
```

Kernel implementation:

```rust
#[cube(launch_unchecked)]
pub fn ternary_attention_score_kernel<F: Float>(
    q_plus: &Array<u32>,    // [batch, heads, seq_q, head_words]
    q_minus: &Array<u32>,
    q_scales: &Array<F>,    // [batch, heads, seq_q]
    k_plus: &Array<u32>,    // [batch, kv_heads, seq_k, head_words]
    k_minus: &Array<u32>,
    k_scales: &Array<F>,    // [batch, kv_heads, seq_k]
    scores: &mut Array<F>,  // [batch, heads, seq_q, seq_k]
    #[comptime] config: TernaryAttentionConfig,
) {
    let batch_idx = CUBE_POS_X;
    let head_idx = CUBE_POS_Y;
    let q_idx = UNIT_POS_X;
    
    // Handle GQA: Map head to kv_head
    let kv_head = head_idx / config.num_heads_per_kv;
    
    // Load Q row into shared memory
    let mut q_p_tile = SharedMemory::<u32>::new(config.head_words);
    let mut q_m_tile = SharedMemory::<u32>::new(config.head_words);
    
    // Cooperative load
    if UNIT_POS_X < config.head_words {
        let q_offset = ((batch_idx * config.num_heads + head_idx) * config.seq_q + q_idx) * config.head_words;
        q_p_tile[UNIT_POS_X] = q_plus[q_offset + UNIT_POS_X];
        q_m_tile[UNIT_POS_X] = q_minus[q_offset + UNIT_POS_X];
    }
    sync_units();
    
    // Compute scores for all K positions
    for k_idx in 0..config.seq_k {
        // Causal masking
        if config.causal && k_idx > q_idx {
            scores[batch_idx * config.num_heads * config.seq_q * config.seq_k 
                   + head_idx * config.seq_q * config.seq_k 
                   + q_idx * config.seq_k 
                   + k_idx] = F::new(-1e10);
            continue;
        }
        
        // Popcount-based dot product
        let mut pos_sum = 0u32;
        let mut neg_sum = 0u32;
        
        let k_offset = ((batch_idx * config.num_kv_heads + kv_head) * config.seq_k + k_idx) * config.head_words;
        
        for w in 0..config.head_words {
            let qp = q_p_tile[w];
            let qm = q_m_tile[w];
            let kp = k_plus[k_offset + w];
            let km = k_minus[k_offset + w];
            
            pos_sum += (qp & kp).count_ones() + (qm & km).count_ones();
            neg_sum += (qp & km).count_ones() + (qm & kp).count_ones();
        }
        
        let dot = F::cast_from((pos_sum as i32) - (neg_sum as i32));
        let q_scale = q_scales[batch_idx * config.num_heads * config.seq_q + head_idx * config.seq_q + q_idx];
        let k_scale = k_scales[batch_idx * config.num_kv_heads * config.seq_k + kv_head * config.seq_k + k_idx];
        
        let score = dot * q_scale * k_scale * F::new(config.scale);
        
        scores[batch_idx * config.num_heads * config.seq_q * config.seq_k 
               + head_idx * config.seq_q * config.seq_k 
               + q_idx * config.seq_k 
               + k_idx] = score;
    }
}
```

**Testing**:
- Compare scores against FP32 attention (expect ~1-3% difference)
- Verify GQA head mapping
- Test causal masking correctness

#### Task 3.2: Online Softmax with Popcount (2-3 days)
**File**: `src/kernels/ternary/attention.rs`

Implement memory-efficient online softmax:

```rust
#[cube(launch_unchecked)]
pub fn ternary_attention_softmax_kernel<F: Float>(
    scores: &Array<F>,      // [batch, heads, seq_q, seq_k]
    probs: &mut Array<F>,   // [batch, heads, seq_q, seq_k]
    #[comptime] config: TernaryAttentionConfig,
) {
    let batch_idx = CUBE_POS_X;
    let head_idx = CUBE_POS_Y;
    let q_idx = UNIT_POS_X;
    
    // Online max computation
    let mut max_score = F::new(-1e10);
    for k_idx in 0..config.seq_k {
        let idx = batch_idx * config.num_heads * config.seq_q * config.seq_k 
                  + head_idx * config.seq_q * config.seq_k 
                  + q_idx * config.seq_k 
                  + k_idx;
        let score = scores[idx];
        if score > max_score {
            max_score = score;
        }
    }
    
    // Warp-level reduction for max (optional optimization)
    max_score = warp_reduce(max_score, |a, b| if a > b { a } else { b });
    
    // Compute exp and sum
    let mut sum = F::new(0.0);
    for k_idx in 0..config.seq_k {
        let idx = batch_idx * config.num_heads * config.seq_q * config.seq_k 
                  + head_idx * config.seq_q * config.seq_k 
                  + q_idx * config.seq_k 
                  + k_idx;
        let score = scores[idx];
        let exp_val = (score - max_score).exp();
        probs[idx] = exp_val;
        sum += exp_val;
    }
    
    // Warp-level reduction for sum
    sum = warp_reduce(sum, |a, b| a + b);
    
    // Normalize
    for k_idx in 0..config.seq_k {
        let idx = batch_idx * config.num_heads * config.seq_q * config.seq_k 
                  + head_idx * config.seq_q * config.seq_k 
                  + q_idx * config.seq_k 
                  + k_idx;
        probs[idx] = probs[idx] / sum;
    }
}
```

**Testing**:
- Verify softmax properties (sum to 1, non-negative)
- Compare against standard softmax
- Test numerical stability with large/small values

#### Task 3.3: Hybrid FP/Ternary Dispatch (1-2 days)
**File**: `src/kernels/attention.rs`

Add intelligent dispatch based on sparsity:

```rust
impl FusedAttention {
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        kv_cache: Option<(&Tensor, &Tensor)>,
    ) -> Result<Tensor> {
        // Project to Q, K, V
        let (q, k, v) = self.qkv_projection(hidden_states)?;
        
        // Decide whether to use ternary path
        let use_ternary = if self.config.enable_ternary {
            // Quantize Q, K to ternary
            let (q_ternary, q_scale) = quantize_tensor(&q, &self.ternary_config)?;
            let (k_ternary, k_scale) = quantize_tensor(&k, &self.ternary_config)?;
            
            // Check if sparsity is high enough
            let q_sparsity = q_ternary.sparsity();
            let k_sparsity = k_ternary.sparsity();
            
            should_use_ternary_attention(q_sparsity, k_sparsity, &self.ternary_config)
        } else {
            false
        };
        
        let attn_output = if use_ternary {
            #[cfg(feature = "cuda")]
            {
                if hidden_states.device().is_cuda() && has_cubecl_cuda_support() {
                    // GPU ternary path
                    ternary_attention_cubecl(&q, &k, &v, attention_mask)?
                } else {
                    // CPU ternary path
                    ternary_attention_cpu(&q, &k, &v, attention_mask)?
                }
            }
            #[cfg(not(feature = "cuda"))]
            ternary_attention_cpu(&q, &k, &v, attention_mask)?
        } else {
            // Standard FP path (Flash Attention or Candle backend)
            if hidden_states.device().is_cuda() {
                flash_attention_cubecl(&q, &k, &v, attention_mask)?
            } else {
                standard_attention(&q, &k, &v, attention_mask)?
            }
        };
        
        // Output projection
        self.output_projection(&attn_output)
    }
}
```

**Testing**:
- Test dispatch logic with varying sparsity levels
- Verify performance improvement on sparse vs dense inputs
- Ensure correctness regardless of path taken

#### Task 3.4: Causal Masking via Plane Operations (1 day)
**File**: `src/kernels/ternary/attention.rs`

Optimize causal masking:

```rust
/// Apply causal mask by zeroing out planes for future positions
pub fn apply_causal_mask_to_planes(
    plus_plane: &mut Tensor,   // [batch, heads, seq, head_words]
    minus_plane: &mut Tensor,
    seq_len: usize,
) -> Result<()> {
    // For each query position q, zero out all K positions > q
    for q_idx in 0..seq_len {
        for k_idx in (q_idx + 1)..seq_len {
            // Zero both planes for masked positions
            // This is more efficient than masking scores
        }
    }
    Ok(())
}
```

**Testing**:
- Verify causal property (no future information leakage)
- Compare against score masking
- Benchmark performance improvement

#### Task 3.5: Integration Tests (2-3 days)
**Files**: `tests/integration/ternary_attention.rs` (new)

Create comprehensive integration tests:

```rust
#[test]
fn test_ternary_attention_end_to_end() {
    // Create FusedAttention with ternary enabled
    let config = FusedAttentionConfig {
        enable_ternary: true,
        ternary_sparsity_threshold: 0.90,
        // ... other config
    };
    
    let attention = FusedAttention::new(config, &device)?;
    
    // Random input
    let hidden_states = Tensor::randn(0.0f32, 1.0, (1, 128, 768), &device)?;
    
    // Forward pass (should use ternary if input becomes sparse after quantization)
    let output = attention.forward(&hidden_states, None, None)?;
    
    // Verify shape
    assert_eq!(output.dims(), &[1, 128, 768]);
    
    // Compare against FP reference (allow small error)
    let fp_output = attention_fp_reference(&hidden_states)?;
    let mae = mean_absolute_error(&output, &fp_output)?;
    assert!(mae < 0.05); // 5% tolerance for ternary
}

#[test]
fn test_ternary_attention_causal() {
    // Test with causal masking
}

#[test]
fn test_ternary_attention_gqa() {
    // Test with grouped-query attention
}

#[test]
#[cfg(feature = "cuda")]
fn test_ternary_attention_gpu() {
    // GPU-specific tests
}
```

### Success Criteria

- [ ] Ternary attention produces valid outputs (no NaN/Inf)
- [ ] Results within 5% MAE of FP32 attention for sparse inputs
- [ ] Achieves ≥4x speedup vs Flash Attention on 95% sparse Q/K
- [ ] Causal masking works correctly
- [ ] GQA support validated
- [ ] Hybrid dispatch selects optimal path based on sparsity

---

## Phase 4: Advanced Sparsity Optimization (1-2 weeks)

**Priority**: 🟡 MEDIUM  
**Dependencies**: Phase 2, Phase 3  
**Branch**: `feature/advanced-sparsity`

### Objectives
1. Implement dynamic plane skipping in all kernels
2. Add chunk-based sparsity optimization
3. Create runtime sparsity detection and profiling
4. Optimize for 95-99% sparsity patterns

### Implementation Tasks

#### Task 4.1: Dynamic Plane Skipping (3-4 days)
**Files**: `src/kernels/ternary/matmul_cubecl.rs`, `src/kernels/ternary/attention.rs`

Already partially implemented in Task 2.5, enhance with:

1. **Compile-time optimization**:
```rust
if #[comptime] config.enable_plane_skipping {
    // Skipping logic compiled in
} else {
    // Dense computation path
}
```

2. **Runtime profiling**:
```rust
pub struct SparsityStats {
    pub planes_skipped: usize,
    pub planes_computed: usize,
    pub chunks_skipped: usize,
    pub chunks_computed: usize,
    pub actual_flops: usize,
    pub theoretical_flops: usize,
}

impl SparsityStats {
    pub fn efficiency(&self) -> f64 {
        self.planes_skipped as f64 / (self.planes_skipped + self.planes_computed) as f64
    }
}
```

3. **Adaptive threshold**:
```rust
pub fn should_enable_plane_skipping(sparsity: f64, overhead: f64) -> bool {
    // Enable if sparsity > threshold + overhead
    sparsity > (0.85 + overhead * 0.1)
}
```

**Testing**:
- Profile skip rates across sparsity levels (50%, 75%, 90%, 95%, 99%)
- Verify no correctness impact
- Measure overhead for dense inputs

#### Task 4.2: Chunk-Based Optimization (2-3 days)
**File**: `src/kernels/ternary/types.rs`

Enhance SparsityMetadata:

```rust
impl SparsityMetadata {
    /// Rebuild metadata after weight updates
    pub fn rebuild(&mut self, plus_plane: &[u32], minus_plane: &[u32]) {
        // Recompute chunk activity bitmap
        for chunk_idx in 0..self.num_chunks {
            let chunk_start = chunk_idx * CHUNK_SIZE;
            let chunk_end = (chunk_idx + 1) * CHUNK_SIZE;
            
            let mut is_active = false;
            for word_idx in chunk_start..chunk_end {
                if word_idx < plus_plane.len() {
                    if plus_plane[word_idx] != 0 || minus_plane[word_idx] != 0 {
                        is_active = true;
                        break;
                    }
                }
            }
            
            self.set_chunk_active(chunk_idx, is_active);
        }
    }
    
    /// Get optimal chunk size based on architecture
    pub fn optimal_chunk_size(device: &Device) -> usize {
        match device {
            Device::Cuda(_) => 64,  // 64 u32 words = 2048 dims
            Device::Metal(_) => 32,
            _ => 32,
        }
    }
}
```

**Testing**:
- Test rebuild correctness
- Verify chunk size selection
- Benchmark different chunk sizes

#### Task 4.3: Sparsity Profiler (2-3 days)
**File**: `src/kernels/ternary/profiler.rs` (new)

Create profiling utilities:

```rust
pub struct TernaryProfiler {
    pub kernel_stats: HashMap<String, SparsityStats>,
    pub total_time_ms: f64,
    pub total_memory_bytes: usize,
}

impl TernaryProfiler {
    pub fn profile_kernel<F>(
        &mut self,
        name: &str,
        kernel_fn: F,
    ) -> Result<()>
    where
        F: FnOnce() -> Result<SparsityStats>,
    {
        let start = std::time::Instant::now();
        let stats = kernel_fn()?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        
        self.kernel_stats.insert(name.to_string(), stats);
        self.total_time_ms += elapsed;
        
        Ok(())
    }
    
    pub fn report(&self) -> String {
        let mut report = String::from("=== Ternary Kernel Profile ===\n");
        
        for (name, stats) in &self.kernel_stats {
            report.push_str(&format!(
                "{}: {:.2}% planes skipped, {:.2}% FLOP reduction\n",
                name,
                stats.efficiency() * 100.0,
                (1.0 - stats.actual_flops as f64 / stats.theoretical_flops as f64) * 100.0
            ));
        }
        
        report.push_str(&format!("\nTotal time: {:.2}ms\n", self.total_time_ms));
        report
    }
}
```

**Testing**:
- Profile all kernels with various inputs
- Generate profiling reports
- Verify overhead is minimal (<5%)

### Success Criteria

- [ ] Plane skipping works correctly across all kernels
- [ ] Achieves ≥2x speedup on 99% sparse inputs vs 95% sparse
- [ ] Overhead on dense inputs <10%
- [ ] Profiler provides actionable insights
- [ ] Chunk-based optimization validated on target GPUs

---

## Phase 5: End-to-End Integration & Validation (2-3 weeks)

**Priority**: 🟢 MEDIUM-LOW  
**Dependencies**: Phase 2, Phase 3, Phase 4  
**Branch**: `feature/ternary-e2e`

### Objectives
1. Create full model quantization pipeline
2. Add comprehensive benchmarking suite
3. Validate on real transformer models
4. Document performance characteristics

### Implementation Tasks

#### Task 5.1: Model Quantization Pipeline (3-4 days)
**File**: `src/kernels/ternary/model.rs` (enhance existing)

Already has foundation, add:

```rust
/// Quantize entire transformer model to ternary
pub fn quantize_transformer_model(
    model: &impl TransformerModel,
    config: &ModelQuantizationConfig,
) -> Result<TernaryModel> {
    let mut quantized_layers = Vec::new();
    let mut stats = QuantizationStats::default();
    
    // Walk model layers
    for layer in model.layers() {
        match layer {
            Layer::Linear(linear) => {
                if should_quantize_layer(linear, config) {
                    let ternary = quantize_linear_layer(linear, config)?;
                    stats.update(&ternary);
                    quantized_layers.push(QuantizedLayer::Ternary(ternary));
                } else {
                    quantized_layers.push(QuantizedLayer::FP(linear.clone()));
                }
            }
            Layer::Attention(attn) => {
                // Quantize Q, K, V projections
                let q_ternary = quantize_linear_layer(&attn.q_proj, config)?;
                let k_ternary = quantize_linear_layer(&attn.k_proj, config)?;
                let v_ternary = quantize_linear_layer(&attn.v_proj, config)?;
                let o_ternary = quantize_linear_layer(&attn.o_proj, config)?;
                
                quantized_layers.push(QuantizedLayer::TernaryAttention {
                    q: q_ternary,
                    k: k_ternary,
                    v: v_ternary,
                    o: o_ternary,
                });
            }
            _ => {
                // Keep other layers in FP (LayerNorm, etc.)
                quantized_layers.push(QuantizedLayer::FP(layer.clone()));
            }
        }
    }
    
    Ok(TernaryModel {
        layers: quantized_layers,
        stats,
        config: config.clone(),
    })
}

/// Fine-tune quantized model to recover accuracy
pub fn calibrate_quantized_model(
    model: &mut TernaryModel,
    calibration_data: &Dataset,
    config: &CalibrationConfig,
) -> Result<CalibrationStats> {
    // Short fine-tuning pass to adjust scales
    // Similar to QAT (Quantization-Aware Training)
    todo!()
}
```

**Testing**:
- Quantize small test models (1-2 layers)
- Verify weight reconstruction
- Test skip patterns
- Validate memory reduction

#### Task 5.2: Comprehensive Benchmarking (4-5 days)
**File**: `benches/ternary_kernels.rs` (new)

Create benchmark suite:

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

fn bench_ternary_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("ternary_matmul");
    
    for sparsity in &[0.0, 0.50, 0.75, 0.90, 0.95, 0.99] {
        for dim in &[768, 2048, 4096] {
            let id = BenchmarkId::new(
                format!("sparsity{:.0}", sparsity * 100.0),
                format!("dim{}", dim)
            );
            
            group.bench_with_input(id, &(sparsity, dim), |b, &(s, d)| {
                let input = create_random_tensor((16, d));
                let weights = create_sparse_ternary_weights(d, d, s);
                
                b.iter(|| {
                    black_box(ternary_matmul(&input, &weights, None).unwrap())
                });
            });
        }
    }
    
    group.finish();
}

fn bench_ternary_vs_fp16(c: &mut Criterion) {
    let mut group = c.benchmark_group("ternary_vs_fp16");
    
    // Compare ternary matmul vs FP16 matmul
    for seq_len in &[128, 512, 1024, 2048] {
        let input = create_random_tensor((8, *seq_len, 768));
        let weights = create_sparse_ternary_weights(768, 768, 0.95);
        let fp16_weights = weights.dequantize()?;
        
        group.bench_with_input(
            BenchmarkId::new("ternary", seq_len),
            seq_len,
            |b, _| {
                b.iter(|| black_box(ternary_matmul(&input, &weights, None).unwrap()))
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("fp16", seq_len),
            seq_len,
            |b, _| {
                b.iter(|| black_box(fp16_matmul(&input, &fp16_weights, None).unwrap()))
            },
        );
    }
    
    group.finish();
}

fn bench_ternary_attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("ternary_attention");
    
    for seq_len in &[128, 512, 1024, 2048] {
        group.bench_with_input(
            BenchmarkId::from_parameter(seq_len),
            seq_len,
            |b, &sl| {
                let hidden = create_random_tensor((1, sl, 768));
                let attn = create_ternary_attention(768, 12);
                
                b.iter(|| {
                    black_box(attn.forward(&hidden, None, None).unwrap())
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_ternary_matmul,
    bench_ternary_vs_fp16,
    bench_ternary_attention,
);
criterion_main!(benches);
```

**Profiling Commands**:
```bash
# Run benchmarks
cargo bench --features cuda -- ternary

# Profile with CubeCL
CUBECL_PROFILE=1 cargo bench --features cuda -- ternary

# VRAM profiling
CUDA_VISIBLE_DEVICES=0 cargo bench --features cuda -- ternary

# Compare results
cargo bench --baseline fp16 -- ternary
```

**Testing**:
- Run on CPU and GPU
- Compare across sparsity levels
- Generate performance reports

#### Task 5.3: Integration Tests (3-4 days)
**Files**: `tests/integration/` (new directory)

Create end-to-end tests:

```rust
// tests/integration/model_quantization.rs
#[test]
fn test_quantize_small_transformer() {
    let model = create_test_transformer(2, 768, 12);
    let config = ModelQuantizationConfig::default();
    
    let quantized = quantize_transformer_model(&model, &config).unwrap();
    
    // Verify compression
    assert!(quantized.stats.compression_ratio() > 10.0);
    
    // Verify forward pass works
    let input = Tensor::randn(0.0f32, 1.0, (1, 128, 768), &Device::Cpu).unwrap();
    let output = quantized.forward(&input).unwrap();
    
    assert_eq!(output.dims(), &[1, 128, 768]);
}

#[test]
fn test_accuracy_degradation() {
    // Test that accuracy degradation is acceptable
    let model = load_pretrained_model("test-model");
    let quantized = quantize_transformer_model(&model, &config).unwrap();
    
    let test_data = load_test_dataset("wikitext-2");
    
    let fp32_ppl = evaluate_perplexity(&model, &test_data).unwrap();
    let ternary_ppl = evaluate_perplexity(&quantized, &test_data).unwrap();
    
    let degradation = (ternary_ppl - fp32_ppl) / fp32_ppl;
    assert!(degradation < 0.02); // <2% degradation
}

// tests/integration/gpu_validation.rs
#[test]
#[cfg(feature = "cuda")]
fn test_gpu_correctness() {
    let device = Device::new_cuda(0).unwrap();
    
    // Create model on GPU
    let model = create_ternary_model(&device);
    let input = create_random_input(&device);
    
    // Forward pass on GPU
    let gpu_output = model.forward(&input).unwrap();
    
    // Compare with CPU
    let cpu_model = model.to_device(&Device::Cpu).unwrap();
    let cpu_input = input.to_device(&Device::Cpu).unwrap();
    let cpu_output = cpu_model.forward(&cpu_input).unwrap();
    
    let mae = mean_absolute_error(
        &gpu_output.to_device(&Device::Cpu).unwrap(),
        &cpu_output
    ).unwrap();
    
    assert!(mae < 1e-5);
}
```

**Testing**:
- Run all integration tests on CPU and GPU
- Verify end-to-end correctness
- Test memory usage
- Validate performance claims

#### Task 5.4: Documentation Updates (2-3 days)
**Files**: Multiple documentation files

Update documentation:

1. **BENCHMARKS.md**: Add ternary kernel results
2. **README.md**: Add usage examples for ternary operations
3. **ROADMAP.md**: Mark completed phases
4. **Create TERNARY_USAGE_GUIDE.md**:

```markdown
# Ternary Operations Usage Guide

## Quick Start

```rust
use unsloth_rs::kernels::ternary::{TernaryLinear, quantize_tensor};

// Quantize weights
let (ternary_weights, scale) = quantize_tensor(&fp_weights, &config)?;

// Create ternary linear layer
let layer = TernaryLinear::new(ternary_weights, scale, Some(bias))?;

// Forward pass (GPU accelerated if CUDA available)
let output = layer.forward(&activations)?;
```

## Model Quantization

[... detailed guide ...]

## Performance Tuning

[... tuning guide ...]

## Troubleshooting

[... common issues ...]
```

**Testing**:
- Verify all code examples compile
- Check documentation coverage
- Review for clarity

### Success Criteria

- [ ] Full model quantization pipeline works end-to-end
- [ ] Achieves ≥10x weight memory reduction
- [ ] Achieves ≥5x speedup on 95% sparse models
- [ ] Accuracy degradation <2% on WikiText-2
- [ ] All benchmarks pass and documented
- [ ] Integration tests pass on CPU and GPU
- [ ] Documentation complete and accurate

---

## Hardware Validation Plan

**Critical**: Phases 2-5 require GPU hardware for validation. Current setup blocks at:
- Flash Attention GPU profiling (Phase 1 kernel complete, needs RTX 5080)
- Ternary kernel GPU execution (needs CUDA device)

### Hardware Requirements

| Phase | GPU | Purpose |
|-------|-----|---------|
| Phase 2 | RTX 5080 or RTX 3090 Ti | Ternary matmul development & profiling |
| Phase 3 | RTX 5080 | Ternary attention integration |
| Phase 4 | RTX 5080 | Sparsity optimization tuning |
| Phase 5 | RTX 5080 + RTX 3090 Ti | Cross-GPU validation |

### Validation Checklist

When GPU hardware becomes available:

- [ ] Validate Flash Attention kernel (Phase 1)
  - [ ] Run on RTX 5080
  - [ ] Measure 2-5x speedup vs Candle
  - [ ] Validate 70-80% VRAM reduction
  - [ ] Cross-validate on RTX 3090 Ti
  
- [ ] Validate Ternary Matmul (Phase 2)
  - [ ] Run GPU kernels
  - [ ] Measure 5x speedup vs FP16
  - [ ] Verify 10x memory reduction
  - [ ] Profile with nsight
  
- [ ] Validate Ternary Attention (Phase 3)
  - [ ] Run integrated kernels
  - [ ] Measure 4x speedup vs Flash Attention
  - [ ] Verify hybrid dispatch
  
- [ ] Validate Sparsity (Phase 4)
  - [ ] Profile skip rates
  - [ ] Measure efficiency gains
  
- [ ] Validate End-to-End (Phase 5)
  - [ ] Run full model quantization
  - [ ] Measure real-world performance
  - [ ] Validate accuracy on benchmarks

---

## Branch Strategy

All development follows the established pattern:

```
main (base)
├── feature/ternary-matmul-gpu (Phase 2)
├── feature/ternary-attention-gpu (Phase 3)
├── feature/advanced-sparsity (Phase 4)
└── feature/ternary-e2e (Phase 5)
```

### Merge Process

1. Create feature branch from `main`
2. Implement tasks with tests
3. Run `cargo test`, `cargo clippy`, `cargo fmt`
4. Create PR to `main`
5. Code review
6. Merge after approval and CI pass

---

## Risk Assessment

### High Risk 🔴

**GPU Hardware Availability**
- **Risk**: No access to RTX 5080/3090 Ti for validation
- **Impact**: Cannot validate GPU kernels, blocking Phases 2-5
- **Mitigation**: 
  - Prioritize CPU implementations and tests
  - Use CI/CD with GPU runners when available
  - Consider cloud GPU instances (AWS, Lambda Labs)

**CubeCL Performance**
- **Risk**: CubeCL overhead may reduce speedup gains
- **Impact**: May not hit 5x speedup targets
- **Mitigation**:
  - Profile early and often
  - Optimize hot paths
  - Consider falling back to raw CUDA if needed

### Medium Risk 🟠

**Numerical Stability**
- **Risk**: Ternary quantization may cause accuracy issues
- **Impact**: Model performance degradation >2%
- **Mitigation**:
  - Implement calibration methods
  - Add fine-tuning support
  - Monitor perplexity during testing

**Sparsity Threshold Tuning**
- **Risk**: Optimal thresholds may vary by model/hardware
- **Impact**: Suboptimal dispatch decisions
- **Mitigation**:
  - Make thresholds configurable
  - Add auto-tuning capability
  - Document recommended settings

### Low Risk 🟢

**Integration Complexity**
- **Risk**: Integrating multiple kernel types may be complex
- **Impact**: Development time increase
- **Mitigation**:
  - Follow established patterns from Flash Attention
  - Incremental integration with tests at each step

---

## Timeline Estimate

| Phase | Estimated Time | Dependencies |
|-------|----------------|--------------|
| Phase 2: Ternary Matmul GPU | 2-3 weeks | None (ready to start) |
| Phase 3: Ternary Attention | 2-3 weeks | Phase 2 |
| Phase 4: Advanced Sparsity | 1-2 weeks | Phase 2, 3 |
| Phase 5: E2E Integration | 2-3 weeks | Phase 2, 3, 4 |
| GPU Validation | Ongoing | Hardware access |

**Total**: 7-11 weeks for full implementation + validation

**Critical Path**: GPU hardware access determines when validation can begin

---

## Success Metrics

### Performance Targets

- [ ] **Ternary Matmul**: ≥5x speedup vs FP16 on 95% sparse
- [ ] **Ternary Attention**: ≥4x speedup vs Flash Attention
- [ ] **Memory**: ≥10x weight reduction
- [ ] **VRAM**: ≥70% reduction with all optimizations
- [ ] **Accuracy**: <2% perplexity degradation on WikiText-2

### Code Quality

- [ ] All tests passing (target: 200+ tests)
- [ ] Zero clippy warnings
- [ ] Documentation coverage >90%
- [ ] Benchmarks for all kernels

### Deliverables

- [ ] Working GPU kernels for ternary operations
- [ ] Full model quantization pipeline
- [ ] Comprehensive benchmarking suite
- [ ] Integration tests
- [ ] Updated documentation

---

## Next Steps for Approval

1. **Review this plan** - Verify phases and tasks align with goals
2. **Prioritize phases** - Confirm order and importance
3. **Allocate resources** - GPU hardware access timeline
4. **Approve implementation** - Green-light to create feature branches

Once approved, implementation will begin with:
- Create `feature/ternary-matmul-gpu` branch
- Implement Task 2.1 (u32 tensor interop)
- Proceed through Phase 2 tasks systematically

---

**Document Status**: Ready for Review  
**Last Updated**: 2026-01-07  
**Next Review**: After Phase 2 completion
