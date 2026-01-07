# unsloth-rs Development Roadmap

**Status**: Phase 1 (Flash Attention) - In Progress  
**Last Updated**: 2026-01-06  
**CubeCL Version**: v0.8.1 (Validated)  
**Base Branch**: `experimental`

## Project Overview

`unsloth-rs` is a Rust implementation of transformer building blocks for LLM inference and fine-tuning, built on the Candle ML framework. The project aims to provide memory-efficient, high-performance GPU kernels using CubeCL.

## Hardware Targets

| Phase | GPU | Purpose |
|-------|-----|--------|
| Primary | GeForce RTX 5080 | Development and initial testing |
| Validation | GeForce RTX 3090 Ti | Cross-GPU compatibility |
| Future | A100/H100, AMD MI series | Datacenter deployment |

## Current Implementation Status

### ✅ Completed (on `experimental` branch as of ff87fec)

**Kernel Implementations** (CPU reference + Candle CUDA dispatch):
- Multi-head attention with grouped-query attention (GQA)
  - QKV projection and splitting
  - Scaled dot-product attention
  - Optional attention masking
  - Output projection
  - VRAM estimation utilities
  - Status: ✅ **Working** - CPU reference complete, CUDA path delegates to Candle backend
  
- Rotary Position Embeddings (RoPE)
  - Pre-computed cos/sin caches
  - Rotation in pairs (head_dim/2 pairs)
  - Separate CPU and CUDA paths (both functional)
  - Status: ✅ **Working** - Full implementation with tests
  
- RMS Normalization
  - Root mean square computation with epsilon
  - Learned scale parameters
  - CPU and CUDA paths
  - Status: ✅ **Working** - Numerical stability verified
  
- SwiGLU Activation
  - Gate, up, and down projections
  - Swish activation (x * sigmoid(x))
  - Element-wise multiplication
  - Status: ✅ **Working** - All operations functional

**Memory Management Utilities**:
- MemoryPool with allocation tracking and limits
  - Per-device type tracking (CPU, CUDA, Metal, Vulkan)
  - Peak memory monitoring
  - Efficiency calculations
  - Status: ✅ **Working** - Basic implementation complete
  
- Memory estimation functions
  - Forward pass memory estimation with checkpointing support
  - Attention VRAM estimation
  - Human-readable byte formatting
  - Status: ✅ **Working** - Estimation algorithms implemented
  
- Checkpoint configuration structure
  - Configuration for gradient checkpointing strategy
  - Memory reduction factor calculation
  - Status: ✅ **Scaffolding** - Config exists, but checkpointing not implemented

**Training Utilities**:
- Training configuration structure
  - Batch size, sequence length, accumulation steps
  - Mixed precision flag (not yet functional)
  - Checkpoint configuration integration
  - Status: ✅ **Scaffolding** - Structure exists, partial implementation
  
- Gradient scaling utilities
  - Function to scale gradients by factor
  - Status: ✅ **Working** - Basic implementation
  
- Gradient checkpointing function
  - Status: ❌ **Not Implemented** - Returns `unimplemented!()`

**Testing Infrastructure**:
- Unit tests: 20 tests passing
  - Attention: creation, forward shape, random input, numerical stability, VRAM estimate
  - RoPE: creation, shape preservation
  - RMSNorm: creation, forward pass, normalization, numerical stability
  - SwiGLU: creation, forward pass
  - Memory: pool allocation, device tracking, checkpoint reduction, format bytes, attention VRAM
  - Training: default config
  - Status: ✅ **Working** - Good coverage of basic functionality

**Benchmarking Infrastructure**:
- CPU benchmarks for all kernels
  - Attention, RoPE, RMSNorm, SwiGLU
  - Multiple batch sizes and sequence lengths
  - Memory estimates printed during benchmarks
  - Status: ✅ **Working** - Comprehensive CPU benchmark suite
  - Note: GPU benchmarks require `cuda` feature (not yet tested)

**Documentation**:
- Basic README with overview and usage examples
- Module-level documentation for all components
- Inline function documentation
- Status: ✅ **Partial** - Basic docs present, room for expansion

**Project Infrastructure**:
- Cargo configuration with features (cuda)
- MIT License
- .gitignore for Rust projects
- GitHub Skills directory (kernel optimization skill)
  - Status: ✅ **Complete** - Basic project setup done

### ❌ Not Yet Implemented (from initial planning/scaffold)

These features were mentioned in early planning but are **NOT** on the experimental branch:

- **Fused CubeCL GPU Kernels**
  - Current: Uses Candle's CUDA backend (not custom kernels)
  - Planned: Custom fused kernels for all operations
  
- **Flash Attention Algorithm**
  - Current: Standard attention with O(N²) memory
  - Planned: Memory-efficient O(N) Flash Attention
  
- **Gradient Checkpointing**
  - Current: Function exists but returns `unimplemented!()`
  - Planned: Full activation recomputation for memory savings
  
- **Mixed Precision Training**
  - Current: Flag exists in config but not functional
  - Planned: FP16/BF16 support with automatic casting
  
- **CubeCL Memory Pool Integration**
  - Current: Basic memory pool without device-aware allocation
  - Planned: Full CubeCL integration for GPU memory management
  
- **KV Cache for Inference**
  - Current: Parameter accepted but unused
  - Planned: Incremental decoding support
  
- **Distributed Training**
  - Current: No distributed support
  - Planned: Tensor and pipeline parallelism
  
- **GPU Benchmarks**
  - Current: Only CPU benchmarks functional
  - Planned: Full GPU benchmark suite with memory profiling
  
- **Integration Tests**
  - Current: Only unit tests
  - Planned: End-to-end transformer layer tests

### 🎯 Current Development State Summary

The `experimental` branch (ff87fec) has:
- ✅ Solid foundation with CPU reference implementations
- ✅ Working kernels that pass tests
- ✅ Good code structure and documentation
- ✅ Benchmark infrastructure for CPU
- ⚠️ GPU paths delegate to Candle (not custom fused kernels)
- ⚠️ No real gradient checkpointing yet
- ⚠️ No mixed precision support yet
- ⚠️ No Flash Attention yet

### 🚧 In Progress (2026-01-06)

**Flash Attention CubeCL Implementation:**
- ✅ CubeCL v0.8.1 API research completed (`docs/cubecl-context.md`, `docs/cubecl-guide.md`)
- ✅ Module structure: `src/kernels/cubecl/` with config, interop, kernel
- ✅ Dependencies updated: `cubecl = "0.8.1"`, `cubecl-cuda = "0.8.1"`
- ✅ Candle ↔ CubeCL tensor conversion utilities
- ✅ Kernel configuration (`FlashAttentionConfig`)
- 🚧 Actual CubeCL kernel implementation

This is a solid starting point for implementing the advanced features planned in the phases below.

## Development Phases

---

## Phase 1: Core Infrastructure 🎯

**Priority**: High  
**Estimated Effort**: 2-3 weeks  
**Dependencies**: None

### Task 1.1: Gradient Checkpointing Implementation
**Branch**: `experimental/gradient-checkpointing`  
**File**: `src/training.rs`  
**Status**: Not Started

**Objective**: Implement activation recomputation to reduce memory usage during training.

**Current State**:
```rust
// src/training.rs:44
// TODO: Implement gradient checkpointing
unimplemented!("Gradient checkpointing not yet implemented")
```

**Implementation Plan**:
1. Design checkpointing strategy (which layers to checkpoint)
2. Implement forward pass with selective activation saving
3. Implement backward pass with activation recomputation
4. Add tests to verify correctness and memory savings
5. Benchmark memory usage vs. compute overhead

**Success Criteria**:
- [ ] Function `compute_gradient_checkpointed` works correctly
- [ ] Memory usage reduced by 50-80% for checkpointed layers
- [ ] Tests verify gradient correctness
- [ ] Documentation updated with usage examples

**References**:
- Existing config: `CheckpointConfig` in `src/memory.rs`
- Memory estimation: `estimate_forward_memory()` in `src/memory.rs`

---

### Task 1.2: Mixed Precision Training Support
**Branch**: `experimental/mixed-precision`  
**Files**: `src/training.rs`, `src/kernels/mod.rs`  
**Status**: Not Started

**Objective**: Add FP16/BF16 support to reduce memory and increase throughput.

**Implementation Plan**:
1. Add dtype configuration to `TrainingConfig`
2. Implement dtype conversion utilities
3. Add mixed precision forward/backward passes
4. Implement loss scaling for numerical stability
5. Update all kernels to support FP16/BF16
6. Add tests for numerical accuracy

**Success Criteria**:
- [ ] All kernels support FP32, FP16, BF16
- [ ] Automatic mixed precision (AMP) configuration works
- [ ] Loss scaling prevents underflow
- [ ] Tests verify numerical stability within tolerance
- [ ] Benchmarks show 2x+ speedup on GPU

**New APIs**:
```rust
pub enum DType {
    F32,
    F16,
    BF16,
}

pub struct MixedPrecisionConfig {
    pub compute_dtype: DType,
    pub master_dtype: DType,
    pub loss_scale: f32,
    pub dynamic_scaling: bool,
}
```

---

### Task 1.3: Memory Pool CubeCL Integration
**Branch**: `experimental/memory-pool-cubecl`  
**File**: `src/memory.rs`  
**Status**: Not Started

**Objective**: Integrate device-aware memory allocation with CubeCL.

**Current State**:
```rust
// src/memory.rs:26
// Future versions will integrate with CubeCL for device-aware allocation.
```

**Implementation Plan**:
1. Study CubeCL memory management APIs
2. Add CubeCL memory pool backend
3. Implement per-device tracking (CUDA, Vulkan, Metal)
4. Add memory transfer utilities between devices
5. Implement memory pool statistics and monitoring
6. Add tests for multi-device scenarios

**Success Criteria**:
- [ ] Memory pool works with CubeCL devices
- [ ] Per-device memory tracking accurate
- [ ] Memory transfer utilities implemented
- [ ] Monitoring and statistics available
- [ ] Tests cover multi-device scenarios

---

## Phase 2: GPU Kernel Optimizations 🚀

**Priority**: High  
**Estimated Effort**: 4-6 weeks  
**Dependencies**: Phase 1 (partial)

### Task 2.1: Fused Flash Attention Kernel
**Branch**: `experimental/flash-attention-cubecl`  
**Files**: `src/kernels/cubecl/kernel.rs`, `src/kernels/attention_cubecl.rs`  
**Status**: 🚧 In Progress (Phase 1)

**Objective**: Implement single-pass QKV^T attention with O(N) memory complexity.

**Current Progress (2026-01-06):**
- ✅ CubeCL v0.8.1 API research completed
- ✅ Module structure created (`src/kernels/cubecl/`)
- ✅ Candle ↔ CubeCL interop implemented
- ✅ Kernel configuration implemented
- 🚧 CubeCL kernel launch implementation

**Implementation Plan (Revised):**

| Phase | Description | Hardware | Est. Time |
|-------|-------------|----------|----------|
| 1 | Minimal Viable Kernel (f32, non-masked) | RTX 5080 | 1-3 weeks |
| 2 | Cross-GPU Validation + Causal Mask | RTX 3090 Ti | 1-2 weeks |
| 3 | f16/bf16, GQA/MQA support | Both | 2-4 weeks |

**CubeCL v0.8.1 Key APIs:**
- `#[cube(launch_unchecked)]` - Performance kernel definition
- `Array<Line<F>>` - Vectorized 4-element loads
- `SharedMemory::<F>::new(size)` - 1D shared memory
- `sync_units()` - Block barrier
- `warp_reduce(val, |a, b| ...)` - Warp-level reductions

**Performance Targets**:
- 2-5x speedup vs. current implementation
- 70-80% VRAM reduction
- >50% GPU occupancy

**Success Criteria:**
- [ ] CubeCL kernel compiles and launches
- [ ] Numerical equivalence within 1e-5 (f32), 1e-3 (f16)
- [ ] Performance targets met on RTX 5080
- [ ] Cross-GPU validation on RTX 3090 Ti
- [ ] Benchmarks documented

**References:**
- `docs/cubecl-context.md` - CubeCL v0.8.1 API reference
- `docs/cubecl-guide.md` - Implementation roadmap

---

### Task 2.2: Optimized RoPE Kernel
**Branch**: `experimental/rope-optimization`  
**File**: `src/kernels/rope.rs`  
**Status**: Not Started

**Objective**: Fuse rotation computation for improved GPU utilization.

**Implementation Plan**:
1. Analyze current CPU implementation
2. Design fused CubeCL kernel for rotation
3. Optimize memory access patterns
4. Add shared memory for cos/sin caches
5. Benchmark against current implementation
6. Add tests for correctness

**Performance Targets**:
- 2-3x speedup vs. current implementation
- Efficient memory coalescing

**Success Criteria**:
- [ ] CubeCL kernel replaces CUDA path
- [ ] Performance targets met
- [ ] Tests verify correctness
- [ ] Benchmarks show improvements

---

### Task 2.3: Optimized RMSNorm Kernel
**Branch**: `experimental/rmsnorm-optimization`  
**File**: `src/kernels/rmsnorm.rs`  
**Status**: Not Started

**Objective**: Single-pass normalization with optional fusion.

**Implementation Plan**:
1. Implement single-pass RMS computation
2. Add Welford's online algorithm for stability
3. Optimize with warp-level primitives
4. Implement fusion with subsequent operations
5. Benchmark and test

**Performance Targets**:
- 2-4x speedup vs. baseline
- Kernel fusion reduces memory bandwidth

**Success Criteria**:
- [ ] Single-pass kernel implemented
- [ ] Performance targets met
- [ ] Fusion APIs available
- [ ] Tests verify numerical stability

---

### Task 2.4: Optimized SwiGLU Kernel
**Branch**: `experimental/swiglu-optimization`  
**File**: `src/kernels/swiglu.rs`  
**Status**: Not Started

**Objective**: Fuse gate, up, and down projections.

**Implementation Plan**:
1. Analyze current three-step implementation
2. Design fused CubeCL kernel
3. Optimize matrix multiplication layout
4. Add mixed precision support
5. Benchmark and test

**Performance Targets**:
- 2-3x speedup vs. baseline
- 40-50% VRAM reduction through fusion

**Success Criteria**:
- [ ] Fused kernel implemented
- [ ] Performance and memory targets met
- [ ] Mixed precision support added
- [ ] Tests verify correctness

---

## Phase 3: Testing and Benchmarking 🧪

**Priority**: Medium  
**Estimated Effort**: 2-3 weeks  
**Dependencies**: Phase 2

### Task 3.1: Comprehensive Benchmark Suite
**Branch**: `experimental/benchmark-expansion`  
**File**: `benches/kernels.rs`  
**Status**: Partially Complete

**Objective**: Expand benchmarks to cover all configurations and GPU execution.

**Implementation Plan**:
1. Add GPU benchmarks with `cuda` feature
2. Add memory usage profiling
3. Compare against baseline implementations
4. Add visualization of results
5. Document benchmarking methodology

**Success Criteria**:
- [ ] GPU benchmarks for all kernels
- [ ] Memory profiling integrated
- [ ] Comparison reports generated
- [ ] CI integration for performance regression testing

---

### Task 3.2: Integration Tests
**Branch**: `experimental/integration-tests`  
**File**: `tests/integration.rs` (new)  
**Status**: Not Started

**Objective**: End-to-end tests for transformer layers.

**Implementation Plan**:
1. Create integration test directory
2. Implement full transformer layer tests
3. Add gradient checkpointing validation
4. Add mixed precision correctness tests
5. Add memory limit tests

**Success Criteria**:
- [ ] Full transformer layer tests pass
- [ ] Gradient checkpointing verified
- [ ] Mixed precision accuracy validated
- [ ] Memory constraints tested

---

## Phase 4: Documentation and Examples 📚

**Priority**: Medium  
**Estimated Effort**: 1-2 weeks  
**Dependencies**: Phase 1, Phase 2

### Task 4.1: API Documentation
**Branch**: `experimental/api-docs`  
**Files**: All `src/**/*.rs`  
**Status**: Partially Complete

**Objective**: Comprehensive documentation for all public APIs.

**Implementation Plan**:
1. Audit current documentation coverage
2. Add missing docstrings
3. Add code examples to all public functions
4. Generate and review rustdoc output
5. Add architecture diagrams

**Success Criteria**:
- [ ] 100% documentation coverage for public APIs
- [ ] All examples compile and run
- [ ] Architecture guide added
- [ ] Published to docs.rs

---

### Task 4.2: Example Applications
**Branch**: `experimental/examples`  
**Files**: `examples/` (new directory)  
**Status**: Not Started

**Objective**: Practical examples demonstrating library usage.

**Implementation Plan**:
1. Create `examples/` directory
2. Add simple inference example
3. Add fine-tuning example with checkpointing
4. Add memory optimization guide
5. Add performance tuning guide

**Examples to Create**:
- `examples/simple_attention.rs` - Basic attention usage
- `examples/inference.rs` - Model inference workflow
- `examples/finetuning.rs` - Training with gradient checkpointing
- `examples/memory_optimization.rs` - Memory management patterns
- `examples/mixed_precision.rs` - Mixed precision training

**Success Criteria**:
- [ ] 5+ working examples
- [ ] Examples documented and tested
- [ ] README references examples
- [ ] CI runs examples

---

## Phase 5: Advanced Features 🔮

**Priority**: Low  
**Estimated Effort**: 3-4 weeks  
**Dependencies**: Phase 1, Phase 2, Phase 3

### Task 5.1: KV Cache for Inference
**Branch**: `experimental/kv-cache`  
**File**: `src/kernels/attention.rs`  
**Status**: Not Started

**Objective**: Enable incremental decoding for efficient inference.

**Current State**:
```rust
// src/kernels/attention.rs:121
// kv_cache parameter currently unused
_kv_cache: Option<(&Tensor, &Tensor)>
```

**Implementation Plan**:
1. Design KV cache data structure
2. Implement cache management (init, update, eviction)
3. Modify attention to use cached K/V
4. Add position tracking for incremental decode
5. Benchmark inference speedup
6. Add tests for cache correctness

**Success Criteria**:
- [ ] KV cache reduces inference latency
- [ ] Cache management is memory-efficient
- [ ] Tests verify correctness
- [ ] Documentation includes usage guide

---

### Task 5.2: Distributed Training Utilities
**Branch**: `experimental/distributed`  
**Files**: `src/distributed/` (new module)  
**Status**: Not Started

**Objective**: Support tensor and pipeline parallelism.

**Implementation Plan**:
1. Research distributed training patterns
2. Design API for tensor parallelism
3. Implement all-reduce and all-gather primitives
4. Add pipeline parallelism helpers
5. Add multi-GPU examples
6. Document distributed strategies

**Success Criteria**:
- [ ] Tensor parallelism API available
- [ ] Pipeline parallelism helpers working
- [ ] Multi-GPU tests pass
- [ ] Scaling benchmarks documented

---

## Branch Strategy

All development happens on feature branches off `experimental`:

```
experimental (base)
├── experimental/gradient-checkpointing
├── experimental/mixed-precision
├── experimental/memory-pool-cubecl
├── experimental/flash-attention-cubecl
├── experimental/rope-optimization
├── experimental/rmsnorm-optimization
├── experimental/swiglu-optimization
├── experimental/benchmark-expansion
├── experimental/integration-tests
├── experimental/api-docs
├── experimental/examples
├── experimental/kv-cache
└── experimental/distributed
```

### Merge Process
1. Feature branch created from `experimental`
2. Development and testing on feature branch
3. Code review
4. Merge to `experimental` for integration testing
5. After integration testing, merge to `main`

### Branch Naming Convention
- `experimental/<feature-name>` - New features
- `experimental/fix-<issue>` - Bug fixes
- `experimental/docs-<topic>` - Documentation updates

---

## Performance Targets

### Memory Efficiency
- 70-80% VRAM reduction with gradient checkpointing
- 40-50% VRAM reduction with mixed precision
- 50-60% VRAM reduction with fused kernels

### Compute Performance
- 2-5x speedup for Flash Attention vs. baseline
- 2-4x speedup for fused kernels vs. unfused
- >50% GPU occupancy for all kernels
- Linear scaling with tensor parallelism (up to 8 GPUs)

### Numerical Accuracy
- FP32: exact match with reference implementations
- FP16/BF16: within 1e-3 relative error
- Gradient checkpointing: exact gradient match

---

## Testing Strategy

### Unit Tests
- Per-function correctness tests
- Numerical stability tests
- Shape validation tests
- Edge case handling

### Integration Tests
- Full transformer layer tests
- Multi-component workflows
- Memory constraint tests
- Multi-device tests

### Performance Tests
- Benchmark suite for all kernels
- Memory profiling
- GPU occupancy measurement
- Scaling tests (multi-GPU)

### Property-Based Tests
- Use `proptest` for invariant checking
- Randomized input testing
- Fuzzing for edge cases

---

## Dependencies

### Current Dependencies
- `candle-core` 0.9 - Core tensor operations
- `candle-nn` 0.9 - Neural network layers
- `cubecl` 0.8.1 - GPU kernel framework (validated Jan 2026)
- `cubecl-cuda` 0.8.1 - CUDA runtime (optional, via `cuda` feature)
- `thiserror` 2.0 - Error handling
- `tracing` 0.1 - Logging/instrumentation

### Planned Dependencies
- `cubek-matmul` - Tensor-core accelerated matmul (when validated)
- `half` - FP16/BF16 support (if needed)
- `nccl` or equivalent - For distributed training

---

## Release Plan

### v0.2.0 (Q1 2026)
- Gradient checkpointing
- Mixed precision support
- Enhanced benchmarking

### v0.3.0 (Q2 2026)
- Flash Attention CubeCL kernel
- Optimized RoPE, RMSNorm, SwiGLU kernels
- Comprehensive documentation

### v0.4.0 (Q3 2026)
- KV cache for inference
- Integration tests
- Example applications

### v1.0.0 (Q4 2026)
- Production-ready
- Distributed training support
- Performance guarantees

---

## Contributing

See individual task branches for contribution opportunities. Each task has:
- Clear objectives and success criteria
- Implementation plan
- Test requirements
- Documentation requirements

For questions or discussions, open an issue or reach out to maintainers.

---

## References

### Papers
- Flash Attention: Dao et al., 2022
- RoFormer (RoPE): Su et al., 2021
- RMSNorm: Zhang & Sennrich, 2019
- SwiGLU: Shazeer, 2020

### Related Projects
- Candle: https://github.com/huggingface/candle
- CubeCL: https://github.com/tracel-ai/cubecl
- PyTorch: https://github.com/pytorch/pytorch
- Flash Attention: https://github.com/Dao-AILab/flash-attention

---

**Note**: This roadmap is a living document and will be updated as the project evolves.
