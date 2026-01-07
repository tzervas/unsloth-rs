# Flash Attention Implementation Plan

**CubeCL Version**: v0.8.1 (Validated January 2026)  
**Last Updated**: 2026-01-06  
**Status**: Phase 1 (Minimal Viable Kernel) - ✅ Completed

## Objective
Implement Fused Flash Attention GPU Kernel (Issue #5) using CubeCL for memory-efficient LLM training.

## Hardware Targets
- **Phase 1**: GeForce RTX 5080 (primary development/test machine)
- **Phase 2**: GeForce RTX 3090 Ti (validation and tuning)
- **Future**: A100/H100 (datacenter), AMD MI series, WGPU/CPU backends

## Performance Targets
- **Speedup:** 2-5x vs naive implementation
- **VRAM Reduction:** 70-80% vs baseline
- **GPU Occupancy:** >50%
- **Numerical Accuracy:** Within 1e-5 tolerance vs CPU reference (f32), 1e-3 (f16)

## Implementation Phases

### Phase 1: Minimal Viable Kernel (RTX 5080 Focus)
**Goal:** Correct, basic f32 Flash Attention on RTX 5080  
**Estimated Time:** 1-3 weeks  
**Status:** ✅ Completed (PRs #14 and #15)

**Tasks:**
- [x] Review existing CPU reference implementation
- [x] Study Flash Attention paper and algorithm
- [x] Complete CubeCL v0.8.1 API research (see `docs/cubecl-context.md`)
- [x] Create module structure (`src/kernels/cubecl/`)
- [x] Implement Candle ↔ CubeCL interop (`interop.rs`)
- [x] Implement kernel configuration (`config.rs`)
- [x] Implement basic tiled kernel with improved tiling (tile_size=256)
- [x] Add causal masking support
- [x] Add small-test equivalence suite (tolerance 1e-5) - 65 tests passing
- [ ] Profile on RTX 5080 (target >2x speedup vs Candle fallback) - **BLOCKED: Requires CUDA hardware**
- [ ] Validate VRAM reduction on seq=2048 - **BLOCKED: Requires CUDA hardware**

**Deliverables:**
- ✅ `src/kernels/cubecl/` module with kernel, interop, config
- ✅ Working Flash Attention kernel (f32, with causal masking)
- ✅ Test suite with numerical equivalence validation (65 tests)

**Acceptance Criteria:**
- ✅ All small tests pass (65/65)
- ✅ No NaN/Inf in outputs
- ✅ Matches Candle fallback within 1e-5 tolerance
- ⏸️ Measurable speedup on RTX 5080 - **BLOCKED: Requires CUDA hardware**

**Merged PRs:**
- PR #14: Phase 1 base implementation (commit e619c08)
- PR #15: Improved tiling + causal masking (commit 61800e6)

---

### Phase 2: Cross-GPU Validation (RTX 3090 Ti)
**Goal:** Cross-GPU compatibility and initial tuning  
**Estimated Time:** 1-2 weeks  
**Status:** Not Started

**Tasks:**
- [ ] Run Phase 1 kernel on RTX 3090 Ti
- [ ] Fix any arch-specific issues (shared memory limits, tensor core differences)
- [ ] Tune tile sizes per GPU (256 on 5080, 128 on 3090 Ti)
- [ ] Add causal masking via index checks
- [ ] Benchmark longer sequences (4096+ tokens)

**Acceptance Criteria:**
- Equivalence and stability on both GPUs
- Measurable VRAM savings vs O(N²) baseline

---

### Phase 3: Advanced Features & Precision
**Goal:** Production-ready features  
**Estimated Time:** 2-4 weeks  
**Status:** Not Started

**Tasks:**
- [ ] Padding/arbitrary mask support
- [ ] GQA/MQA via head repetition (repeat_interleave on K/V)
- [ ] f16/bf16 paths with tensor core usage (via `cubek-matmul`)
- [ ] Sliding-window attention variant
- [ ] Memory optimization (register pressure, shared memory tuning)

**Acceptance Criteria:**
- GQA support verified
- f16 accuracy within 1e-3 tolerance
- Mixed precision benchmarks documented

---

### Phase 4: Testing & Validation
**Goal:** Ensure correctness and numerical stability  
**Estimated Time:** 1-2 weeks  
**Status:** Not Started

**Tasks:**
- [ ] Add unit tests for kernel functions
- [ ] Implement numerical equivalence tests (CPU vs GPU, 1e-5 tolerance)
- [ ] Test edge cases (small batches, long sequences, GQA)
- [ ] Add property-based tests with proptest
- [ ] Validate gradient computation (if needed)
- [ ] Memory leak detection

**Deliverables:**
- Comprehensive test suite in `src/kernels/attention_cubecl.rs`
- Test coverage for all code paths
- Validation report

**Acceptance Criteria:**
- All tests pass
- CPU and GPU outputs match within 1e-5 tolerance
- No memory leaks detected
- Edge cases handled correctly

---

### Phase 5: Benchmarking & Profiling
**Goal:** Validate performance targets and document results

**Tasks:**
- [ ] Add GPU benchmarks to `benches/kernels.rs`
- [ ] Benchmark across configurations:
  - Batch sizes: 1, 4, 8, 16
  - Sequence lengths: 512, 1024, 2048, 4096
  - Hidden sizes: 768, 1024, 2048, 4096
- [ ] Profile VRAM usage per configuration
- [ ] Measure GPU occupancy
- [ ] Compare against baseline (Candle CUDA backend)
- [ ] Document results in BENCHMARKS.md

**Deliverables:**
- GPU benchmark suite
- VRAM profiling data
- Performance comparison report
- BENCHMARKS.md with results table

**Acceptance Criteria:**
- Achieves 2-5x speedup vs naive implementation
- Demonstrates 70-80% VRAM reduction (with checkpointing)
- GPU occupancy >50%
- Results documented and reproducible

---

### Phase 6: Documentation & Integration
**Goal:** Finalize integration and document usage

**Tasks:**
- [ ] Update API documentation
- [ ] Add usage examples
- [ ] Update README with Flash Attention status
- [ ] Document performance characteristics
- [ ] Add troubleshooting guide
- [ ] Update ISSUE_STATUS.md

**Deliverables:**
- Complete API documentation
- Usage examples in docs/README
- Updated project documentation

**Acceptance Criteria:**
- All public APIs documented
- Examples compile and run
- Documentation is clear and accurate

---

## Technical Architecture

### Module Structure
```
src/kernels/
  ├── attention.rs           # Main attention module (CPU reference)
  ├── attention_cubecl.rs    # CubeCL GPU kernel (NEW)
  └── mod.rs                 # Module exports
```

### Key Components

#### 1. CubeCL Kernel
```rust
#[cube(launch)]
fn flash_attention_kernel<F: Float>(
    q: &Tensor<F>,           // Query [batch, heads, seq, dim]
    k: &Tensor<F>,           // Key [batch, kv_heads, seq, dim]
    v: &Tensor<F>,           // Value [batch, kv_heads, seq, dim]
    output: &mut Tensor<F>,  // Output [batch, heads, seq, dim]
    scale: F,                // 1/sqrt(head_dim)
    // Tiling parameters
) {
    // Tiled Flash Attention algorithm
    // - Load Q, K, V tiles into shared memory
    // - Compute attention scores in blocks
    // - Fused softmax computation
    // - Compute output with running statistics
}
```

#### 2. Device Dispatch
```rust
impl FusedAttention {
    fn forward_cuda(&self, ...) -> Result<Tensor> {
        if self.config.use_flash && has_cubecl_support() {
            flash_attention_cubecl(...)
        } else {
            // Fallback to Candle CUDA backend
            self.forward_cpu(...)
        }
    }
}
```

#### 3. Memory Optimization
- **Tiling:** Process attention in tiles to fit in shared memory
- **Streaming:** Load tiles on-demand, minimize global memory access
- **Fused Operations:** Combine Q·K^T, softmax, and attention·V
- **Mixed Precision:** Use f16 for computation, f32 for accumulation

---

## Flash Attention Algorithm Overview

### Traditional Attention (Memory-Intensive)
```
1. Compute S = Q·K^T               [batch, heads, seq, seq]  # Large!
2. Compute P = softmax(S)          [batch, heads, seq, seq]  # Large!
3. Compute O = P·V                 [batch, heads, seq, dim]
```
**Memory:** O(batch × heads × seq²) - quadratic in sequence length

### Flash Attention (Memory-Efficient)
```
Tiled computation with online softmax:
1. Initialize output O, statistics (m, l)
2. For each tile of Q:
   For each tile of K, V:
     - Load Q_tile, K_tile, V_tile to shared memory
     - Compute S_tile = Q_tile·K_tile^T
     - Update softmax statistics incrementally
     - Compute attention output incrementally
     - Update running max and sum
3. Final normalization with accumulated statistics
```
**Memory:** O(batch × heads × seq × dim) - linear in sequence length

### Key Insights
1. **Online Softmax:** Compute softmax incrementally without materializing full matrix
2. **Tiling:** Process attention in blocks that fit in fast memory (shared memory)
3. **Recomputation:** Trade compute for memory (recompute in backward pass)
4. **IO-Awareness:** Minimize slow HBM access, maximize fast SRAM usage

---

## Risk Assessment

### High Risk
- **CubeCL API Complexity:** Limited documentation, steep learning curve
  - Mitigation: Start with simple kernels, iterate
- **Numerical Stability:** Softmax with large values can overflow
  - Mitigation: Use log-sum-exp trick, careful numerics

### Medium Risk
- **Performance Tuning:** Achieving 2-5x speedup requires optimization
  - Mitigation: Profile early, optimize hot paths
- **Memory Management:** Complex tiling and memory access patterns
  - Mitigation: Test thoroughly, use memory profiling tools

### Low Risk
- **Integration:** FusedAttention API already designed for dispatch
  - Mitigation: Follow existing patterns

---

## Dependencies & Prerequisites

### Required
- ✅ CubeCL v0.8.1 (configured)
- ✅ Candle v0.9 (configured)
- ✅ CPU reference implementation (exists)
- ✅ Benchmarking infrastructure (exists)

### Optional
- CUDA-capable GPU (for testing)
- Profiling tools (nsight, nvprof)

---

## Success Metrics

### Must Have
- [ ] Kernel compiles and runs without errors
- [ ] Numerical equivalence with CPU (1e-5 tolerance)
- [ ] All tests pass
- [ ] 2x minimum speedup vs naive implementation

### Should Have
- [ ] 3-5x speedup vs naive implementation
- [ ] 70-80% VRAM reduction
- [ ] GPU occupancy >50%
- [ ] Comprehensive benchmarks

### Nice to Have
- [ ] Support for various precisions (f32, f16, bf16)
- [ ] Adaptive tiling based on available memory
- [ ] Gradient checkpointing integration

---

## Timeline Estimate (Revised)

Based on validated CubeCL v0.8.1 API research:

| Phase | Description | Estimated Time |
|-------|-------------|----------------|
| Phase 1 | Minimal Viable Kernel (RTX 5080) | 1-3 weeks |
| Phase 2 | Cross-GPU Validation (RTX 3090 Ti) | 1-2 weeks |
| Phase 3 | Advanced Features & Precision | 2-4 weeks |
| Phase 4 | Testing & Validation | 1-2 weeks |
| Phase 5 | Benchmarking & Profiling | 1-2 weeks |
| Phase 6 | Documentation & Integration | 1 week |

**Total:** 7-14 weeks (realistic for production-ready kernel)

**Note:** Previous estimate of 14-22 hours was overly optimistic. CubeCL kernel development requires significant iteration for correctness and performance.

---

## Current Status
- **Phase:** 1 (Minimal Viable Kernel)
- **Branch:** `feature/flash-attention-cubecl`
- **Module:** `src/kernels/cubecl/`
- **Next Action:** Implement actual CubeCL kernel launch in `kernel.rs`
