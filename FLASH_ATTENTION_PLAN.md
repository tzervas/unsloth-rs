# Flash Attention Implementation Plan

## Objective
Implement Fused Flash Attention GPU Kernel (Issue #5) using CubeCL for memory-efficient LLM training.

## Performance Targets
- **Speedup:** 2-5x vs naive implementation
- **VRAM Reduction:** 70-80% vs baseline
- **GPU Occupancy:** >50%
- **Numerical Accuracy:** Within 1e-5 tolerance vs CPU reference

## Implementation Phases

### Phase 1: Foundation & Research (Current)
**Goal:** Understand Flash Attention algorithm and CubeCL API

**Tasks:**
- [x] Review existing CPU reference implementation
- [x] Study Flash Attention paper and algorithm
- [ ] Explore CubeCL API and examples
- [ ] Design kernel memory layout and access patterns
- [ ] Document mathematical operations and optimization strategy

**Deliverables:**
- Algorithm documentation
- Memory access pattern design
- Kernel implementation strategy

---

### Phase 2: Basic CubeCL Kernel
**Goal:** Implement working CubeCL kernel with basic functionality

**Tasks:**
- [ ] Create CubeCL kernel module structure
- [ ] Implement basic Q·K^T computation in CubeCL
- [ ] Implement softmax computation
- [ ] Implement attention·V computation
- [ ] Add device dispatch (CUDA/CPU)
- [ ] Basic integration with FusedAttention struct

**Deliverables:**
- `src/kernels/attention_cubecl.rs` - CubeCL kernel implementation
- Working end-to-end attention computation on GPU

**Acceptance Criteria:**
- Kernel compiles without errors
- Produces output with correct shape
- No crashes or memory errors

---

### Phase 3: Optimization & Fusion
**Goal:** Optimize for performance and memory efficiency

**Tasks:**
- [ ] Implement tiling strategy for memory efficiency
- [ ] Fuse operations (single-pass Q·K^T·V)
- [ ] Optimize memory access patterns (coalesced access)
- [ ] Use shared memory for intermediate results
- [ ] Implement mixed precision support (f16/bf16)
- [ ] Optimize block/thread configuration

**Deliverables:**
- Optimized kernel with fused operations
- Memory-efficient tiled algorithm
- Tuned launch configuration

**Acceptance Criteria:**
- Reduced memory footprint (measure VRAM usage)
- Improved performance (benchmark vs baseline)
- Maintains numerical stability

---

### Phase 4: Testing & Validation
**Goal:** Ensure correctness and numerical stability

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

## Timeline Estimate

- **Phase 1 (Foundation):** 1-2 hours
- **Phase 2 (Basic Kernel):** 4-6 hours
- **Phase 3 (Optimization):** 4-6 hours
- **Phase 4 (Testing):** 2-3 hours
- **Phase 5 (Benchmarking):** 2-3 hours
- **Phase 6 (Documentation):** 1-2 hours

**Total:** 14-22 hours (2-3 work days)

---

## Current Status
- **Phase:** 1 (Foundation & Research)
- **Branch:** `feature/flash-attention-cubecl`
- **Next Action:** Explore CubeCL API and design kernel structure
