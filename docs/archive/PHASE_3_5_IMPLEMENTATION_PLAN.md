# Phase 3-5 Implementation Plan

**Date**: 2026-01-07  
**Status**: Awaiting Approval  
**Current Branch**: `copilot/review-phase-2-5-implementation`  
**Phase 2 Status**: ✅ Complete (121 tests passing)

## Executive Summary

This document outlines the implementation plan for Phases 3-5, building on the completed Phase 2 GPU ternary matmul kernels. The plan follows the same systematic approach that successfully delivered Phase 2: plan → implement → test → review → iterate.

**Timeline**: Estimated 10-15 days for all three phases  
**Approach**: One task at a time, with approval checkpoints

---

## Phase 3: Ternary Attention GPU Integration

**Objective**: Integrate ternary operations with Flash Attention for memory-efficient attention computation.

**Duration**: 4-6 days  
**Complexity**: High (combines two complex systems)

### Phase 3 Task Breakdown

#### Task 3.1: Q·K^T Ternary Scoring Kernel (1-2 days)

**Goal**: Implement GPU kernel for ternary Q·K^T computation.

**Implementation**:
- Create `ternary_attention_score_kernel` in new file `src/kernels/ternary/attention_cubecl.rs`
- Reuse vectorized matmul infrastructure from Phase 2
- Transpose-aware memory access patterns
- Batch processing for multiple attention heads
- Output to shared memory for softmax stage

**Key Features**:
```rust
// Compute S = Q · K^T where K is ternary
// Q: [batch, heads, seq_len, head_dim] - FP32
// K: [batch, heads, seq_len, head_dim] - Ternary bitsliced
// S: [batch, heads, seq_len, seq_len] - FP32 scores
```

**Tests**:
- Single head attention scoring
- Multi-head attention scoring
- Grouped query attention (GQA) support
- Numerical equivalence with FP32 reference
- Various sequence lengths (64, 128, 256, 512)

**Deliverables**:
- `ternary_attention_score_kernel()` function
- Configuration struct for attention scoring
- 5+ unit tests

---

#### Task 3.2: Online Softmax with Popcount (1 day)

**Goal**: Adapt Flash Attention's online softmax for ternary scoring output.

**Implementation**:
- Integrate with existing Flash Attention kernel
- Fused softmax computation on ternary scores
- Maintain numerical stability (subtract max)
- Efficient register usage for running statistics

**Key Features**:
```rust
// Online softmax algorithm:
// For each query position:
//   1. Compute ternary scores for K segment
//   2. Update running max
//   3. Update exp sum with rescaling
//   4. Final normalization
```

**Tests**:
- Softmax correctness on ternary scores
- Numerical stability tests
- Comparison with standard softmax
- Edge cases (very small/large values)

**Deliverables**:
- Modified Flash Attention kernel with ternary support
- 3+ integration tests

---

#### Task 3.3: Hybrid FP/Ternary Dispatch (1-2 days)

**Goal**: Intelligent dispatch between full FP32 and ternary attention based on model configuration.

**Implementation**:
- Extend attention configuration with ternary settings
- Runtime decision based on:
  - Memory pressure (prefer ternary for large models)
  - Accuracy requirements (prefer FP32 for sensitive tasks)
  - Sparsity level (ternary more beneficial for sparse K/V)
- Unified API that abstracts the choice

**Key Features**:
```rust
pub enum AttentionMode {
    FullPrecision,           // Standard FP32 attention
    TernaryKeys,             // Ternary K, FP32 Q/V
    TernaryKeysValues,       // Ternary K/V, FP32 Q
    Adaptive,                // Runtime decision
}
```

**Tests**:
- Dispatch logic for each mode
- Mode switching without errors
- Memory usage comparison
- Accuracy validation for each mode

**Deliverables**:
- `AttentionMode` enum and configuration
- Dispatch function in attention module
- 4+ tests for different modes

---

#### Task 3.4: Causal Masking via Plane Operations (1 day)

**Goal**: Implement causal masking efficiently using bitsliced plane operations.

**Implementation**:
- Mask ternary planes directly (no FP conversion needed)
- Bitwise AND with causal mask pattern
- Zero out future positions in K planes
- Minimal memory overhead

**Key Features**:
```rust
// Causal mask on bitsliced planes:
// For position i, mask K[j] where j > i
// Apply mask directly to plus/minus planes
mask_plane = generate_causal_mask(seq_len)
k_plus_masked = k_plus & mask_plane
k_minus_masked = k_minus & mask_plane
```

**Tests**:
- Causal mask correctness
- Various sequence lengths
- Verify no information leakage
- Compare with FP32 causal attention

**Deliverables**:
- Causal masking helper functions
- Integration with attention kernel
- 3+ tests

---

#### Task 3.5: End-to-End Attention Integration Tests (1 day)

**Goal**: Comprehensive validation of ternary attention system.

**Implementation**:
- Full attention forward pass with ternary K/V
- Multi-layer attention stack
- Gradient flow tests (if applicable)
- Memory profiling
- Latency measurements (CPU simulation)

**Tests**:
- Single-layer attention
- Multi-layer transformer block
- Long sequence handling (1024+ tokens)
- Batch processing
- Memory usage validation

**Deliverables**:
- Integration test suite
- Performance characterization document
- 5+ end-to-end tests

---

### Phase 3 Success Criteria

- [ ] All ternary attention kernels compile and pass tests
- [ ] Numerical equivalence with FP32 attention (MAE < 1e-4)
- [ ] Hybrid dispatch working correctly
- [ ] Causal masking validated
- [ ] At least 20 new tests passing
- [ ] Memory usage documented
- [ ] Ready for GPU hardware validation

---

## Phase 4: Advanced Sparsity Optimization

**Objective**: Refine and optimize sparsity handling for maximum performance.

**Duration**: 3-4 days  
**Complexity**: Medium (optimization and tuning)

### Phase 4 Task Breakdown

#### Task 4.1: Dynamic Plane Skipping Refinement (1-2 days)

**Goal**: Improve plane skipping from Phase 2 with dynamic optimization.

**Implementation**:
- Runtime profiling of skip rates
- Adaptive chunk size selection
- Hot/cold chunk tracking
- Prefetch optimization for active chunks

**Key Features**:
```rust
pub struct DynamicSparsityConfig {
    // Adapt chunk size based on actual sparsity pattern
    min_chunk_size: u32,
    max_chunk_size: u32,
    // Track which chunks are frequently active
    hot_chunk_threshold: f32,
    // Enable prefetching for hot chunks
    enable_prefetch: bool,
}
```

**Tests**:
- Various sparsity patterns (uniform, clustered, random)
- Adaptive behavior validation
- Performance metrics (skip rate, overhead)

**Deliverables**:
- Enhanced sparse kernel configuration
- Runtime profiling utilities
- 4+ tests

---

#### Task 4.2: Chunk-Based Sparsity Optimization (1-2 days)

**Goal**: Optimize memory layout and access patterns for sparse chunks.

**Implementation**:
- Compressed storage for very sparse tensors
- Chunk-aligned memory allocation
- Vectorized chunk processing
- CSR-like format for ultra-sparse (>98%) weights

**Key Features**:
```rust
pub enum SparseTensorFormat {
    Bitsliced,              // Standard bitsliced (Phase 2)
    CompressedChunks,       // Only store active chunks
    CSR,                    // Compressed sparse row (ultra-sparse)
}
```

**Tests**:
- Format conversion correctness
- Memory savings validation
- Access pattern efficiency
- Performance on various sparsity levels

**Deliverables**:
- Multi-format sparse tensor support
- Conversion utilities
- 5+ tests

---

#### Task 4.3: Sparsity Profiler and Analysis Tools (1 day)

**Goal**: Tools to analyze and optimize sparsity patterns.

**Implementation**:
- Sparsity pattern visualizer (text-based)
- Per-layer sparsity statistics
- Chunk activity heatmap
- Optimization recommendations

**Key Features**:
```rust
pub struct SparsityProfile {
    total_elements: usize,
    zero_elements: usize,
    sparsity: f32,
    chunk_activity: Vec<u8>,  // Per-chunk activity count
    recommended_format: SparseTensorFormat,
    expected_speedup: f32,
}
```

**Tests**:
- Profile generation correctness
- Recommendation accuracy
- Various tensor patterns

**Deliverables**:
- Profiling utilities
- Analysis report generation
- 3+ tests

---

### Phase 4 Success Criteria

- [ ] Dynamic sparsity optimization working
- [ ] Multiple sparse formats supported
- [ ] Profiling tools functional
- [ ] At least 12 new tests passing
- [ ] Performance improvements documented
- [ ] Optimization guide written

---

## Phase 5: End-to-End Integration & Validation

**Objective**: Complete system integration with full model support and validation.

**Duration**: 3-5 days  
**Complexity**: High (full system integration)

### Phase 5 Task Breakdown

#### Task 5.1: Model Quantization Pipeline (1-2 days)

**Goal**: End-to-end pipeline to convert FP32 models to ternary.

**Implementation**:
- Model loading from HuggingFace/safetensors
- Layer-by-layer quantization
- Calibration data handling
- Mixed precision strategy
- Export quantized model

**Key Features**:
```rust
pub struct QuantizationPipeline {
    // Which layers to quantize
    pub layer_config: LayerQuantizationConfig,
    // Calibration settings
    pub calibration: CalibrationConfig,
    // Output format
    pub output_format: ModelFormat,
}

pub enum LayerQuantizationConfig {
    All,                    // Quantize everything
    AttentionOnly,          // Only attention K/V
    FFNOnly,                // Only feed-forward weights
    Custom(Vec<String>),    // Specific layers by name
}
```

**Tests**:
- Single layer quantization
- Full model quantization
- Format conversion
- Round-trip save/load

**Deliverables**:
- Quantization pipeline implementation
- CLI tool or API
- 5+ integration tests

---

#### Task 5.2: Comprehensive Benchmarking Suite (1-2 days)

**Goal**: Performance validation and comparison framework.

**Implementation**:
- CPU vs GPU comparison (when hardware available)
- FP32 vs Ternary accuracy comparison
- Memory usage profiling
- Latency measurements
- Throughput benchmarks

**Key Features**:
```rust
pub struct BenchmarkSuite {
    // Test configurations
    batch_sizes: Vec<usize>,
    sequence_lengths: Vec<usize>,
    model_sizes: Vec<ModelSize>,
    // Metrics to collect
    measure_latency: bool,
    measure_memory: bool,
    measure_accuracy: bool,
}
```

**Tests**:
- Benchmark harness correctness
- Metric collection accuracy
- Report generation

**Deliverables**:
- Benchmarking framework
- Sample benchmark results
- Performance report template
- 4+ tests

---

#### Task 5.3: Integration Tests and Validation (1 day)

**Goal**: Comprehensive validation of entire system.

**Implementation**:
- Full transformer layer tests
- Multi-layer model tests
- Real-world use case simulations
- Stress tests (large models, long sequences)
- Regression test suite

**Tests**:
- Single transformer block
- 6-layer, 12-layer models
- Various model architectures (GPT, LLaMA-style)
- Edge cases and error handling

**Deliverables**:
- Integration test suite
- Validation report
- 8+ end-to-end tests

---

#### Task 5.4: Documentation and Examples (1 day)

**Goal**: Complete documentation for end users.

**Implementation**:
- API documentation updates
- User guide for quantization
- Performance tuning guide
- Example notebooks/scripts
- Migration guide from FP32

**Deliverables**:
- Updated README
- API documentation
- User guide document
- 3+ example scripts
- Performance tuning guide

---

### Phase 5 Success Criteria

- [ ] Quantization pipeline functional
- [ ] Benchmarking suite complete
- [ ] All integration tests passing
- [ ] Documentation complete
- [ ] At least 17 new tests passing
- [ ] Example code working
- [ ] Ready for production use

---

## Overall Implementation Strategy

### Development Approach

1. **One Task at a Time**: Complete each task fully before moving to next
2. **Test-Driven**: Write tests first or alongside implementation
3. **Incremental Commits**: Commit after each verified task
4. **Review Checkpoints**: Request approval after each major milestone
5. **Documentation**: Update docs as we go, not at the end

### Testing Strategy

**Test Pyramid**:
- Unit tests (60%): Individual functions and kernels
- Integration tests (30%): Multi-component workflows
- End-to-end tests (10%): Full system validation

**Coverage Goals**:
- Phase 3: +20 tests (total: 141)
- Phase 4: +12 tests (total: 153)
- Phase 5: +17 tests (total: 170)

### Quality Gates

Each phase must pass:
- [ ] All tests passing
- [ ] No clippy warnings
- [ ] Documentation updated
- [ ] Code review completed
- [ ] Performance characterized

---

## Risk Assessment

### Technical Risks

**Integration Complexity** 🟡 MEDIUM
- Risk: Ternary + Flash Attention integration may have edge cases
- Mitigation: Extensive testing, gradual integration
- Impact: Could add 1-2 days to Phase 3

**GPU Hardware Unavailability** 🔴 HIGH
- Risk: Cannot validate actual GPU performance
- Mitigation: Comprehensive CPU simulation, defer GPU-specific tuning
- Impact: Performance claims unvalidated until hardware available

**Memory Layout Issues** 🟡 MEDIUM
- Risk: Sparse formats may have cache/alignment issues
- Mitigation: Multiple format options, profiling tools
- Impact: May need format redesign in Phase 4

**Numerical Stability** 🟢 LOW
- Risk: Ternary quantization affecting model accuracy
- Mitigation: Extensive validation tests, adjustable thresholds
- Impact: Well-understood problem with known solutions

---

## Timeline and Milestones

### Week 1 (Days 1-5): Phase 3
- Day 1: Task 3.1 (Q·K^T kernel)
- Day 2: Task 3.1 completion + Task 3.2 start (online softmax)
- Day 3: Task 3.2 completion + Task 3.3 start (hybrid dispatch)
- Day 4: Task 3.3 completion + Task 3.4 (causal masking)
- Day 5: Task 3.5 (integration tests) + **Phase 3 Review**

### Week 2 (Days 6-9): Phase 4
- Day 6: Task 4.1 (dynamic plane skipping)
- Day 7: Task 4.1 completion + Task 4.2 start (chunk optimization)
- Day 8: Task 4.2 completion + Task 4.3 (profiler)
- Day 9: Task 4.3 completion + **Phase 4 Review**

### Week 3 (Days 10-15): Phase 5
- Day 10: Task 5.1 (quantization pipeline)
- Day 11: Task 5.1 completion + Task 5.2 start (benchmarking)
- Day 12: Task 5.2 completion + Task 5.3 start (integration tests)
- Day 13: Task 5.3 completion + Task 5.4 start (documentation)
- Day 14: Task 5.4 completion
- Day 15: **Final review and PR preparation**

**Total**: 10-15 working days

---

## Dependencies and Prerequisites

### External Dependencies
- CubeCL v0.8.1 (already integrated)
- Candle ML framework (already integrated)
- No new dependencies required

### Internal Dependencies
- Phase 3 depends on Phase 2 (✅ complete)
- Phase 4 depends on Phase 3 (sparsity optimization uses attention kernels)
- Phase 5 depends on Phases 3 & 4 (full system integration)

### Hardware Requirements
- Development: CPU-only (all phases can develop without GPU)
- Validation: CUDA GPU (deferred until hardware available)
- Testing: CI/CD environment (CPU tests only)

---

## Deliverables Summary

### Code Artifacts
- 3 new kernel files (attention_cubecl.rs, sparsity_optimizer.rs, quantization_pipeline.rs)
- 49+ new tests across all phases
- Enhanced configuration system
- Profiling and analysis tools

### Documentation
- This implementation plan
- Phase-specific design documents
- API documentation updates
- User guide and examples
- Performance tuning guide

### Reports
- Phase completion reports (3)
- Integration test results
- Performance characterization (when GPU available)
- Final validation report

---

## Approval Checkpoints

I will pause for your approval at:
1. After this plan review (now)
2. After Phase 3 completion (~Day 5)
3. After Phase 4 completion (~Day 9)
4. Before creating final merge PR (~Day 14)

---

## Next Steps

Once this plan is approved:

1. **Immediate**: Begin Task 3.1 (Q·K^T Ternary Scoring Kernel)
2. **After Phase 3**: Review and approve before starting Phase 4
3. **After Phase 4**: Review and approve before starting Phase 5
4. **After Phase 5**: Create PR to merge into parent branch (please specify)

---

## Questions for Clarification

Before proceeding, please clarify:

1. **Parent Branch**: What is the target branch for the final PR merge?
   - Current branch: `copilot/review-phase-2-5-implementation`
   - Need to know: Target parent branch name

2. **Priority**: Should I prioritize any specific phase or feature?

3. **Scope Adjustments**: Any tasks to add/remove/modify?

4. **Hardware Timeline**: Any updates on GPU hardware availability?

---

**Status**: Awaiting Approval to Proceed

Once approved, I will begin systematic execution of Phase 3, Task 3.1.
