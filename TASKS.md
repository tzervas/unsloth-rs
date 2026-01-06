# unsloth-rs Task List

**Base Branch**: `experimental` (commit: ff87fec)  
**Last Updated**: 2026-01-06

This document provides a prioritized, actionable task list for implementing the features outlined in [ROADMAP.md](ROADMAP.md).

---

## 🚀 High Priority Tasks (Start Here)

These tasks provide the most value and should be completed first.

### Task 1: Implement Gradient Checkpointing
**Branch**: `experimental/gradient-checkpointing`  
**Estimated Time**: 3-5 days  
**Files**: `src/training.rs`

**Current State**: Function stub exists that returns `unimplemented!()`

**Subtasks**:
1. [ ] Design checkpointing strategy for transformer layers
2. [ ] Implement forward pass with selective activation storage
3. [ ] Implement backward pass with activation recomputation
4. [ ] Add unit tests to verify gradient correctness
5. [ ] Add integration test comparing with/without checkpointing
6. [ ] Benchmark memory usage reduction
7. [ ] Update documentation with usage examples
8. [ ] Add example to demonstrate checkpointing

**Success Criteria**:
- Function `compute_gradient_checkpointed()` works correctly
- Tests verify gradients match non-checkpointed version
- Memory usage reduced by 50-80% for checkpointed layers
- Documentation updated

**Entry Point**: `src/training.rs:36-47`

---

### Task 2: Add Mixed Precision Support
**Branch**: `experimental/mixed-precision`  
**Estimated Time**: 5-7 days  
**Files**: `src/training.rs`, `src/kernels/*.rs`

**Current State**: Boolean flag in config but no functionality

**Subtasks**:
1. [ ] Design mixed precision configuration API
2. [ ] Add dtype enum (F32, F16, BF16)
3. [ ] Implement dtype conversion utilities
4. [ ] Add loss scaling for numerical stability
5. [ ] Update attention kernel to support FP16/BF16
6. [ ] Update RoPE to support FP16/BF16
7. [ ] Update RMSNorm to support FP16/BF16
8. [ ] Update SwiGLU to support FP16/BF16
9. [ ] Add numerical accuracy tests
10. [ ] Benchmark performance improvements
11. [ ] Document mixed precision usage

**Success Criteria**:
- All kernels support F32, F16, BF16
- Numerical accuracy within tolerance (1e-3)
- Tests verify stability
- 2x+ speedup on GPU demonstrated

**Entry Point**: `src/training.rs:29` (mixed_precision field)

---

### Task 3: Implement Flash Attention CubeCL Kernel
**Branch**: `experimental/flash-attention-cubecl`  
**Estimated Time**: 7-10 days  
**Files**: `src/kernels/attention.rs`, new `src/kernels/attention_cubecl.rs`

**Current State**: Uses Candle's CUDA backend, not custom fused kernel

**Subtasks**:
1. [ ] Study Flash Attention paper (Dao et al., 2022)
2. [ ] Design tiling strategy for shared memory
3. [ ] Implement block-wise softmax computation
4. [ ] Write CubeCL kernel for attention
5. [ ] Add GQA support to fused kernel
6. [ ] Integrate kernel with existing attention API
7. [ ] Add numerical accuracy tests
8. [ ] Benchmark performance vs baseline
9. [ ] Verify memory usage reduction
10. [ ] Document Flash Attention implementation

**Success Criteria**:
- 2-5x speedup vs current implementation
- 70-80% VRAM reduction
- >50% GPU occupancy
- Numerical accuracy within 1e-4
- GQA support verified

**Entry Point**: `src/kernels/attention.rs:195-213` (forward_cuda method)

---

## 🎯 Medium Priority Tasks

These tasks enhance the project but can wait until high priority tasks are complete.

### Task 4: Optimize RoPE with CubeCL
**Branch**: `experimental/rope-optimization`  
**Estimated Time**: 3-5 days  
**Files**: `src/kernels/rope.rs`

**Subtasks**:
1. [ ] Design fused CubeCL kernel for rotation
2. [ ] Implement shared memory caching for cos/sin
3. [ ] Optimize memory access patterns
4. [ ] Benchmark vs current implementation
5. [ ] Add tests for correctness

**Success Criteria**:
- 2-3x speedup vs baseline
- Tests verify correctness

---

### Task 5: Optimize RMSNorm with CubeCL
**Branch**: `experimental/rmsnorm-optimization`  
**Estimated Time**: 2-4 days  
**Files**: `src/kernels/rmsnorm.rs`

**Subtasks**:
1. [ ] Implement single-pass RMS computation
2. [ ] Add Welford's algorithm for stability
3. [ ] Use warp-level primitives
4. [ ] Add kernel fusion API
5. [ ] Benchmark and test

**Success Criteria**:
- 2-4x speedup vs baseline
- Single-pass implementation
- Numerical stability verified

---

### Task 6: Optimize SwiGLU with CubeCL
**Branch**: `experimental/swiglu-optimization`  
**Estimated Time**: 3-5 days  
**Files**: `src/kernels/swiglu.rs`

**Subtasks**:
1. [ ] Design fused kernel for gate+up+down
2. [ ] Optimize matrix multiplication layout
3. [ ] Add mixed precision support
4. [ ] Benchmark and test

**Success Criteria**:
- 2-3x speedup vs baseline
- 40-50% VRAM reduction
- Mixed precision works

---

### Task 7: Enhance Memory Pool with CubeCL
**Branch**: `experimental/memory-pool-cubecl`  
**Estimated Time**: 4-6 days  
**Files**: `src/memory.rs`

**Subtasks**:
1. [ ] Study CubeCL memory APIs
2. [ ] Add CubeCL memory pool backend
3. [ ] Implement per-device tracking
4. [ ] Add memory transfer utilities
5. [ ] Add monitoring and statistics
6. [ ] Test multi-device scenarios

**Success Criteria**:
- Works with CubeCL devices
- Per-device tracking accurate
- Multi-device tests pass

---

### Task 8: Expand Benchmark Suite
**Branch**: `experimental/benchmark-expansion`  
**Estimated Time**: 2-3 days  
**Files**: `benches/kernels.rs`

**Subtasks**:
1. [ ] Add GPU benchmarks with cuda feature
2. [ ] Add memory profiling
3. [ ] Compare against baselines
4. [ ] Add result visualization
5. [ ] Document methodology

**Success Criteria**:
- GPU benchmarks functional
- Memory profiling integrated
- CI integration ready

---

### Task 9: Add Integration Tests
**Branch**: `experimental/integration-tests`  
**Estimated Time**: 3-4 days  
**Files**: `tests/integration.rs` (new)

**Subtasks**:
1. [ ] Create integration test directory
2. [ ] Add full transformer layer tests
3. [ ] Add gradient checkpointing validation
4. [ ] Add mixed precision tests
5. [ ] Add memory constraint tests

**Success Criteria**:
- End-to-end tests pass
- Checkpointing verified
- Mixed precision validated

---

## 📚 Low Priority Tasks

These tasks improve usability but aren't critical for core functionality.

### Task 10: Comprehensive API Documentation
**Branch**: `experimental/api-docs`  
**Estimated Time**: 3-5 days  
**Files**: All `src/**/*.rs`

**Subtasks**:
1. [ ] Audit documentation coverage
2. [ ] Add missing docstrings
3. [ ] Add code examples
4. [ ] Generate and review rustdoc
5. [ ] Add architecture diagrams

**Success Criteria**:
- 100% public API documentation
- All examples compile
- Published to docs.rs

---

### Task 11: Create Example Applications
**Branch**: `experimental/examples`  
**Estimated Time**: 3-5 days  
**Files**: `examples/` (new directory)

**Examples to Create**:
1. [ ] `simple_attention.rs` - Basic attention usage
2. [ ] `inference.rs` - Model inference workflow
3. [ ] `finetuning.rs` - Training with checkpointing
4. [ ] `memory_optimization.rs` - Memory management
5. [ ] `mixed_precision.rs` - Mixed precision training

**Success Criteria**:
- 5+ working examples
- Examples tested in CI
- README references examples

---

### Task 12: Implement KV Cache
**Branch**: `experimental/kv-cache`  
**Estimated Time**: 4-6 days  
**Files**: `src/kernels/attention.rs`, new `src/cache.rs`

**Current State**: Parameter accepted but unused

**Subtasks**:
1. [ ] Design KV cache data structure
2. [ ] Implement cache management
3. [ ] Modify attention to use cache
4. [ ] Add position tracking
5. [ ] Benchmark inference speedup
6. [ ] Add tests

**Success Criteria**:
- KV cache reduces latency
- Cache management efficient
- Tests verify correctness

---

### Task 13: Add Distributed Training Support
**Branch**: `experimental/distributed`  
**Estimated Time**: 10-14 days  
**Files**: `src/distributed/` (new module)

**Subtasks**:
1. [ ] Research distributed patterns
2. [ ] Design tensor parallelism API
3. [ ] Implement all-reduce/all-gather
4. [ ] Add pipeline parallelism
5. [ ] Add multi-GPU examples
6. [ ] Document strategies

**Success Criteria**:
- Tensor parallelism works
- Pipeline parallelism works
- Multi-GPU tests pass
- Scaling benchmarks done

---

## 📋 Quick Reference: File Locations

| Component | File | Status |
|-----------|------|--------|
| Attention | `src/kernels/attention.rs` | ✅ CPU ref, needs CubeCL |
| RoPE | `src/kernels/rope.rs` | ✅ Working, needs opt |
| RMSNorm | `src/kernels/rmsnorm.rs` | ✅ Working, needs opt |
| SwiGLU | `src/kernels/swiglu.rs` | ✅ Working, needs opt |
| Memory Pool | `src/memory.rs` | ✅ Basic, needs CubeCL |
| Training | `src/training.rs` | ⚠️ Scaffolding only |
| Benchmarks | `benches/kernels.rs` | ✅ CPU only |
| Tests | `src/**/*.rs` | ✅ 20 unit tests |
| Skill | `.github/skills/unsloth-kernel-optimization/SKILL.md` | ✅ Complete |

---

## 🔄 Task Workflow

For each task:

1. **Create branch** from `experimental`
   ```bash
   git checkout experimental
   git pull origin experimental
   git checkout -b experimental/<task-name>
   ```

2. **Implement** following the subtask list

3. **Test** thoroughly
   ```bash
   cargo test
   cargo clippy
   cargo bench  # if applicable
   ```

4. **Document** changes
   - Update inline docs
   - Update README if needed
   - Add examples if needed

5. **Commit and push**
   ```bash
   git add .
   git commit -m "feat: <description>"
   git push origin experimental/<task-name>
   ```

6. **Merge to experimental**
   - Create PR to `experimental` branch
   - Get code review
   - Run integration tests
   - Merge when approved

7. **Update this file** to mark task complete

---

## 📊 Progress Tracking

### Phase 1: Core Infrastructure (0/3 complete)
- [ ] Task 1: Gradient Checkpointing
- [ ] Task 2: Mixed Precision
- [ ] Task 7: Memory Pool CubeCL

### Phase 2: GPU Optimizations (0/4 complete)
- [ ] Task 3: Flash Attention
- [ ] Task 4: RoPE Optimization
- [ ] Task 5: RMSNorm Optimization
- [ ] Task 6: SwiGLU Optimization

### Phase 3: Testing & Benchmarking (0/2 complete)
- [ ] Task 8: Benchmark Expansion
- [ ] Task 9: Integration Tests

### Phase 4: Documentation (0/2 complete)
- [ ] Task 10: API Documentation
- [ ] Task 11: Example Applications

### Phase 5: Advanced Features (0/2 complete)
- [ ] Task 12: KV Cache
- [ ] Task 13: Distributed Training

**Overall Progress**: 0/13 tasks complete (0%)

---

## 🎯 Recommended Order

For a single developer working sequentially:

1. **Start with Task 1** (Gradient Checkpointing) - Immediate value, not GPU-dependent
2. **Then Task 2** (Mixed Precision) - Builds on Task 1, enables faster GPU work
3. **Then Task 3** (Flash Attention) - Highest impact GPU optimization
4. **Then Task 8** (Benchmark Expansion) - Validate optimizations
5. **Then Tasks 4-6** (Other kernel optimizations) - Incremental improvements
6. **Then Task 9** (Integration Tests) - Ensure stability
7. **Then Tasks 10-11** (Documentation/Examples) - Improve usability
8. **Finally Tasks 12-13** (Advanced features) - Nice-to-have features

---

## 💡 Tips for Success

- **Start small**: Focus on one task at a time
- **Test frequently**: Run tests after each subtask
- **Document as you go**: Don't leave docs for the end
- **Ask for help**: Use the skill agent for GPU kernel work
- **Benchmark early**: Verify performance improvements during development
- **Keep branches clean**: Merge to experimental frequently to avoid conflicts

---

For detailed information on each task, see [ROADMAP.md](ROADMAP.md).
