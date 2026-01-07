# unsloth-rs Task List

**Base Branch**: `experimental` (commit: ff87fec)  
**Last Updated**: 2026-01-06  
**CubeCL Version**: v0.8.1 (Validated)

This document provides a prioritized, actionable task list for implementing the features outlined in [ROADMAP.md](ROADMAP.md).

---

## 🚧 Currently In Progress

### Task 3: Implement Flash Attention CubeCL Kernel
**Branch**: `feature/flash-attention-cubecl`  
**Estimated Time**: 7-14 weeks (revised from 7-10 days based on research)  
**Status**: 🚧 Phase 1 - In Progress

**Current Progress (2026-01-06):**
- [x] CubeCL v0.8.1 API research completed (`docs/cubecl-context.md`, `docs/cubecl-guide.md`)
- [x] Module structure created (`src/kernels/cubecl/`)
- [x] Candle ↔ CubeCL tensor interop (`interop.rs`)
- [x] Kernel configuration (`config.rs`)
- [x] Kernel scaffolding (`kernel.rs`)
- [ ] Implement actual CubeCL kernel launch
- [ ] Add numerical equivalence tests
- [ ] Profile on RTX 5080
- [ ] Validate on RTX 3090 Ti

**Phase Breakdown:**

| Phase | Description | Hardware | Est. Time |
|-------|-------------|----------|----------|
| 1 | Minimal Viable Kernel (f32, non-masked) | RTX 5080 | 1-3 weeks |
| 2 | Cross-GPU Validation + Causal Mask | RTX 3090 Ti | 1-2 weeks |
| 3 | f16/bf16, GQA/MQA support | Both | 2-4 weeks |
| 4 | Testing & Validation | Both | 1-2 weeks |
| 5 | Benchmarking | Both | 1-2 weeks |

**Key CubeCL v0.8.1 APIs (Validated):**
```rust
// Kernel definition
#[cube(launch_unchecked)]
fn kernel<F: Float>(input: &Array<Line<F>>, output: &mut Array<Line<F>>) { ... }

// Launch configuration
let cube_count = CubeCount::Static(batch_heads, num_tiles, 1);
let cube_dim = CubeDim::new(256, 1, 1);
kernel::launch_unchecked::<F, Runtime>(&client, cube_count, cube_dim, args...);

// Shared memory (1D only)
let mut tile = SharedMemory::<F>::new(TILE_SIZE * HEAD_DIM);
sync_units();  // Barrier
```

**Next Steps:**
1. Implement CubeCL kernel launch in `src/kernels/cubecl/kernel.rs`
2. Add small-size test cases (batch=2, heads=4, seq=8-64)
3. Profile on RTX 5080

---

## 🚀 High Priority Tasks (After Flash Attention)

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

### Phase 2: GPU Optimizations (1/4 in progress)
- [🚧] Task 3: Flash Attention (In Progress - Phase 1)
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

**Overall Progress**: 1/13 tasks in progress (Flash Attention Phase 1)

---

## 🎯 Recommended Order (Revised 2026-01-06)

1. **🚧 Task 3** (Flash Attention) - Currently in progress, highest impact GPU optimization
2. **Then Task 8** (Benchmark Expansion) - Validate Flash Attention performance
3. **Then Task 1** (Gradient Checkpointing) - Memory optimization
4. **Then Task 2** (Mixed Precision) - Performance optimization
5. **Then Tasks 4-6** (Other kernel optimizations) - Apply Flash Attention patterns
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
