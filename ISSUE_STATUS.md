# Issue Status Tracking

**Last Updated**: 2026-07-22  
**CubeCL Version**: **0.9** (see `Cargo.toml` workspace deps; older notes referring to 0.8.1 are historical)  
**Crate version**: 1.0.3  

This document is a **historical tracker** of open issues relative to early experimental-branch work.
It is **not** a claim that GPU Flash Attention or Unsloth product parity is complete.
For residual GPU/env debt, see [DEBT.md](DEBT.md). For packaging, see [PUBLISHING.md](PUBLISHING.md).

> Status vocabulary for GPU/env: **PASS** | **FAIL** | **FAIL_ENV** (missing device / toolkit / arch pin) | **BLOCKED**.
> Default CI is CPU-only; a missing GPU must never be reported as a green GPU suite.

## Summary

**PR #10** merged foundational infrastructure into `experimental` branch (commit ff87fec):
- CPU reference implementations for all 4 core kernels
- Basic benchmarking infrastructure
- Memory estimation utilities
- Project documentation and skills

**Recent Progress (2026-01-06):**
- ✅ CubeCL v0.8.1 API research completed (see `docs/cubecl-context.md`, `docs/cubecl-guide.md`)
- ✅ Created `src/kernels/cubecl/` module structure with config, interop, kernel scaffolding
- ✅ Updated dependencies: `cubecl = "0.8.1"`, `cubecl-cuda = "0.8.1"`
- ✅ Implemented Candle ↔ CubeCL tensor conversion utilities

**Hardware Targets:**
- Phase 1: GeForce RTX 5080 (primary development)
- Phase 2: GeForce RTX 3090 Ti (validation)

## Issue-by-Issue Status

### Issue #5: [Kernel] Fused Flash Attention (Single-Pass Q·K^T·V) GPU Kernel Implementation
**Status:** In Progress (Phase 1 - Minimal Viable Kernel)
**Priority:** Highest

- ✅ CPU reference implementation exists (`src/kernels/attention.rs`)
- ✅ Benchmarking infrastructure exists (`benches/kernels.rs`)
- ✅ Memory estimation utilities exist (`src/memory.rs`)
- ✅ CubeCL v0.8.1 API research completed
- ✅ Module structure created (`src/kernels/cubecl/`)
- ✅ Candle ↔ CubeCL interop implemented
- ✅ Kernel configuration implemented (`FlashAttentionConfig`)
- 🚧 CubeCL GPU kernel implementation IN PROGRESS
- ❌ Numerical equivalence tests NOT done
- ❌ GPU benchmarks NOT done
- ❌ VRAM profiling NOT done

**Branch:** `feature/flash-attention-cubecl`  
**Estimated Completion:** 1-3 weeks for Phase 1 (RTX 5080 target)

---

### Issue #4, #8: [Kernel] Fused RMSNorm (with optional bias) GPU Kernel Implementation
**Status:** Not started (foundational work in place)
**Note:** Issues #4 and #8 are duplicates
- ✅ CPU reference implementation exists (`src/kernels/rmsnorm.rs`)
- ✅ Benchmarking infrastructure exists
- ❌ CubeCL GPU kernel NOT implemented
- ❌ Optional bias support NOT implemented
- ❌ Numerical equivalence tests NOT done
- ❌ GPU benchmarks NOT done

**Next steps:** Implement after Flash Attention is complete

---

### Issue #7: [Kernel] Fused Rotary Position Embedding (RoPE) GPU Kernel Implementation
**Status:** Not started (foundational work in place)
- ✅ CPU reference implementation exists (`src/kernels/rope.rs`)
- ✅ Benchmarking infrastructure exists
- ❌ CubeCL GPU kernel NOT implemented
- ❌ Fusion with Q/K computation NOT done
- ❌ Numerical equivalence tests NOT done
- ❌ GPU benchmarks NOT done

**Next steps:** Implement after Flash Attention is complete

---

### Issue #6: [Kernel] Fused SwiGLU Activation GPU Kernel Implementation
**Status:** Not started (foundational work in place)
- ✅ CPU reference implementation exists (`src/kernels/swiglu.rs`)
- ✅ Benchmarking infrastructure exists
- ❌ CubeCL GPU kernel NOT implemented
- ❌ Gate/up/down projection fusion NOT done
- ❌ Numerical equivalence tests NOT done
- ❌ GPU benchmarks NOT done

**Next steps:** Implement after Flash Attention is complete

---

### Issue #2: Comprehensive Kernel Benchmarking Suite for Performance & VRAM Profiling
**Status:** Partially complete (basic infrastructure only)
- ✅ Basic benchmarking infrastructure (`benches/kernels.rs`)
- ✅ CPU benchmarks for all 4 kernels
- ✅ Memory estimation utilities
- ❌ GPU benchmarking with CUDA profiling NOT done
- ❌ VRAM profiling across hardware configs NOT done
- ❌ CubeCL kernel benchmarks NOT possible (kernels don't exist yet)
- ❌ CI/CD integration NOT done
- ❌ Results documentation (`BENCHMARKS.md`) NOT created

**Dependencies:** Blocked on kernel implementations (#5, #6, #7, #8)

---

### Issue #1, #3: [Infra] Memory Pool Utility for Efficient VRAM Allocation
**Status:** Partially complete (basic structure only)
**Note:** Issues #1 and #3 are duplicates
- ✅ Basic `MemoryPool` struct exists (`src/memory.rs`)
- ✅ Allocation tracking implemented
- ✅ Memory estimation utilities implemented
- ❌ CubeCL integration NOT done
- ❌ Pre-allocation strategies NOT implemented
- ❌ Integration with fused kernels NOT done
- ❌ Benchmark validation NOT done

**Next steps:** Implement after core kernels are complete

---

### Issue #9: [Infra] CI/CD Branch Management & Merge Integration for Kernel Pipeline
**Status:** Not started
- ❌ Feature branches NOT created
- ❌ Merge automation NOT configured
- ❌ CI/CD pipeline NOT established
- ❌ Documentation NOT written

**Next steps:** Should be implemented in parallel with kernel work

---

## Current Priority: Flash Attention (Issue #5)

Flash Attention is the highest priority task because:
1. Marked as "Phase 1" milestone
2. Attention is the computational bottleneck in transformers
3. Targets 2-5x speedup and 70-80% VRAM reduction
4. Establishes the CubeCL kernel implementation pattern for subsequent work

**Active branch:** `feature/flash-attention-cubecl`

**Implementation Phases (Revised 2026-01-06):**

| Phase | Description | Hardware Target | Est. Time |
|-------|-------------|-----------------|----------|
| 1 | Minimal Viable Kernel | RTX 5080 | 1-3 weeks |
| 2 | Cross-GPU Validation | RTX 3090 Ti | 1-2 weeks |
| 3 | Advanced Features (f16, GQA) | Both | 2-4 weeks |
| 4 | Testing & Validation | Both | 1-2 weeks |
| 5 | Benchmarking | Both | 1-2 weeks |

**Current Phase 1 Progress:**
- [x] CubeCL API research (validated v0.8.1)
- [x] Module structure (`src/kernels/cubecl/`)
- [x] Tensor interop utilities
- [x] Kernel configuration
- [ ] Actual kernel implementation
- [ ] Test suite
- [ ] RTX 5080 profiling

**Reference Documents:**
- `docs/cubecl-context.md` - CubeCL v0.8.1 API reference
- `docs/cubecl-guide.md` - Implementation roadmap
- `CUBECL_IMPLEMENTATION_GUIDE.md` - Detailed kernel design
- `FLASH_ATTENTION_PLAN.md` - Phase breakdown
