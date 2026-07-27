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
**Status:** Completed
**Priority:** Highest

- ✅ CPU reference implementation exists (`src/kernels/attention.rs`)
- ✅ Benchmarking infrastructure exists (`benches/kernels.rs`)
- ✅ Memory estimation utilities exist (`src/memory.rs`)
- ✅ CubeCL v0.9 API migration completed
- ✅ Module structure created (`src/kernels/cubecl/`)
- ✅ Candle ↔ CubeCL interop implemented
- ✅ Kernel configuration implemented (`FlashAttentionConfig`)
- ✅ CubeCL GPU kernel implementation completed (`src/kernels/cubecl/kernel.rs`)
- ✅ Numerical equivalence tests completed (65 tests passing, MAE < 1e-5)
- ✅ GPU benchmarks integrated
- ✅ VRAM profiling integrated

---

### Issue #4, #8: [Kernel] Fused RMSNorm (with optional bias) GPU Kernel Implementation
**Status:** Completed
**Note:** Issues #4 and #8 are duplicates
- ✅ CPU reference implementation exists (`src/kernels/rmsnorm.rs`)
- ✅ Benchmarking infrastructure exists
- ✅ CubeCL GPU kernel implemented (`fused_rmsnorm_rope.rs`)
- ✅ Optional bias and scale parameters supported
- ✅ Numerical equivalence tests completed and passing
- ✅ GPU benchmarks integrated

---

### Issue #7: [Kernel] Fused Rotary Position Embedding (RoPE) GPU Kernel Implementation
**Status:** Completed
- ✅ CPU reference implementation exists (`src/kernels/rope.rs`)
- ✅ Benchmarking infrastructure exists
- ✅ CubeCL GPU kernel implemented (`fused_rmsnorm_rope.rs`)
- ✅ Fusion with RMSNorm computation supported (`fused_rmsnorm_rope`)
- ✅ Numerical equivalence tests completed and passing
- ✅ GPU benchmarks integrated

---

### Issue #6: [Kernel] Fused SwiGLU Activation GPU Kernel Implementation
**Status:** Completed
- ✅ CPU reference implementation exists (`src/kernels/swiglu.rs`)
- ✅ Benchmarking infrastructure exists
- ✅ CubeCL GPU kernel implemented (`fused_swiglu.rs`)
- ✅ Gate/up/down projection fusion supported (`fused_ffn_swiglu`)
- ✅ Numerical equivalence tests completed and passing
- ✅ GPU benchmarks integrated

---

### Issue #2: Comprehensive Kernel Benchmarking Suite for Performance & VRAM Profiling
**Status:** Completed
- ✅ Basic benchmarking infrastructure (`benches/kernels.rs`)
- ✅ CPU benchmarks for all 4 kernels
- ✅ Memory estimation utilities
- ✅ GPU benchmarking with CUDA feature supported
- ✅ VRAM profiling across hardware configs supported
- ✅ CubeCL kernel benchmarks integrated and functional
- ✅ CI/CD integration configured
- ✅ Performance expectations documented

---

### Issue #1, #3: [Infra] Memory Pool Utility for Efficient VRAM Allocation
**Status:** Completed
**Note:** Issues #1 and #3 are duplicates
- ✅ Basic `MemoryPool` struct exists (`src/memory.rs`)
- ✅ Allocation tracking implemented
- ✅ Memory estimation utilities implemented
- ✅ CubeCL integration supported
- ✅ Peak memory and efficiency tracking implemented
- ✅ Integration with fused kernels supported
- ✅ Benchmark validation completed

---

### Issue #9: [Infra] CI/CD Branch Management & Merge Integration for Kernel Pipeline
**Status:** Completed
- ✅ Feature branches created and validated
- ✅ Merge automation and issue close policies configured
- ✅ CI/CD pipeline established (`.github/workflows/ci.yml`, `fleet-ci.yml`)
- ✅ Complete developer and workflow documentation written

---

## Completed Priority: Flash Attention (Issue #5)

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
