# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- CustomOp `rmsnorm` / `swiglu` / `attention` accept [`DType::F16`]
  (f16 I/O, float accumulate). `ops` names unchanged. MAE vs f32 ref
  is the check (`< 2e-3`). f16 CUDA attention uses the online kernel,
  not the SRAM tile. `custom_op_f32_only()` is now false.
- `ops::attention_window` — sliding-window causal on the same CustomOp
  tiled/online path as `attention`. CUDA SRAM tiles honor the window.
  MAE vs masked softmax at s512 is the check, not a throughput claim.
- `rope_with_position_ids` + `RotaryEmbedding::forward` honors packed `[B,S]` ids.
- `fused_linear_cross_entropy` avoids `[N, V]` on every device: CPU CustomOp,
  others vocab-tile Candle GEMM (peak extra `[N, chunk]`).
- CustomOp CPU dots use AVX2+FMA when advertised (14700K has these; no
  AVX-512/AMX). Fused-CE rows parallelize via `std::thread`.
- CUDA fused-CE tile smoke on 5080: fwd matches CPU (`err < 1e-4`), bwd
  finite. Not a throughput claim.
- Compare harness: persist Unsloth in Podman volume `unsloth-rs-compare-site`
  (skip pip when import works). Image bake still avoided.
- `unsloth_rs::ops` — Triton-shaped names (`rmsnorm`, `rope`, `swiglu`,
  `attention`, `cross_entropy`, `fused_linear_ce`). Backends stay internal.
- CustomOp online-attention `cuda_fwd` (NVRTC). Default CUDA path no longer
  materializes `[B,H,S,S]` when there is no extra mask.
- CustomOp CUDA attention is SRAM-tiled Flash-style softmax for head dim
  ≤ 128 (owned NVRTC, not Unsloth PTX). Wider heads keep the HBM-streaming
  online kernel. Extra masks still Candle GEMM.
- `ops::{geglu, layernorm, rope_with_ids, attention_softcap}` close Unsloth
  Apache-2.0 kernel-name gaps (exact GeGLU, affine LayerNorm, packed RoPE
  ids, Gemma tanh softcap). Not peft/QLoRA/MoE/FP8.
- CustomOp CUDA call sites no longer write `unsafe`. Outputs use
  `alloc_zeros`. Remaining `unsafe`: `nvrtc::launch` (cudarc cannot
  check PTX ABI) and CPU AVX2+FMA in `cpu_isa` (runtime-detected).
- NVRTC C→PTX cache in `nvrtc::load_func` (`Arc<str>` by `module_name`).
- `cargo bench --features cuda --bench custom_op_cuda` writes
  `artifacts/custom_op_cuda.json` (host vs CUDA-event p50/p99).
  `CUDA_COMPUTE_CAP` + `TMPDIR` on this host. Missing feature/device/toolkit:
  `FAIL_ENV` exit 2. Kernel/launch errors: `FAIL`. Artifact write errors:
  `FAIL_IO`. `compile_cached` is first-vs-second NVRTC, not a launch-tax close.

### Changed
- Compare harness records torch / Unsloth / rust **host+sync p50/p99**
  (warmup 5, n=100) in `artifacts/py-rs-compare.json`. Not one-shot.
  Device-event rust numbers remain `artifacts/custom_op_cuda.json`.
- `custom_op_cuda` bench harvests `fused_linear_ce` host+event p50/p99
  (s128/s512 launch-bound + compute N=512 D=4096 V=32768). Not a 2× claim.
- `examples/compare_ops.rs` calls `unsloth_rs::ops` (`rmsnorm`, `rope`,
  `swiglu`, `attention`, `cross_entropy`), not `*_custom_op` names.
- **G0:** `flash_attention_cubecl` defaults to CustomOp / Candle CUDA (no
  CubeCL `to_vec1`). CubeCL FA only if `UNSLOTH_CUBECL_FA` is set.
- Added `attention_custom_op` / `attention_device` (online softmax on CPU).
- **`cubecl` is optional.** Default and CPU (`--no-default-features`) builds no longer pull the cubecl graph. Enable `cuda` (`dep:cubecl`, `cubecl/cuda`, `dep:cubecl-cuda`) when GPU kernels are needed.

### Documentation
- `docs/TRITON.md`: Triton compiler/FFI is
  [triton-bridge-rs](https://github.com/tzervas/triton-bridge-rs) `v0.1.0`
  (contract only). Hook: `kernels::triton_bridge` (no Cargo dep, never dispatches).
- `compare/README.md`: compare artifact now has torch/Unsloth/rust
  host+sync p50/p99. Event rust p50/p99 stays in
  `artifacts/custom_op_cuda.json`.

## [1.0.4] - 2026-08-17


### Added
- **G0 CustomOp family** (`kernels::custom_op`):
  - RMSNorm (`CustomOp2`)
  - SwiGLU `silu⊙up` (`CustomOp2`)
  - RoPE apply (`CustomOp3`)
  - Chunked cross-entropy (`CustomOp2`, mean over non-ignored tokens)
- Shared NVRTC helper (`custom_op::nvrtc`, `--features cuda`).
- Honesty flags: `custom_op_device_resident()`, `custom_op_f32_only()`.
- `RmsNorm` / `RotaryEmbedding` / `SwiGLU` / `fused_swiglu::swiglu` /
  `fused_rmsnorm_rope::{rmsnorm,rope}` now take the CustomOp path (f32).
- `chunked_cross_entropy(logits, targets, ignore_index, chunk_size)`.
- `docs/P1_CUSTOMOP_PLAN.md` (planning pass that this release implements).

### Notes
- CubeCL Flash Attention interop is **unchanged** (`interop_requires_host_roundtrip() == true`).
- CustomOp is f32-only. CE backward still allocates `dlogits [N,V]` (needed
  for `lm_head` autograd). Fused linear+CE is **not** in this crate.
- `position_ids` on `RotaryEmbedding::forward` remain unused (sequential
  `0..S` cache), same as 1.0.3.

### Documentation
- DEBT.md: P0d + P1 CustomOps landed; CubeCL host copy remains BLOCKED:api.

## [1.0.3] - 2026-07-22

### Fixed
- **Flash Attention scale** in `FusedAttention::forward_flash_attention`: pass `1/sqrt(head_dim)` into the multiply-scale API (was incorrectly using `sqrt(head_dim)`).
- **crates.io packaging case collision**: removed duplicate lowercase `roadmap.md`; keep only `ROADMAP.md` (1.0.2 tarball on crates.io is broken).

### Changed
- Package version **1.0.3** (patch: installable packaging + honesty; no API break intended for CPU kernel surface).
- README / crate docs: honest **kernels-only** positioning (not an Unsloth product port; no unproven 2× / 70% VRAM claims).
- PUBLISHING.md aligned with version 1.0.3 and packaging verification steps.
- Documented `CUDA_COMPUTE_CAP` pin and **FAIL_ENV** classification for missing GPU / toolkit mismatch (GPU_SETUP.md, DEBT.md, CI comments).
- **PR-070:** Documented permanent Candle↔CubeCL **host D2H/H2D** interop limitation; demoted FA speed claims; `interop_requires_host_roundtrip()`.
- **PR-071 / FINISH:** GPU numerical equivalence gate runs under `--features cuda` (not `#[ignore]`); MAE thresholds; **BLOCKED:env** vs **FAIL (accuracy)** classification. Default (no cuda) stays green.
- **FINISH:** CubeCL FA launch wraps `CudaRuntime::client` in `catch_unwind` (cudarc can panic `CUDA_ERROR_NO_DEVICE`); falls back to Candle CUDA. Document WSL `LD_LIBRARY_PATH=/usr/lib/wsl/lib` for healthy CubeCL on WSL hosts.
- **PR-083:** Removed public always-`Err` `compute_gradient_checkpointed` stub; checkpoint config remains for memory estimates only.

### Added
- Unit test `test_flash_path_scale_matches_cpu_one_over_sqrt_d` to catch scale regressions.
- `interop_requires_host_roundtrip()` honesty helper.
- `interop_f32_only()` / `interop_supports_dtype()` (UNS-P1-04: CubeCL path f32-only).
- GPU numerical gate env instructions in `tests/gpu/flash_attention.rs` and DEBT.md.

### Removed / archived
- Ternary CubeCL GPU drafts moved to `archive/ternary_cubecl/` (UNS-P2-01 non-goal; excluded from package).

### Notes
- crates.io **1.0.2** remains published but **unpacks broken** (case collision). Prefer **1.0.3**. Optional human yank of 1.0.2 only after 1.0.3 is live.

## [1.0.2] - 2026-01-25

### Changed
- Migrated transformer GPU kernels to CubeCL 0.9 API
- Updated `Bytes::from_bytes_vec()` for buffer creation
- Fixed `CubeDim::new()` 2-argument signature
- Replace `F::new()` with `F::cast_from()` for float construction
- Added `usize` suffix to SharedMemory::new() calls
- Added proper usize casts at array index sites
- Wrapped kernel launches in unsafe blocks with SAFETY comments
- Added cfg guards for CUDA-only variables

### Known Limitations
- Flash Attention kernel has numerical accuracy issues (under investigation)
- Some integration tests skip due to accuracy thresholds

## [1.0.1] - 2026-01-24

### Added
- CPU fallback warning when CUDA is unavailable

### Changed
- Bumped minimum Rust version to 1.92
- README badges added for crates.io and docs.rs

## [1.0.0] - 2026-01-24

### Added
- **Examples directory** with runnable examples:
  - `basic_attention.rs` - FusedAttention demonstration
  - `ternary_quantization.rs` - Ternary quantization with compression stats
- Comprehensive documentation for all training.rs functions
- CLAUDE.md for Claude Code development workflow
- Feature flags for experimental GPU kernels

### Fixed
- All clippy warnings resolved with strategic allows
- Documentation formatting and completeness

### Known Limitations (1.0)
- Flash Attention uses CPU reference implementation with Candle CUDA dispatch (not fused CubeCL kernel)
- Ternary quantization GPU kernels are experimental and not validated
- Gradient checkpointing is stub-only (returns NotImplemented error)
- Full CubeCL GPU kernel validation pending RTX 5080 hardware

### Fixed
- Gradient checkpointing stub now returns proper error instead of panicking
- Updated documentation to accurately reflect implemented vs planned features
- All source files now include SPDX license identifiers (MIT)

### Added
- CI/CD pipeline via GitHub Actions
  - Automated testing on push and pull requests
  - Clippy linting and rustfmt checks
  - Documentation build validation
  - Dependabot for dependency updates
- Large-scale integration tests
  - Multi-layer transformer stack testing
  - Long sequence attention (1024+ tokens)
  - Large batch processing validation
  - Gradient checkpointing configuration tests
  - Mixed precision mode validation
- Branch management and merge deconfliction strategy (BRANCH_STRATEGY.md)
- Local build script for GPU testing and Docker builds (scripts/local-build.sh)

### Changed
- CI configured for local-only GPU builds and benchmarks
- Documentation updated to reflect RTX 5080 GPU availability
- Performance targets marked as "pending validation" until GPU profiling complete
- Test count updated from 65 to 148 tests

## [0.1.0-alpha.1] - 2026-01-09

### Added

#### Core Transformer Components
- Multi-head attention with Grouped Query Attention (GQA) support
- Flash Attention infrastructure with CPU fallback (GPU kernels in development)
- Rotary Position Embeddings (RoPE) implementation
- RMS Normalization layer
- SwiGLU activation function

#### Ternary Quantization System
- Ternary weight quantization (values: {-1, 0, +1})
- Sparsity-aware compression with metadata
- Multiple calibration methods (AbsMax, Percentile, MeanStd)
- TernaryLinear layer with quantized weights
- Model-level quantization with skip patterns
- 5-15x memory compression ratios achieved

#### Memory Management
- VRAM estimation for attention operations
- Memory pool with device-aware tracking
- Gradient checkpointing configuration
- Peak memory usage tracking
- Out-of-memory error handling

#### Training Utilities
- Mixed precision training support (fp16, bf16, fp32)
- Loss and gradient scaling operations
- Dynamic loss scaling with overflow detection
- Gradient overflow detection (inf/nan handling)
- Training configuration with validation

#### Testing & Quality
- 114 unit tests covering all core functionality
- 34 integration tests validating end-to-end workflows
- Ternary quantization pipeline tests
- Flash Attention GPU infrastructure tests
- Memory tracking and VRAM estimation tests
- Training utilities validation tests
- Error handling robustness tests
- Comprehensive benchmarking suite
- 86% clippy warning reduction (283 → 39 warnings)

#### Documentation
- Complete API documentation for public interfaces
- Usage examples in README.md
- Memory estimation examples
- Benchmark results and performance baselines
- Publication guide (PUBLISHING.md)

### Notes

**Alpha Release Status**:
- This is an early alpha release for testing and community feedback
- Core functionality is tested and working (148 tests passing)
- GPU CUDA kernels require CUDA toolkit installation
- APIs may evolve based on feedback and usage patterns
- Not recommended for production use yet

**Known Limitations**:
- Flash Attention CubeCL GPU kernels still in development
- CUDA features require nvcc compiler installation
- Performance optimizations ongoing
- Some clippy warnings remain (primarily documentation and type complexity)

**Hardware Tested**:
- CPU: Intel/AMD x86_64 architectures
- GPU: NVIDIA RTX 5080 (16GB VRAM, CUDA 13.1, Driver 590.48.01)

**Feedback Welcome**:
- Issue reports: https://github.com/tzervas/unsloth-rs/issues
- Feature requests: https://github.com/tzervas/unsloth-rs/discussions
- Email: tz-dev@vectorweight.com
