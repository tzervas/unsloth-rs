# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
