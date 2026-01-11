# GPU Development Setup Guide

**Last Updated**: 2026-01-10  
**Target GPU**: NVIDIA GeForce RTX 5080 (Compute Capability 12.0)

## Current Status

✅ **GPU Available**: RTX 5080 detected  
✅ **NVIDIA Driver**: 590.48.01 installed  
❌ **CUDA Toolkit**: Not installed (required for compilation)

## Prerequisites

### Check GPU Status

```bash
nvidia-smi
# Should show: NVIDIA GeForce RTX 5080, 16303 MiB
```

### Check Driver Version

```bash
nvidia-smi --query-gpu=driver_version --format=csv,noheader
# Should show: 590.48.01 or newer
```

## Installing CUDA Toolkit

The CUDA toolkit is required to compile GPU kernels. It includes `nvcc` (NVIDIA C++ Compiler) and CUDA libraries.

### Option 1: System Package Manager (Recommended for Ubuntu/Debian)

```bash
# Add NVIDIA package repository (if not already added)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Install CUDA Toolkit 12.x (matches RTX 5080 Compute Capability 12.0)
sudo apt-get install -y cuda-toolkit-12-6

# Add to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify installation
nvcc --version
```

### Option 2: NVIDIA Runfile Installer

```bash
# Download CUDA 12.6 runfile
wget https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda_12.6.0_535.183.01_linux.run

# Install (driver already installed, so skip driver)
sudo sh cuda_12.6.0_535.183.01_linux.run --toolkit --silent

# Add to PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Option 3: Conda Environment

```bash
conda create -n cuda-env python=3.11
conda activate cuda-env
conda install -c nvidia cuda-toolkit=12.6
```

## Verifying Installation

After installing CUDA toolkit:

```bash
# Check nvcc is available
nvcc --version

# Should show something like:
# nvcc: NVIDIA (R) Cuda compiler driver
# Copyright (c) 2005-2024 NVIDIA Corporation
# Build cuda_12.6.r12.6/...

# Check CUDA libraries
ls /usr/local/cuda/lib64/libcudart.so*

# Test compilation
cd ~/Documents/projects/rust-ai/unsloth-rs
cargo build --features cuda
```

## Building with CUDA Support

Once CUDA toolkit is installed:

```bash
# Build with CUDA feature
cargo build --features cuda

# Run tests with CUDA
cargo test --features cuda

# Run benchmarks with CUDA
cargo bench --features cuda

# Profile with CUDA
./scripts/profile-gpu.sh all
```

## Troubleshooting

### Error: `nvcc` not found

**Problem**: CUDA toolkit not installed or not in PATH

**Solution**:
```bash
# Check if nvcc exists
which nvcc

# If not found, add to PATH
export PATH=/usr/local/cuda/bin:$PATH
```

### Error: Cannot find `-lcudart`

**Problem**: CUDA libraries not in library path

**Solution**:
```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Error: Compute capability not supported

**Problem**: Installed CUDA version too old for RTX 5080

**Solution**: Install CUDA 12.0 or newer

### Error: Out of memory

**Problem**: Insufficient VRAM for operation

**Solution**:
- Reduce batch size in tests/benchmarks
- Use smaller sequence lengths
- Enable gradient checkpointing

## Environment Variables

For convenience, add these to `~/.bashrc` or `~/.zshrc`:

```bash
# CUDA Toolkit paths
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Optional: Specify CUDA version for cudarc
export CUDARC_CUDA_VERSION=12.6

# Optional: Enable CUDA debugging
export CUDA_LAUNCH_BLOCKING=1  # For debugging only (slower)
export RUST_BACKTRACE=1
```

## Next Steps After Setup

Once CUDA toolkit is installed:

1. **Run GPU Tests**:
   ```bash
   cargo test --features cuda
   ```

2. **Run GPU Profiling**:
   ```bash
   ./scripts/profile-gpu.sh flash-attention
   ```

3. **Compare CPU vs GPU**:
   ```bash
   ./scripts/profile-gpu.sh compare
   ```

4. **Full Profiling Suite**:
   ```bash
   ./scripts/profile-gpu.sh all
   ```

## Disk Space Requirements

- CUDA Toolkit 12.6: ~3-4 GB
- Rust build artifacts (with CUDA): ~5-10 GB
- Total recommended: 20 GB free space

## Performance Monitoring Tools

### NVIDIA Nsight Systems (Optional but Recommended)

For detailed kernel profiling:

```bash
# Download from NVIDIA website
wget https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2024_6/nsight-systems-2024.6.1_2024.6.1.92-1_amd64.deb

# Install
sudo dpkg -i nsight-systems-*.deb

# Profile with nsys
./scripts/profile-gpu.sh nsys

# View results
nsys-ui target/profiling/nsys_profile.nsys-rep
```

### NVIDIA Nsight Compute (Optional)

For detailed kernel analysis:

```bash
# Download from NVIDIA website
wget https://developer.nvidia.com/downloads/assets/tools/secure/nsight-compute/2024_4/ncu_2024.4.0_linux.run

# Install
sudo sh ncu_2024.4.0_linux.run --silent

# Profile specific kernel
ncu --target-processes all cargo bench --features cuda
```

## CI/CD Considerations

GitHub Actions runners do not have GPU access. Per project design:
- GPU tests/benchmarks run **locally only**
- CI runs CPU-only tests
- Use `./scripts/local-build.sh cuda` for local GPU builds
- Results documented in BENCHMARKING.md

## References

- [CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
- [RTX 5080 Specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/)
- [CubeCL Documentation](https://github.com/tracel-ai/cubecl)
- [Candle CUDA Backend](https://github.com/huggingface/candle/tree/main/candle-cuda)

---

**Status**: Driver installed, toolkit installation needed  
**Next**: Install CUDA toolkit, then run GPU profiling
