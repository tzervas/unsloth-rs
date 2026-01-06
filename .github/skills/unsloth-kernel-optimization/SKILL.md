---
name: unsloth-kernel-optimization
description: Optimize and implement GPU kernels using CubeCL for memory-efficient LLM training
---

# Kernel Optimization Skill

## When to Use

Invoke when the user asks to:
- Implement new GPU kernels with CubeCL
- Optimize existing kernel performance
- Profile and benchmark kernel execution
- Reduce VRAM usage in training operations
- Add CPU fallback implementations

## Performance Targets

- 2-5x speedup vs naive implementation
- 70-80% VRAM reduction vs baseline
- >50% GPU occupancy

## Kernel Implementation Workflow

### 1. Design Phase
- Document mathematical operation
- Calculate memory access pattern
- Choose between compute-bound vs memory-bound strategy

### 2. CPU Reference
```rust
fn operation_cpu(input: &Tensor) -> Result<Tensor> {
    // Correct implementation for validation
}
```

### 3. CubeCL Kernel
```rust
#[cube(launch)]
fn operation_kernel<F: Float>(
    input: &Tensor<F>,
    output: &mut Tensor<F>,
) {
    let idx = ABSOLUTE_POS;
    // GPU implementation
}
```

### 4. Dispatch
```rust
pub fn operation(input: &Tensor) -> Result<Tensor> {
    match input.device() {
        Device::Cuda(_) => operation_cuda(input),
        _ => operation_cpu(input),
    }
}
```

### 5. Benchmarking
```bash
cargo bench -p unsloth-rs -- kernel_name
```

## Memory Optimization Techniques

1. **Fused Operations** - Combine sequential ops
2. **Tiled Algorithms** - Use shared memory
3. **Streaming** - Process in chunks
4. **Mixed Precision** - f16/bf16 where possible

## Key Files

- `src/kernels/` - Kernel implementations
- `benches/kernels.rs` - Performance benchmarks
- [Global CUDA Instructions](../../.github/instructions/cuda-kernels.instructions.md)
