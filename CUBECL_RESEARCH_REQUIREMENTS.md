# CubeCL Research Requirements for Flash Attention Implementation

## Purpose
This document outlines the specific information, documentation, and examples needed to implement a production-ready Flash Attention kernel using CubeCL. Use this as a research guide to gather the necessary materials for Phase 3+ implementation.

---

## 1. CubeCL Core API Documentation

### 1.1 Kernel Definition and Launch
**What we need:**
- Complete syntax and semantics of the `#[cube(launch)]` macro
- How to define kernel functions with generic float types `<F: Float>`
- Parameter passing conventions (tensors, scalars, configurations)
- Return value handling (if any)
- Compilation and code generation process

**Specific questions:**
- What types can be passed as kernel parameters?
- How are Rust types mapped to GPU types?
- What are the constraints on kernel function signatures?
- How does CubeCL handle generic float types (F32, F16, BF16)?

**Example needed:**
```rust
#[cube(launch)]
fn example_kernel<F: Float>(
    input: &Tensor<F>,
    output: &mut Tensor<F>,
    scalar_param: F,
    config: SomeConfig,
) {
    // Implementation
}
```

### 1.2 Thread and Block Indexing
**What we need:**
- All available indexing primitives (ABSOLUTE_POS, CUBE_POS_X/Y/Z, THREAD_POS, etc.)
- How to map multi-dimensional workloads to thread blocks
- Best practices for 2D/3D indexing patterns
- Block size limits and considerations

**Specific questions:**
- What's the difference between ABSOLUTE_POS, CUBE_POS, and THREAD_POS?
- How do we handle 4D tensors [batch, heads, seq, dim]?
- What are the maximum block/grid dimensions?
- How to compute global indices from local/block indices?

**Example needed:**
```rust
#[cube(launch)]
fn indexed_kernel<F: Float>(...) {
    let batch_idx = CUBE_POS_X;
    let head_idx = CUBE_POS_Y;
    let seq_idx = THREAD_POS;
    // How to combine these correctly?
}
```

### 1.3 Tensor Operations
**What we need:**
- Complete list of tensor operations available in CubeCL kernels
- Matrix multiplication APIs (if available)
- Element-wise operations
- Reduction operations (sum, max)
- Transpose operations

**Specific questions:**
- How to perform matrix multiplication in a kernel?
- Can we use Tensor<F> methods inside kernels?
- Are there built-in functions for common operations?
- How to access tensor elements by index?

**Examples needed:**
```rust
// Matrix multiplication
let result = matmul(a, b); // Does this exist?

// Element access
let value = tensor[batch, head, i, j]; // Syntax?

// Reductions
let max_val = row_max(tensor); // How to implement?
let sum_val = row_sum(tensor); // How to implement?
```

---

## 2. Memory Management

### 2.1 Shared Memory
**What we need:**
- How to allocate shared memory in CubeCL
- Size limits and best practices
- Synchronization primitives for shared memory
- Bank conflict avoidance strategies

**Specific questions:**
- What's the API for shared memory allocation?
- How to declare shared memory arrays?
- Is there a SharedMemory<F> type?
- How much shared memory is typically available?

**Example needed:**
```rust
#[cube(launch)]
fn kernel_with_shared_memory<F: Float>(...) {
    // Allocate shared memory for tiles
    let mut q_tile = SharedMemory::<F>::new(tile_size, head_dim); // Syntax?
    
    // Load data into shared memory
    // ... how?
    
    // Synchronize threads
    sync_threads(); // Does this exist?
}
```

### 2.2 Global Memory Access
**What we need:**
- Best practices for coalesced memory access
- How tensor data is laid out in memory
- Stride calculations for multi-dimensional tensors
- Performance characteristics of different access patterns

**Specific questions:**
- What's the memory layout for Tensor<F> (row-major, column-major)?
- How to ensure coalesced access?
- Are there alignment requirements?
- How to handle strided accesses?

### 2.3 Register Allocation
**What we need:**
- How to use registers effectively
- Local variable storage semantics
- Register pressure considerations
- When data spills to local memory

**Example needed:**
```rust
#[cube]
fn compute_attention(...) {
    // These should be in registers
    let mut acc = F::from_f32(0.0);
    let mut max_val = F::from_f32(-INFINITY);
    // How many registers can we use?
}
```

---

## 3. Launch Configuration

### 3.1 Grid and Block Dimensions
**What we need:**
- How to specify grid dimensions (number of blocks)
- How to specify block dimensions (threads per block)
- API for launching kernels with these configurations
- How to choose optimal configurations

**Specific questions:**
- What's the syntax for kernel launch?
- How to pass grid/block dimensions?
- Are there helper functions for common patterns?
- What are typical block sizes (128, 256, 512)?

**Example needed:**
```rust
fn launch_attention_kernel(...) -> Result<Tensor> {
    let grid_dim = (batch, num_heads, num_tiles);
    let block_dim = (tile_size, 1, 1);
    
    // How to launch?
    attention_kernel::launch(grid_dim, block_dim, q, k, v, output, ...)?;
    // Or different syntax?
}
```

### 3.2 Occupancy and Performance
**What we need:**
- How to measure and optimize GPU occupancy
- Trade-offs between block size, shared memory, and registers
- Performance profiling tools for CubeCL
- Debugging and validation techniques

**Specific questions:**
- How to profile CubeCL kernels?
- What tools are available for performance analysis?
- How to validate kernel correctness?
- How to debug kernel issues?

---

## 4. Runtime and Device Management

### 4.1 CubeCL Runtime Initialization
**What we need:**
- How to initialize CubeCL runtime
- Device selection and enumeration
- Context management
- Error handling

**Specific questions:**
- Is there a global runtime or per-device?
- How to detect available devices?
- How to select which GPU to use?
- What's the error handling model?

**Example needed:**
```rust
fn setup_cubecl() -> Result<CubeCLDevice> {
    // Initialize runtime
    let runtime = CubeCLRuntime::new()?;
    
    // Get device
    let device = runtime.get_device(0)?;
    
    Ok(device)
}
```

### 4.2 Tensor Interoperability
**What we need:**
- How to convert Candle tensors to CubeCL tensors
- Data transfer between CPU and GPU
- Zero-copy operations (if available)
- Memory ownership and lifetime management

**Specific questions:**
- Can we use Candle tensors directly in CubeCL?
- Do we need to copy data?
- How to handle tensor on different devices?
- What's the performance of data transfers?

**Example needed:**
```rust
fn convert_tensor(candle_tensor: &Tensor) -> Result<CubeCLTensor> {
    // Convert from Candle to CubeCL
    // How to do this?
}

fn launch_kernel(candle_q: &Tensor, ...) -> Result<Tensor> {
    // Convert inputs
    let cubecl_q = convert_tensor(candle_q)?;
    
    // Launch kernel
    // ...
    
    // Convert output back to Candle
    let candle_output = convert_back(cubecl_output)?;
    Ok(candle_output)
}
```

---

## 5. Flash Attention Specific Requirements

### 5.1 Softmax Implementation
**What we need:**
- Numerically stable softmax implementation in CubeCL
- Log-sum-exp trick implementation
- Reduction operations for computing max and sum
- Parallel reduction strategies

**Specific questions:**
- How to compute row-wise max in parallel?
- How to compute row-wise sum in parallel?
- What's the syntax for parallel reductions?
- How to handle numerical stability?

**Example needed:**
```rust
#[cube]
fn softmax_stable<F: Float>(scores: &Tensor<F>, output: &mut Tensor<F>) {
    // Find max value (for numerical stability)
    let max_val = /* how to compute row max? */;
    
    // Compute exp(x - max)
    // ...
    
    // Compute sum
    let sum_exp = /* how to compute row sum? */;
    
    // Normalize
    // ...
}
```

### 5.2 Tiled Matrix Multiplication
**What we need:**
- How to implement tiled matmul in CubeCL
- Loading tiles into shared memory
- Computing partial results
- Accumulation strategies

**Specific questions:**
- Best tile sizes for different hardware?
- How to overlap computation and memory access?
- Synchronization points in tiled algorithms?
- Handling edge cases (non-multiple tile sizes)?

**Example needed:**
```rust
#[cube]
fn tiled_matmul<F: Float>(
    a: &Tensor<F>,  // [M, K]
    b: &Tensor<F>,  // [K, N]
    c: &mut Tensor<F>,  // [M, N]
    tile_size: u32,
) {
    // Allocate shared memory for tiles
    // ...
    
    // Loop over tiles
    for tile_k in 0..num_tiles {
        // Load A tile
        // Load B tile
        // Sync
        // Compute partial result
        // Sync
    }
}
```

### 5.3 Online Softmax with Running Statistics
**What we need:**
- How to maintain running max and sum across tiles
- Update formulas for incremental statistics
- Memory layout for statistics
- Synchronization requirements

**Specific questions:**
- Where to store running statistics (registers, shared memory)?
- How to update statistics as we process tiles?
- What's the precision requirement for statistics?
- How to handle the final normalization?

**Example needed:**
```rust
#[cube]
fn online_softmax_update<F: Float>(
    scores: &Tensor<F>,     // Current tile scores
    values: &Tensor<F>,     // Current tile values
    running_max: &mut F,    // Running max
    running_sum: &mut F,    // Running sum
    output_acc: &mut [F],   // Output accumulator
) {
    // Compute local max
    let local_max = /* ... */;
    
    // Update running max
    let new_max = max(*running_max, local_max);
    
    // Update running sum with correction factor
    let correction = exp(*running_max - new_max);
    *running_sum = *running_sum * correction + /* local sum */;
    
    // Update output with correction
    // ...
    
    *running_max = new_max;
}
```

---

## 6. Testing and Validation

### 6.1 Unit Testing CubeCL Kernels
**What we need:**
- How to test kernels in isolation
- Setting up test fixtures
- Assertions and validation
- Test data generation

**Specific questions:**
- Can we call kernels directly in tests?
- How to set up small test tensors?
- What's the testing framework integration?
- How to test for numerical accuracy?

**Example needed:**
```rust
#[test]
fn test_attention_kernel() {
    // Create small test tensors
    let q = Tensor::randn(...);
    let k = Tensor::randn(...);
    let v = Tensor::randn(...);
    
    // Run kernel
    let output = flash_attention_cubecl(&q, &k, &v, scale, None)?;
    
    // Validate against reference
    let expected = flash_attention_fallback(&q, &k, &v, scale, None)?;
    assert_tensors_close(&output, &expected, 1e-5)?;
}
```

### 6.2 Numerical Validation
**What we need:**
- Tolerance levels for different precisions (F32, F16, BF16)
- Testing methodology for numerical stability
- Edge cases to test (large values, zeros, etc.)
- Comparison with reference implementations

---

## 7. Performance Benchmarking

### 7.1 Benchmarking Infrastructure
**What we need:**
- How to benchmark CubeCL kernels
- Timing measurements (wall clock vs. GPU time)
- Memory bandwidth measurements
- Throughput calculations

**Example needed:**
```rust
fn benchmark_flash_attention() {
    let mut group = c.benchmark_group("flash_attention");
    
    for seq_len in [512, 1024, 2048, 4096] {
        group.bench_function(format!("seq_{}", seq_len), |b| {
            b.iter(|| {
                flash_attention_cubecl(&q, &k, &v, scale, None)
            });
        });
    }
}
```

### 7.2 Performance Metrics
**What we need:**
- How to measure VRAM usage
- GPU occupancy measurement
- FLOPS calculation
- Comparison baseline establishment

---

## 8. Examples and Reference Implementations

### 8.1 Simple CubeCL Kernel Examples
**What we need:**
- Complete, working examples of CubeCL kernels
- Element-wise operation example
- Reduction operation example
- Matrix operation example
- Any example with shared memory

**Request:**
Please provide 2-3 complete, minimal, working examples of CubeCL kernels that we can build upon.

### 8.2 Similar Algorithm Implementations
**What we need:**
- Any existing attention implementations in CubeCL
- Matrix multiplication examples
- Softmax implementations
- Tiled algorithm examples

**Request:**
If there are any CubeCL examples in the wild (other repositories, examples directory), please provide links or code.

---

## 9. Documentation Resources

### 9.1 Official Documentation
**Request:**
- Link to official CubeCL documentation (API docs, guides, tutorials)
- GitHub repository links
- Any published papers or technical reports
- Blog posts or articles about CubeCL

### 9.2 Community Resources
**Request:**
- Discord/Slack channels
- Forum discussions
- GitHub issues with examples
- Stack Overflow questions

---

## 10. Hardware and Runtime Considerations

### 10.1 Target Hardware
**What we need:**
- CUDA compute capability requirements
- Shared memory sizes for different GPUs
- Register file sizes
- Warp/wavefront sizes

### 10.2 Backend Differences
**What we need:**
- Differences between CUDA, ROCm, and Vulkan backends
- API differences (if any)
- Performance characteristics
- Feature availability per backend

---

## Priority Ranking

### Critical (Must Have)
1. Kernel definition syntax and launch API (#1.1, #3.1)
2. Thread/block indexing (#1.2)
3. Tensor operations and access (#1.3)
4. Candle-CubeCL tensor conversion (#4.2)
5. At least one complete working example (#8.1)

### Important (Should Have)
6. Shared memory API (#2.1)
7. Softmax implementation strategy (#5.1)
8. Testing methodology (#6.1)
9. Runtime initialization (#4.1)
10. Official documentation links (#9.1)

### Nice to Have
11. Performance profiling (#3.2)
12. Tiled algorithms (#5.2)
13. Online softmax (#5.3)
14. Benchmarking infrastructure (#7.1)
15. Hardware specifications (#10.1)

---

## Deliverables Request

Please provide:

1. **Code Examples**: 3-5 working CubeCL kernel examples with explanations
2. **API Documentation**: Complete API reference for kernel development
3. **Tutorial/Guide**: Step-by-step guide for writing a CubeCL kernel
4. **Conversion Examples**: How to convert between Candle and CubeCL tensors
5. **Performance Guide**: Best practices for kernel optimization
6. **Testing Examples**: How to test and validate kernels

---

## How to Use This Document

1. Use sections 1-10 as a checklist for research
2. Prioritize items marked as "Critical" first
3. For each section, gather:
   - Code examples
   - API documentation
   - Links to resources
   - Working code snippets
4. Organize findings in a response document
5. Include links to all sources for reference

Once this information is available, we can proceed with implementing the actual CubeCL kernel in Phases 3-6.
