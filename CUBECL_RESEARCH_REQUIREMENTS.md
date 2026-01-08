# CubeCL Research Requirements for Flash Attention Implementation

**Status**: ✅ Research Complete (January 2026)  
**CubeCL Version**: v0.8.1 (Validated)  
**Reference Documents**: `docs/cubecl-context.md`, `docs/cubecl-guide.md`

## Purpose
This document outlines the specific information, documentation, and examples needed to implement a production-ready Flash Attention kernel using CubeCL. 

**Update (2026-01-06)**: Research phase completed. See `docs/cubecl-context.md` for validated API reference and `docs/cubecl-guide.md` for implementation roadmap.

---

## 1. CubeCL Core API Documentation

### 1.1 Kernel Definition and Launch
**Status**: ✅ RESOLVED

**What we learned:**
- Use `#[cube(launch_unchecked)]` for production (10-20% faster, skips bounds checks)
- Use `#[cube(launch)]` for debugging
- Generics: `<F: Float>` dispatches f32/f16/bf16
- No return values; mutate outputs directly via `&mut Array<F>`
- No dynamic allocation; everything comptime or register-based

**Validated syntax:**
```rust
use cubecl::prelude::*;

#[cube(launch_unchecked)]
fn example_kernel<F: Float>(
    input: &Array<F>,
    output: &mut Array<F>,
    scalar_param: F,
    config: Comptime<SomeConfig>,
) {
    let value = input[ABSOLUTE_POS];
    output[ABSOLUTE_POS] = F::exp(value) * scalar_param;
}
```

**Types that can be passed:**
- `&Array<F>` / `&mut Array<F>` - Primary input/output (1D buffer view)
- `&Array<Line<F>>` - Vectorized 4-element loads
- `F` (scalar) - Float constants
- `Comptime<T>` - Compile-time constants for unrolling

### 1.2 Thread and Block Indexing
**Status**: ✅ RESOLVED
**Status**: ✅ RESOLVED

**Available indexing primitives (validated):**
| Primitive | Description |
|-----------|-------------|
| `ABSOLUTE_POS` / `ABSOLUTE_POS_X/Y/Z` | Global thread ID |
| `CUBE_POS_X/Y/Z` | Block position in grid |
| `UNIT_POS_X/Y/Z` | Thread position in block |
| `CUBE_DIM_X/Y/Z` | Block dimensions |
| `PLANE_POS` | Warp/wavefront lane (0-31 NVIDIA, 0-63 AMD) |

**Best practice for 4D tensors [batch, heads, seq, dim]:**
```rust
// Flatten batch × heads on CUBE_POS_X, tile sequence on CUBE_POS_Y
let batch_head = CUBE_POS_X;
let tile_idx = CUBE_POS_Y;
let tid = UNIT_POS_X;

// Manual stride calculation
let batch_idx = batch_head / num_heads;
let head_idx = batch_head % num_heads;
let global_idx = batch_idx * (heads * seq * dim) + head_idx * (seq * dim) + ...;
```

**Maximum dimensions:** Backend-dependent (1024 threads/block typical for CUDA)

### 1.3 Tensor Operations
**Status**: ✅ RESOLVED
**Status**: ✅ RESOLVED

**Key finding:** CubeCL kernels work with `Array<F>` (1D buffer views), NOT high-level `Tensor<F>`.

**Matrix multiplication:**
- Use `cubek-matmul` crate for optimized Q@Kᵀ and Attn@V (tensor cores)
- Manual implementation: nested loops with register accumulation

**Element access:**
```rust
let value = array[ABSOLUTE_POS];  // 1D indexing only
// For 2D: array[row * cols + col]
```

**Built-in math functions:**
```rust
F::exp(x)      // Exponential
F::max(a, b)   // Maximum
F::neg_inf()   // Negative infinity (for softmax init)
F::new(2.0)    // Create constant (NOT F::from_f32)
```

**Reductions:**
```rust
warp_reduce(value, |a, b| a.max(b))  // Warp-level max
warp_reduce(value, |a, b| a + b)     // Warp-level sum
```

---

## 2. Memory Management

### 2.1 Shared Memory
**Status**: ✅ RESOLVED

**Critical finding:** SharedMemory is 1D only in CubeCL v0.8.1

```rust
#[cube(launch_unchecked)]
fn kernel_with_shared_memory<F: Float>(...) {
    // 1D allocation only - size must be comptime known
    let mut tile = SharedMemory::<F>::new(TILE_SIZE * HEAD_DIM);
    
    // Load data (manual 2D → 1D indexing)
    let row = UNIT_POS_X / HEAD_DIM;
    let col = UNIT_POS_X % HEAD_DIM;
    tile[row * HEAD_DIM + col] = input[global_idx];
    
    // Synchronize threads
    sync_units();  // NOT sync_threads()
}
```

**Typical shared memory limits:**
- RTX 3090 Ti: 48 KB per SM (configurable up to 100 KB)
- RTX 5080: ~64-100 KB per SM
- Tune tile_size accordingly (128 or 256)

**Bank conflict avoidance:** Pad strides (+1) or transpose access patterns.

### 2.2 Global Memory Access
**Status**: ✅ RESOLVED

**Memory layout:** Row-major (C-style), consecutive thread access for coalescing.

**Best practices:**
- Use `Array<Line<F>>` for vectorized 4-element loads (128-bit transactions)
- Ensure consecutive threads access consecutive memory addresses
- Align access to 128-byte boundaries when possible

```rust
// Coalesced access pattern (GOOD)
let idx = ABSOLUTE_POS;
output[idx] = input[idx];

// Vectorized access (BETTER)
// Array<Line<F>> loads 4 elements per thread
let vec = input[ABSOLUTE_POS];  // Loads 4 floats
output[ABSOLUTE_POS] = vec;
```

### 2.3 Register Allocation
**Status**: ✅ RESOLVED

**Validated pattern:**
```rust
#[cube(launch_unchecked)]
fn compute_attention<F: Float>(...) {
    // These are in registers - keep minimal count
    let mut acc_m = F::neg_inf();    // Running max
    let mut acc_l = F::new(0.0);     // Running sum
    let mut acc_o = Line::splat(F::new(0.0));  // Vectorized output
    
    // Use #[unroll] for small loops to reduce register pressure
    #[unroll]
    for i in 0..4 {
        // Small unrolled loop
    }
}
```

**Guidelines:**
- Keep accumulators minimal (1-4 per thread)
- Large arrays spill to local memory (slow)
- Profile with `CUBECL_PROFILE=1`

---

## 3. Launch Configuration

### 3.1 Grid and Block Dimensions
**Status**: ✅ RESOLVED

**Validated launch syntax:**
```rust
use cubecl::prelude::*;
use cubecl_cuda::CudaRuntime as Runtime;

fn launch_attention_kernel(q: &Tensor, ...) -> Result<Tensor> {
    let client = Runtime::client(&Default::default());
    
    // Grid: (batch * num_heads, num_q_tiles, 1)
    let cube_count = CubeCount::Static(
        (batch * num_heads) as u32,
        num_tiles,
        1,
    );
    
    // Block: 256 threads (warp-aligned)
    let cube_dim = CubeDim::new(256, 1, 1);
    
    // Launch with ArrayArg for buffer handles
    attention_kernel::launch_unchecked::<f32, Runtime>(
        &client,
        cube_count,
        cube_dim,
        ArrayArg::from_raw_parts(&q_handle, q_len, 4),  // vectorization=4
        ArrayArg::from_raw_parts(&k_handle, k_len, 4),
        ArrayArg::from_raw_parts(&v_handle, v_len, 4),
        ArrayArg::from_raw_parts(&out_handle, out_len, 4),
        ScalarArg::new(scale),
        Comptime::new(config),
    );
    
    Ok(output)
}
```

**Typical block sizes:** 128 or 256 (must be warp-aligned, max 1024)

### 3.2 Occupancy and Performance
**Status**: ✅ RESOLVED

**Profiling:** Set `CUBECL_PROFILE=1` environment variable.

**Performance targets:**
- GPU occupancy: >50%
- Shared memory: Stay within 48-64 KB per block
- Register count: <64 per thread for full occupancy

---

## 4. Runtime and Device Management

### 4.1 CubeCL Runtime Initialization
**Status**: ✅ RESOLVED

```rust
use cubecl_cuda::CudaRuntime as Runtime;

fn setup_cubecl() -> Result<()> {
    // Get default CUDA device client
    let client = Runtime::client(&Default::default());
    
    // Create buffer from bytes
    let handle = client.create(&tensor_bytes);
    
    // Allocate empty buffer
    let out_handle = client.empty(num_bytes);
    
    // Read buffer back to host
    let result_bytes = client.read(&out_handle);
    
    Ok(())
}
```

**Device selection:** Use `CudaDevice(n)` for multi-GPU.

### 4.2 Tensor Interoperability
**Status**: ✅ RESOLVED (implemented in `src/kernels/cubecl/interop.rs`)

**Candle → CubeCL:**
```rust
pub fn candle_to_cubecl_handle(tensor: &Tensor) -> Result<(Vec<u8>, Vec<usize>, DType)> {
    let tensor = tensor.contiguous()?;  // Must be contiguous
    let data: Vec<f32> = tensor.flatten_all()?.to_vec1()?;
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    Ok((bytes, tensor.dims().to_vec(), tensor.dtype()))
}
```

**CubeCL → Candle:**
```rust
pub fn cubecl_to_candle_tensor(bytes: &[u8], shape: &[usize], device: &Device) -> Result<Tensor> {
    let data: Vec<f32> = bytes.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Tensor::from_vec(data, shape, device)
}
```
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
