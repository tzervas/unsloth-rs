# Flash Attention CubeCL Kernel Implementation Guide

**CubeCL Version**: v0.8.1 (Validated January 2026)  
**Last Updated**: 2026-01-06

## Current Status

Phase 1 (Foundation) is complete. Research phase completed with validated API documentation.

**Infrastructure in place:**
- Module structure: `src/kernels/cubecl/` with `mod.rs`, `config.rs`, `interop.rs`, `kernel.rs`
- Device detection and dispatch logic
- Fallback implementation using Candle operations
- Comprehensive test suite
- VRAM estimation utilities
- Candle ↔ CubeCL tensor conversion utilities

**Hardware Targets:**
- **Phase 1**: GeForce RTX 5080 (primary development)
- **Phase 2**: GeForce RTX 3090 Ti (validation and tuning)
- **Future**: A100/H100, AMD MI series, WGPU/CPU backends

## Phase 2: CubeCL Kernel Implementation

### Overview
Implementing the actual CubeCL GPU kernel requires understanding of:
1. CubeCL kernel syntax and macros
2. Flash Attention tiled algorithm
3. GPU memory hierarchy (global, shared, registers)
4. Softmax numerical stability
5. Candle tensor interoperability

### CubeCL Kernel Basics (v0.8.1 Validated API)

CubeCL uses Rust macros to define GPU kernels that compile to multiple backends (CUDA, ROCm, Vulkan, Metal, WebGPU, CPU).

```rust
use cubecl::prelude::*;

#[cube(launch_unchecked)]  // Use launch_unchecked for ~10-20% performance gain
fn simple_kernel<F: Float>(input: &Array<F>, output: &mut Array<F>) {
    let idx = ABSOLUTE_POS;  // Global thread index
    if idx < input.len() {
        output[idx] = input[idx] * F::new(2.0);
    }
}

// Vectorized version for coalesced memory access
#[cube(launch_unchecked)]
fn vectorized_kernel<F: Float>(input: &Array<Line<F>>, output: &mut Array<Line<F>>) {
    if ABSOLUTE_POS < input.len() / 4 {
        let vec = input[ABSOLUTE_POS];  // Loads 4 elements (128-bit)
        output[ABSOLUTE_POS] = vec * Line::splat(F::new(2.0));
    }
}
```

**Key CubeCL v0.8.1 Concepts:**
- `#[cube(launch_unchecked)]` - Performance-optimized kernel (skip bounds checks)
- `#[cube(launch)]` - Debug-friendly kernel (with bounds checks)
- `Array<F>` - 1D buffer view (global memory), NOT `Tensor<F>`
- `Array<Line<F>>` - Vectorized 4-element loads for coalescing
- `SharedMemory::<F>::new(size)` - 1D shared memory (NOT 2D)
- `Float` trait - Works with f32, f16, bf16
- `F::new(value)` - Create float constant (NOT `F::from_f32`)
- `F::neg_inf()` - Negative infinity for softmax

**Thread Indexing Primitives:**
| Primitive | Description |
|-----------|-------------|
| `ABSOLUTE_POS` / `ABSOLUTE_POS_X/Y/Z` | Global thread ID |
| `CUBE_POS_X/Y/Z` | Block position in grid |
| `UNIT_POS_X/Y/Z` | Thread position in block |
| `CUBE_DIM_X/Y/Z` | Block dimensions |
| `PLANE_POS` | Warp/wavefront lane (0-31/63) |

### Flash Attention Algorithm Implementation

The Flash Attention algorithm processes attention in tiles to minimize memory usage.

#### Traditional Attention (Memory-Intensive)
```
S = Q·K^T                    # [batch, heads, seq, seq] - O(seq²) memory
P = softmax(S, dim=-1)       # [batch, heads, seq, seq]
O = P·V                      # [batch, heads, seq, dim]
```

#### Flash Attention (Memory-Efficient)
```
# Divide Q into Tr blocks, K/V into Tc blocks
# Tile sizes: typically 128x128 or 256x256

for i in range(Tr):  # For each Q block
    Q_i = Q[i*Br:(i+1)*Br, :]  # Load Q tile
    O_i = zeros(Br, d)          # Output for this tile
    l_i = zeros(Br)             # Sum for softmax denominator
    m_i = -inf * ones(Br)       # Max for softmax numerator
    
    for j in range(Tc):  # For each K/V block
        K_j = K[j*Bc:(j+1)*Bc, :]  # Load K tile
        V_j = V[j*Bc:(j+1)*Bc, :]  # Load V tile
        
        # Compute attention scores for this tile
        S_ij = Q_i @ K_j^T / sqrt(d)  # [Br, Bc]
        
        # Update statistics for online softmax
        m_ij = max(m_i, rowmax(S_ij))  # Running max
        P_ij = exp(S_ij - m_ij)         # Numerically stable exp
        l_ij = exp(m_i - m_ij) * l_i + rowsum(P_ij)  # Running sum
        
        # Update output with this tile's contribution
        O_i = (diag(l_i * exp(m_i - m_ij)) @ O_i + P_ij @ V_j) / diag(l_ij)
        
        # Update statistics for next iteration
        l_i = l_ij
        m_i = m_ij
    
    O[i*Br:(i+1)*Br, :] = O_i  # Write output tile
```

### Implementation Steps

#### Step 1: Define Kernel Structure (v0.8.1 API)

```rust
// src/kernels/cubecl/kernel.rs

use cubecl::prelude::*;

/// Compile-time configuration passed to kernel
struct TileConfig {
    tile_size: u32,      // Typically 128 (RTX 3090 Ti) or 256 (RTX 5080)
    head_dim: u32,
    num_kv_tiles: u32,
}

#[cube(launch_unchecked)]
fn flash_attention_forward<F: Float>(
    q: &Array<Line<F>>,         // Vectorized Q [batch*heads, seq/4, dim]
    k: &Array<Line<F>>,         // Vectorized K
    v: &Array<Line<F>>,         // Vectorized V
    out: &mut Array<Line<F>>,   // Output
    scale: F,                   // 1/sqrt(head_dim)
    config: Comptime<TileConfig>, // Compile-time config for unrolling
) {
    // Shared memory is 1D only
    let mut q_tile = SharedMemory::<F>::new(config.tile_size * config.head_dim);
    let mut k_tile = SharedMemory::<F>::new(config.tile_size * config.head_dim);
    let mut v_tile = SharedMemory::<F>::new(config.tile_size * config.head_dim);
    
    // Register accumulators
    let mut acc_m = F::neg_inf();     // Running max
    let mut acc_l = F::new(0.0);      // Running sum
    let mut acc_o = Line::splat(F::new(0.0)); // Vectorized output
    
    // ... kernel implementation
}
```

#### Step 2: Implement Tiling Logic (v0.8.1 API)

```rust
#[cube(launch_unchecked)]
fn flash_attention_forward<F: Float>(
    q: &Array<Line<F>>,
    k: &Array<Line<F>>,
    v: &Array<Line<F>>,
    out: &mut Array<Line<F>>,
    scale: F,
    config: Comptime<TileConfig>,
) {
    // Flatten batch × heads on CUBE_POS_X, tile Q on CUBE_POS_Y
    let batch_head = CUBE_POS_X;
    let q_tile_idx = CUBE_POS_Y;
    let tid = UNIT_POS_X;  // Thread within block
    
    // Calculate tile boundaries
    let q_start = q_tile_idx * config.tile_size;
    
    // Allocate shared memory (1D only in CubeCL v0.8.1)
    let mut q_tile = SharedMemory::<F>::new(config.tile_size * config.head_dim);
    let mut k_tile = SharedMemory::<F>::new(config.tile_size * config.head_dim);
    let mut v_tile = SharedMemory::<F>::new(config.tile_size * config.head_dim);
    
    // Load Q tile into shared memory (cooperative loading)
    // Each thread loads multiple elements
    let elems_per_thread = (config.tile_size * config.head_dim) / CUBE_DIM_X;
    for i in 0..elems_per_thread {
        let idx = tid * elems_per_thread + i;
        let global_idx = batch_head * seq_len * head_dim + (q_start + idx / config.head_dim) * config.head_dim + idx % config.head_dim;
        q_tile[idx] = q[global_idx];
    }
    sync_units();  // Barrier (not sync_threads)
    
    // Initialize accumulators in registers
    let mut acc_m = F::neg_inf();     // Running max
    let mut acc_l = F::new(0.0);      // Running sum  
    let mut acc_o = Line::splat(F::new(0.0)); // Vectorized output accumulator
    
    // Process all K/V tiles
    for kv_tile_idx in 0..config.num_kv_tiles {
        let kv_start = kv_tile_idx * config.tile_size;
        
        // Cooperative load K, V tiles
        // ... similar to Q loading ...
        sync_units();
        
        // Compute S = Q @ K^T (use cubek-matmul or manual dots)
        // For manual implementation without cubek-matmul:
        let mut score = F::new(0.0);
        for d in 0..config.head_dim {
            score = score + q_tile[tid * config.head_dim + d] * k_tile[kv_idx * config.head_dim + d];
        }
        score = score * scale;
        
        // Online softmax update
        let local_max = warp_reduce(score, |a, b| F::max(a, b));
        let new_max = F::max(acc_m, local_max);
        
        // Correction factor for previous accumulator
        let correction = F::exp(acc_m - new_max);
        acc_l = acc_l * correction;
        
        // Scale previous output
        acc_o = acc_o * Line::splat(correction);
        
        // Add new contribution
        let exp_score = F::exp(score - new_max);
        acc_l = acc_l + warp_reduce(exp_score, |a, b| a + b);
        
        // Accumulate attention * V
        // ... matrix multiply with v_tile ...
        
        acc_m = new_max;
        sync_units();
    }
    
    // Final normalization and output
    acc_o = acc_o / Line::splat(acc_l);
    // Store to global memory...
}
```

**Key v0.8.1 API Changes:**
- Use `sync_units()` not `sync_threads()`
- Use `warp_reduce(value, |a, b| ...)` for reductions
- Use `F::new(0.0)` not `F::from_f32(0.0)`
- Use `Line::splat(value)` for vectorized constants
- SharedMemory is 1D: access as `smem[row * cols + col]`

#### Step 3: Launch Configuration (v0.8.1 API)

```rust
use cubecl::prelude::*;
use cubecl_cuda::CudaRuntime as Runtime;

pub fn launch_flash_attention<F: Float>(
    q: &Tensor,  // Candle tensor
    k: &Tensor,
    v: &Tensor,
    scale: f64,
) -> Result<Tensor> {
    let (batch, num_heads, seq_len, head_dim) = q.dims4()?;
    let device = q.device();
    
    // Get CubeCL CUDA client
    let client = Runtime::client(&Default::default());
    
    // Determine tile size based on GPU (tune for RTX 5080: 256, RTX 3090 Ti: 128)
    let tile_size = 128;
    let num_q_tiles = (seq_len as u32 + tile_size - 1) / tile_size;
    let num_kv_tiles = num_q_tiles;
    
    // Grid: (batch * num_heads, num_q_tiles, 1)
    let cube_count = CubeCount::Static(
        (batch * num_heads) as u32,
        num_q_tiles,
        1,
    );
    
    // Block: (256, 1, 1) - 256 threads, warp-aligned
    let cube_dim = CubeDim::new(256, 1, 1);
    
    // Convert Candle tensors to CubeCL handles
    // For vectorized loads (Line<F>), set vectorization=4
    let q_len = q.elem_count();
    let (q_bytes, _, _) = candle_to_cubecl_handle(q)?;
    let q_handle = client.create(&q_bytes);
    
    let (k_bytes, _, _) = candle_to_cubecl_handle(k)?;
    let k_handle = client.create(&k_bytes);
    
    let (v_bytes, _, _) = candle_to_cubecl_handle(v)?;
    let v_handle = client.create(&v_bytes);
    
    // Allocate output
    let out_handle = client.empty(q_len * 4);  // f32 bytes
    
    // Config as compile-time constant
    let config = TileConfig {
        tile_size,
        head_dim: head_dim as u32,
        num_kv_tiles,
    };
    
    // Launch kernel
    flash_attention_forward::launch_unchecked::<F, Runtime>(
        &client,
        cube_count,
        cube_dim,
        ArrayArg::from_raw_parts(&q_handle, q_len, 4),  // vectorization=4 for Line<F>
        ArrayArg::from_raw_parts(&k_handle, k.elem_count(), 4),
        ArrayArg::from_raw_parts(&v_handle, v.elem_count(), 4),
        ArrayArg::from_raw_parts(&out_handle, q_len, 4),
        ScalarArg::new(scale as f32),
        Comptime::new(config),
    );
    
    // Read output and convert back to Candle
    let out_bytes = client.read(&out_handle);
    let out_tensor = cubecl_to_candle_tensor(&out_bytes, q.dims(), device)?;
    
    Ok(out_tensor)
}
```

**Profiling:** Set `CUBECL_PROFILE=1` environment variable for kernel timings.

**Important v0.8.1 Notes:**
- Use `launch_unchecked` for performance (skip bounds checks)
- Use `ArrayArg::from_raw_parts()` with vectorization parameter
- Use `Comptime::new(config)` for compile-time constants
- Grid is called `CubeCount`, block is called `CubeDim`

#[cube]
fn online_softmax_update<F: Float>(
    scores: &SharedMemory<F>,
    max_val: &mut F,
    sum_val: &mut F,
    output_acc: &mut [F],
    v_tile: &SharedMemory<F>,
) {
    // Find max score in this tile
    let local_max = row_max(scores);
    let new_max = max(*max_val, local_max);
    
    // Update statistics with numerical stability
    let correction = exp(*max_val - new_max);
    *sum_val = *sum_val * correction;
    
    // Scale previous output
    for i in 0..output_acc.len() {
        output_acc[i] = output_acc[i] * correction;
    }
    
    // Add contribution from this tile
    let exp_scores = exp(scores - new_max);
    *sum_val = *sum_val + row_sum(exp_scores);
    
    // Accumulate attention * V
    matmul_accumulate(exp_scores, v_tile, output_acc);
    
    *max_val = new_max;
}
```

### Numerical Stability

Critical for preventing overflow/underflow in softmax:

```rust
// Instead of: exp(x) / sum(exp(x))
// Use log-sum-exp trick:
max_val = max(x)
exp(x - max_val) / sum(exp(x - max_val))
```

Online softmax maintains running max and sum:
```rust
m_new = max(m_old, max(x_new))
s_new = s_old * exp(m_old - m_new) + sum(exp(x_new - m_new))
```

### Memory Optimization

1. **Shared Memory**: Use for frequently accessed data (tiles)
2. **Register Memory**: Use for accumulator variables
3. **Coalesced Access**: Align memory access patterns
4. **Bank Conflicts**: Avoid shared memory bank conflicts

### Launch Configuration Best Practices

```rust
// Block sizes: 128-256 threads, warp-aligned (32 for NVIDIA)
let cube_dim = CubeDim::new(256, 1, 1);

// Grid: Flatten batch × heads, tile sequence
// - CUBE_POS_X: batch_idx * num_heads + head_idx
// - CUBE_POS_Y: which Q tile (0..num_q_tiles)
let cube_count = CubeCount::Static(
    (batch * num_heads) as u32,
    num_q_tiles,
    1,
);

// Use CubeCount::Static for compile-time optimization
// Use CubeCount::Dynamic for runtime-determined grid sizes
```

## Phase 3: Optimization

After basic implementation works:

1. **Tuning**: Experiment with tile sizes (64, 128, 256)
2. **Occupancy**: Maximize GPU utilization
3. **Memory Patterns**: Optimize access patterns
4. **Mixed Precision**: f16 compute, f32 accumulate
5. **Fusion**: Combine with QKV projection if beneficial

## Testing Strategy

1. **Unit Tests**: Test individual kernel functions
2. **Numerical Equivalence**: Compare with CPU (tolerance 1e-5)
3. **Edge Cases**: 
   - Small sequences (< tile size)
   - Large sequences (>> tile size)
   - GQA (num_kv_heads < num_heads)
   - Different head dimensions
4. **Stress Tests**: Very large tensors, boundary conditions

## Resources

- Flash Attention Paper: https://arxiv.org/abs/2205.14135
- Flash Attention 2: https://arxiv.org/abs/2307.08691
- CubeCL Documentation: https://github.com/tracel-ai/cubecl
- Triton Flash Attention: https://github.com/openai/triton (reference implementation)

## Known Challenges

1. **CubeCL API**: Limited documentation, may need to explore examples
2. **Softmax Stability**: Must handle large values without overflow
3. **Tiling Complexity**: Boundary conditions when seq_len % tile_size != 0
4. **GQA Support**: Repeat K/V heads for different Q heads
5. **Performance Tuning**: Achieving 2-5x speedup requires optimization

## Incremental Approach

Rather than implementing the full tiled algorithm immediately:

1. Start with simple Q·K^T·V without tiling
2. Add softmax with numerical stability
3. Add basic tiling (fixed tile size)
4. Add online softmax with running statistics
5. Add shared memory optimization
6. Add launch configuration tuning
7. Add GQA support
8. Add mixed precision support

Each step should maintain passing tests before moving to the next.
