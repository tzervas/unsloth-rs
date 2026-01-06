# Flash Attention CubeCL Kernel Implementation Guide

## Current Status
Phase 1 (Foundation) is complete. The infrastructure for Flash Attention is in place with:
- Module structure and integration points
- Device detection and dispatch logic
- Fallback implementation using Candle operations
- Comprehensive test suite
- VRAM estimation utilities

## Phase 2: CubeCL Kernel Implementation

### Overview
Implementing the actual CubeCL GPU kernel requires understanding of:
1. CubeCL kernel syntax and macros
2. Flash Attention tiled algorithm
3. GPU memory hierarchy (global, shared, registers)
4. Softmax numerical stability
5. Candle tensor interoperability

### CubeCL Kernel Basics

CubeCL uses Rust macros to define GPU kernels that can compile to multiple backends (CUDA, ROCm, Vulkan).

```rust
use cubecl::prelude::*;

#[cube(launch)]
fn simple_kernel<F: Float>(input: &Tensor<F>, output: &mut Tensor<F>) {
    let idx = ABSOLUTE_POS;  // Thread index
    if idx < input.len() {
        output[idx] = input[idx] * F::from_f32(2.0);
    }
}
```

Key CubeCL concepts:
- `#[cube(launch)]` - Marks function as GPU kernel
- `ABSOLUTE_POS` - Global thread index
- `Tensor<F>` - Generic tensor type
- `Float` trait - Works with f32, f16, bf16

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

#### Step 1: Define Kernel Structure

```rust
// src/kernels/attention_cubecl.rs

use cubecl::prelude::*;

#[cube(launch)]
fn flash_attention_forward<F: Float>(
    q: &Tensor<F>,              // [batch, num_heads, seq_len, head_dim]
    k: &Tensor<F>,              // [batch, num_kv_heads, seq_len, head_dim]
    v: &Tensor<F>,              // [batch, num_kv_heads, seq_len, head_dim]
    output: &mut Tensor<F>,     // [batch, num_heads, seq_len, head_dim]
    scale: F,                   // 1/sqrt(head_dim)
    config: FlashAttentionConfig,
) {
    // Kernel implementation
}

struct FlashAttentionConfig {
    tile_size: u32,      // Typically 128 or 256
    seq_len: u32,
    head_dim: u32,
    num_heads: u32,
    num_kv_heads: u32,
}
```

#### Step 2: Implement Tiling Logic

```rust
#[cube(launch)]
fn flash_attention_forward<F: Float>(...) {
    let batch_idx = CUBE_POS_X;
    let head_idx = CUBE_POS_Y;
    let tile_idx = CUBE_POS_Z;
    
    // Calculate which tile of Q this thread block processes
    let q_start = tile_idx * config.tile_size;
    let q_end = min(q_start + config.tile_size, config.seq_len);
    
    // Allocate shared memory for tiles
    let mut q_tile = SharedMemory::<F>::new(config.tile_size, config.head_dim);
    let mut k_tile = SharedMemory::<F>::new(config.tile_size, config.head_dim);
    let mut v_tile = SharedMemory::<F>::new(config.tile_size, config.head_dim);
    let mut scores_tile = SharedMemory::<F>::new(config.tile_size, config.tile_size);
    
    // Load Q tile into shared memory
    load_tile(&mut q_tile, q, batch_idx, head_idx, q_start, q_end);
    
    // Initialize output accumulator and statistics
    let mut output_acc = register_array::<F>(config.head_dim);
    let mut max_val = F::from_f32(-INFINITY);
    let mut sum_val = F::from_f32(0.0);
    
    // Process all K/V tiles
    for k_tile_idx in 0..num_k_tiles {
        let k_start = k_tile_idx * config.tile_size;
        let k_end = min(k_start + config.tile_size, config.seq_len);
        
        // Load K, V tiles
        load_tile(&mut k_tile, k, batch_idx, head_idx, k_start, k_end);
        load_tile(&mut v_tile, v, batch_idx, head_idx, k_start, k_end);
        
        sync_threads();
        
        // Compute attention scores for this tile
        compute_scores(&q_tile, &k_tile, &mut scores_tile, scale);
        
        // Online softmax update
        online_softmax_update(
            &scores_tile,
            &mut max_val,
            &mut sum_val,
            &mut output_acc,
            &v_tile,
        );
        
        sync_threads();
    }
    
    // Final normalization
    normalize_output(&mut output_acc, sum_val);
    
    // Write output
    write_output(output, batch_idx, head_idx, q_start, &output_acc);
}
```

#### Step 3: Implement Helper Functions

```rust
#[cube]
fn compute_scores<F: Float>(
    q_tile: &SharedMemory<F>,
    k_tile: &SharedMemory<F>,
    scores: &mut SharedMemory<F>,
    scale: F,
) {
    let tid = THREAD_POS;
    
    // Each thread computes one element of scores matrix
    let i = tid / k_tile.rows();
    let j = tid % k_tile.rows();
    
    if i < q_tile.rows() && j < k_tile.rows() {
        let mut sum = F::from_f32(0.0);
        for k in 0..q_tile.cols() {
            sum = sum + q_tile[i, k] * k_tile[j, k];
        }
        scores[i, j] = sum * scale;
    }
}

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

### Launch Configuration

```rust
pub fn launch_flash_attention<F: Float>(
    q: &Tensor<F>,
    k: &Tensor<F>,
    v: &Tensor<F>,
    output: &mut Tensor<F>,
    scale: F,
) -> Result<()> {
    let (batch, num_heads, seq_len, head_dim) = q.dims4()?;
    
    // Determine tile size based on shared memory limits
    let tile_size = 128;  // Tunable parameter
    let num_tiles = (seq_len + tile_size - 1) / tile_size;
    
    // Grid dimensions
    let grid_dim = (batch, num_heads, num_tiles);
    
    // Block dimensions (threads per block)
    let block_dim = (tile_size, 1, 1);
    
    // Launch kernel
    flash_attention_forward::launch(
        grid_dim,
        block_dim,
        q, k, v, output, scale, config
    )?;
    
    Ok(())
}
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
