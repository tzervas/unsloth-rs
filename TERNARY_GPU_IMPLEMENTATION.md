# Ternary GPU Kernel Implementation Guide

This document provides implementation context for Phases 2-5 of the ternary bitsliced operations in unsloth-rs.

## Prerequisites

Phase 1 (CPU implementation) is complete with:
- `TernaryTensor`, `TernaryPlanes`, `SparsityMetadata` types
- FP→ternary quantization with TWN-style calibration
- CPU popcount-based matmul reference implementation
- `TernaryLinear` drop-in layer (implements `candle_core::Module`)
- 24 passing unit tests

## Phase 2: GPU Ternary Matmul Kernel

### 2.1 CubeCL Popcount Support

CubeCL 0.8.1 provides native `count_ones()` intrinsic:

```rust
// Maps to hardware popcount:
// - CUDA: __popc(u32) / __popcll(u64)
// - Metal: popcount()
// - SPIRV: bit_count()
impl<P: CountOnes> Line<P> {
    pub fn count_ones(self) -> Line<u32>
}
```

### 2.2 Kernel Design

```rust
#[cube(launch_unchecked)]
fn ternary_matmul_kernel<F: Float>(
    input: &Array<F>,           // [batch, in_features]
    w_plus: &Array<u32>,        // [out_features, k_words]
    w_minus: &Array<u32>,       // [out_features, k_words]
    scales: &Array<F>,          // [out_features]
    output: &mut Array<F>,      // [batch, out_features]
    #[comptime] config: TernaryMatmulConfig,
) {
    let batch_idx = CUBE_POS_X;
    let out_idx = CUBE_POS_Y * config.tile_n + UNIT_POS_X;
    
    // Shared memory for input tile
    let mut input_tile = SharedMemory::<u32>::new(config.tile_k);
    
    // Popcount-based dot product
    let mut pos_sum = 0u32;
    let mut neg_sum = 0u32;
    
    for k in 0..config.k_words {
        let wp = w_plus[out_idx * config.k_words + k];
        let wm = w_minus[out_idx * config.k_words + k];
        
        // Plane skipping optimization
        if (wp | wm) == 0 { continue; }
        
        // Native GPU popcount
        pos_sum += (wp & input_tile[k]).count_ones();
        neg_sum += (wm & input_tile[k]).count_ones();
    }
    
    let dot = (pos_sum as i32) - (neg_sum as i32);
    output[batch_idx * config.n + out_idx] = F::cast_from(dot) * scales[out_idx];
}
```

### 2.3 Implementation Tasks

| Task | File | Description |
|------|------|-------------|
| 2.1 | `cubecl/interop.rs` | Add `u32_planes_to_cubecl_handle()` |
| 2.2 | `ternary/matmul_cubecl.rs` | Basic popcount kernel (no tiling) |
| 2.3 | `ternary/matmul_cubecl.rs` | Tiled kernel with shared memory |
| 2.4 | `ternary/matmul_cubecl.rs` | Vectorized `Line<u32>` loads |
| 2.5 | `ternary/matmul_cubecl.rs` | Plane skipping with SparsityMetadata |
| 2.6 | `ternary/matmul.rs` | Wire GPU dispatch, validate vs CPU |

### 2.4 Testing Strategy

1. **Numerical equivalence**: Compare GPU output to `ternary_matmul_cpu()` (tolerance 1e-5)
2. **Shape tests**: [batch, seq, features] → [batch, seq, out_features]
3. **Sparsity tests**: Verify plane skipping for >90% sparse tensors
4. **Performance**: `CUBECL_PROFILE=1` for kernel timing

## Phase 3: Ternary Attention

### 3.1 Q·K^T Scoring via Popcount

```rust
// Score = popcount-based dot of Q row with K column
// Approximate softmax via online max/count normalization
score_ij = popcount_dot(Q_i, K_j) * scale_q * scale_k
```

### 3.2 Integration Points

- Extend `FusedAttention` with ternary path
- Hybrid dispatch: ternary if sparsity > threshold, else FP
- Causal masking: Zero out +plane/-plane for masked positions

### 3.3 Implementation Tasks

| Task | File | Description |
|------|------|-------------|
| 3.1 | `ternary/attention.rs` | Q·K^T ternary scoring kernel |
| 3.2 | `ternary/attention.rs` | Online softmax with counts |
| 3.3 | `ternary/attention.rs` | Causal masking via plane zeroing |
| 3.4 | `attention.rs` | Hybrid FP/ternary dispatch |
| 3.5 | `attention_cubecl.rs` | Integration with flash attention |

## Phase 4: Advanced Sparsity

### 4.1 Sparsity Metadata

Already implemented in `SparsityMetadata`:
- 64-bit activity bitmaps per 2048-dim chunk
- `is_chunk_active(chunk_idx)` for runtime checks
- `chunk_sparsity()` for monitoring

### 4.2 Dynamic Plane Skipping

```rust
// Comptime check in kernel
if #[comptime] config.enable_plane_skipping {
    if !sparsity_meta.is_chunk_active(chunk_idx) {
        continue;  // Skip entire chunk
    }
}
```

### 4.3 In-Place Edits (Lens API)

```rust
impl TernaryTensor {
    /// Modify a single dimension in O(1) via bit manipulation
    pub fn modify_dim(&mut self, row: usize, col: usize, new_val: i8) {
        let word_idx = col / 32;
        let bit_idx = col % 32;
        let mask = 1u32 << bit_idx;
        
        // Clear both planes
        self.plus_plane[row * self.k_words + word_idx] &= !mask;
        self.minus_plane[row * self.k_words + word_idx] &= !mask;
        
        // Set new value
        match new_val {
            1 => self.plus_plane[row * self.k_words + word_idx] |= mask,
            -1 => self.minus_plane[row * self.k_words + word_idx] |= mask,
            0 => {},
            _ => panic!("invalid ternary value"),
        }
    }
}
```

## Phase 5: Integration & Validation

### 5.1 End-to-End Quantization

```rust
// Quantize entire model
pub fn quantize_model(model: &dyn Module, config: &TernaryConfig) -> Result<TernaryModel> {
    // Walk model layers, quantize Linear → TernaryLinear
    // Preserve non-linear layers (LayerNorm, etc.)
}
```

### 5.2 Benchmarks

| Metric | Target | Measurement |
|--------|--------|-------------|
| Matmul speedup | ≥5x vs FP16 | criterion benchmark |
| Attention speedup | ≥4x vs Flash Attn | tokens/sec |
| Weight memory | ≥10x reduction | VRAM profiling |
| Perplexity | <2% degradation | WikiText-103 |

### 5.3 Hardware Targets

| GPU | Tile Size | Block Size | Notes |
|-----|-----------|------------|-------|
| RTX 5080 | 128-256 | 256 | Primary dev target |
| RTX 3090 Ti | 128 | 256 | Validation target |
| A100/H100 | 256+ | 256 | Datacenter |

## File Structure

```
src/kernels/ternary/
├── mod.rs              # Module exports
├── config.rs           # TernaryConfig (✅ done)
├── types.rs            # TernaryTensor, TernaryPlanes (✅ done)
├── quantize.rs         # FP→ternary quantization (✅ done)
├── matmul.rs           # CPU matmul + GPU dispatch (✅ CPU done)
├── matmul_cubecl.rs    # CubeCL GPU kernel (Phase 2)
├── linear.rs           # TernaryLinear layer (✅ done)
├── attention.rs        # Ternary attention (Phase 3)
└── attention_cubecl.rs # CubeCL attention kernel (Phase 3)
```

## Dependencies

```toml
[dependencies]
cubecl = { version = "0.8.1", features = ["cuda"] }
cubecl-cuda = { version = "0.8.1", optional = true }
candle-core = "0.9.1"
candle-nn = "0.9.1"
```

## References

- [TWN Paper (Li et al., 2016)](https://arxiv.org/abs/1605.04711) - Ternary Weight Networks
- [CubeCL Documentation](https://docs.rs/cubecl/0.8.1) - GPU kernel framework
- [math.md](./math.md) - Mathematical proofs (entropy, SNR bounds)
- [details.md](./details.md) - Project specification
- [roadmap.md](./roadmap.md) - 6-phase implementation plan
