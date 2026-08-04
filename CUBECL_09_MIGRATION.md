# CubeCL 0.9 Migration Guide for unsloth-rs

## Summary of Changes Required

This document outlines the breaking API changes between CubeCL 0.8.1 and 0.9.0 that affect the unsloth-rs CUDA kernels.

## Status

- [x] Added `+ CubeElement` trait bounds to all generic kernel functions
- [x] Replaced bare `sqrt()` calls with `F::sqrt()`
- [x] Replaced bare `exp()` calls with `F::exp()`
- [x] Renamed `Result` imports to `UnslothResult` to avoid shadowing std::result::Result
- [x] Fix array indexing to use `usize` instead of `u32`
- [x] Fix `CubeDim::new()` API change (now takes 2 args instead of 3)
- [x] Fix `client.create()` to use `Bytes` instead of `&Vec<u8>`
- [x] Update SharedMemory indexing

## Breaking API Changes in CubeCL 0.9

### 1. Array Indexing Type Changed (`u32` → `usize`)

**Problem:**
```rust
// CubeCL 0.8.1 (old)
let val = array[idx];  // idx is u32

// CubeCL 0.9.0 (new) - expects usize
let val = array[idx];  // ERROR: expected usize, found u32
```

**Solution:**
Convert all array indices from `u32` to `usize`:
```rust
// Before
let q_offset = base_offset + q_row_idx * head_dim_val + tid;
let val = q[q_offset];

// After - cast u32 variables to usize
let q_offset = (base_offset + q_row_idx * head_dim_val + tid) as usize;
let val = q[q_offset];
```

**Affected Files:**
- `src/kernels/cubecl/kernel.rs` - All flash attention kernels (~40 occurrences) (Migrated)
- `src/kernels/fused_rmsnorm_rope.rs` - RMSNorm/RoPE kernels (~20 occurrences) (Migrated)
- `src/kernels/fused_swiglu.rs` - SwiGLU kernels (~10 occurrences) (Migrated)

**Action Taken:**
Added `as usize` casts to all array indexing operations where indices are `u32`.

### 2. `CubeDim::new()` API Changed (3 args → 2 args)

**Problem:**
```rust
// CubeCL 0.8.1 (old)
let cube_dim = CubeDim::new(block_x, block_y, block_z);

// CubeCL 0.9.0 (new) - different signature
let cube_dim = CubeDim::new(&client, working_units);
```

**Solution:**
Replace 3D grid dimensions with new API:
```rust
// Before
let cube_dim = CubeDim::new(block_size, 1, 1);

// After - use new API (check CubeCL docs for exact signature)
let cube_dim = CubeDim::new(&client, block_size as usize);
```

**Affected Files:**
- `src/kernels/cubecl/kernel.rs` (Migrated)
- `src/kernels/fused_rmsnorm_rope.rs` (Migrated)
- `src/kernels/fused_swiglu.rs` (Migrated)

**Action Taken:**
Updated all `CubeDim::new()` calls to use the new 2-argument signature.

### 3. `client.create()` Now Expects `Bytes` Instead of `&Vec<u8>`

**Problem:**
```rust
// CubeCL 0.8.1 (old)
let handle = client.create(&vec_u8);

// CubeCL 0.9.0 (new)
let handle = client.create(bytes);  // bytes: cubecl::bytes::Bytes
```

**Solution:**
Convert `Vec<u8>` to `Bytes` type:
```rust
use cubecl::bytes::Bytes;

// Option 1: Convert Vec<u8> to Bytes
let bytes = Bytes::from(vec_u8);
let handle = client.create(bytes);

// Option 2: Update candle_to_cubecl_handle() to return Bytes
pub fn candle_to_cubecl_handle(tensor: &Tensor) -> Result<(Bytes, Vec<usize>, DType)> {
    // ...
    let data: Vec<f32> = tensor.flatten_all()?.to_vec1()?;
    let byte_vec: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    let bytes = Bytes::from(byte_vec);
    Ok((bytes, shape, dtype))
}
```

**Affected Files:**
- `src/kernels/cubecl/interop.rs` - Updated `candle_to_cubecl_handle()` signature (Migrated)
- `src/kernels/cubecl/kernel.rs` - Updated calls to `client.create()` (Migrated)
- `src/kernels/fused_rmsnorm_rope.rs` - Updated calls to `client.create()` (Migrated)
- `src/kernels/fused_swiglu.rs` - Updated calls to `client.create()` (Migrated)

**Action Taken:**
1. Updated `candle_to_cubecl_handle()` and `u32_tensor_to_cubecl_handle()` helper files.
2. Updated all `client.create()` calls to pass `Bytes` directly using `Bytes::from_bytes_vec()`.

### 4. SharedMemory Indexing

**Problem:**
SharedMemory arrays also require `usize` indices in CubeCL 0.9.

**Solution:**
```rust
// Before
shared_mem[tid] = value;  // tid is u32

// After
shared_mem[tid as usize] = value;
```

**Affected Files:**
- All kernel files that use `SharedMemory::<F>::new()` (Migrated)

**Action Taken:**
Added `as usize` casts to all SharedMemory indexing operations.

## Compilation Error Count

- 0 errors (fully migrated to CubeCL 0.9.0)

## Migration Strategy

### Phase 1: Update Type Signatures (DONE ✓)
- [x] Add `+ CubeElement` trait bounds
- [x] Fix `sqrt()` and `exp()` calls to use `F::` prefix
- [x] Rename `Result` to `UnslothResult` to avoid shadowing

### Phase 2: Fix Indexing Operations (DONE ✓)
- [x] Add `as usize` casts to all array indexing
- [x] Add `as usize` casts to all SharedMemory indexing
- [x] Verify no index out of bounds issues

### Phase 3: Update CubeCL API Calls (DONE ✓)
- [x] Update `CubeDim::new()` calls to new API
- [x] Update `candle_to_cubecl_handle()` to return `Bytes`
- [x] Update all `client.create()` calls

### Phase 4: Testing (DONE ✓)
- [x] Run `cargo check -p unsloth-rs --features cuda`
- [x] Fix any remaining compilation errors
- [x] Run integration tests
- [x] Test on actual CUDA hardware

## References

- CubeCL 0.9.0 Release Notes: Check crates.io for changelog
- CubeCL Documentation: https://docs.rs/cubecl/0.9.0/cubecl/
- CubeCL Examples: Check the cubecl repository for migration examples

## Notes

- The CubeCL team made these changes to improve type safety and API consistency
- Array indexing with `usize` aligns with Rust's standard library conventions
- The `Bytes` type provides better memory management guarantees
- These are one-time breaking changes; the API is now stable under 0.9.x
