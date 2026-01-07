# Phased Roadmap: Integrating Embeddenator Ternary Kernels into CubeCL for unsloth-rs

This roadmap consolidates our discussions into a clean, prioritized implementation plan for enhancing unsloth-rs with ternary bitsliced/holographic capabilities drawn from embeddenator branches (e.g., `cuda-kernels`, `sparse-attention`, `ternary-ops`, `holographic-experiments`).

**Core Philosophy** (Objective Reality Check):
- Prioritize proven, mathematically sound components first: Single-trit {-1,0,+1} bitsliced ops (backed by TWN literature, popcount efficiency, no carry pitfalls).
- Defer speculative/multi-trit depth until basics validate (avoids snake oil risk).
- Focus on GPU testing feasibility: All kernels target CubeCL portability (CUDA/ROCm/Vulkan).
- Density/accuracy trade-offs: Aim for 10–30x memory reduction + 5–20x speedup on sparse matmul/attention, with <2% accuracy drop (calibrated like 4-bit quant).

**High-Level Priorities**:
1. **Immediate Wins**: Single-trit bitsliced matmul/attention (popcount-native, warp-efficient).
2. **Sparsity Enhancements**: Plane/dim skipping for 95–99% zeros.
3. **Algebraic Lens**: In-place edits + basic binding.
4. **Extensions**: Moderate nesting/holographic, packed fallback for higher precision.
5. **Polish**: Configurable defaults, testing, integration.

**Assumptions**:
- Start from existing unsloth-rs CubeCL Flash Attention skeleton.
- Port embeddenator CUDA references as CubeCL prototypes.
- Testing: Tonight/next session on real GPU (A100/RTX40xx recommended for profiling).

---

## Phase 1: Foundation – Single-Trit Bitsliced Ternary Matmul (1–2 Weeks, Highest Priority)

**Goal**: Replace FP16 matmul in attention/FFN with ternary popcount version. Proven fastest (literature: 10–30x vs dense FP on sparse weights).

**Why First**:
- Math proof: Dot = popcount(pos matches) - popcount(neg matches) × scale.
- No carry, warp-native popcount (CubeCL has reductions).
- Immediate speedup/memory win on pruned models.

**Steps**:
1. **Port embeddenator ternary matmul** (from `ternary-ops` or `cuda-kernels` branches):
   - Separate +plane/-plane (u32 arrays).
   - Scale per channel (f32).
2. **CubeCL Kernel**:
   ```rust
   #[cube(launch_unchecked)]
   fn ternary_matmul_bitsliced(
       a_plus: &Array<u32>, a_minus: &Array<u32>, a_scale: f32,
       b_plus: &Array<u32>, b_minus: &Array<u32>, b_scale: f32,
       out: &mut Array<f32>,
       M: u32, N: u32, K_words: u32,  // K_words = K_dim / 32
   ) {
       let row = CUBE_POS_Y;
       let col = UNIT_POS_X * 32;  // 32 cols per thread
       
       let mut acc = F::new(0.0);
       for k in 0..K_words {
           let a_p = a_plus[row * K_words + k];
           let a_m = a_minus[row * K_words + k];
           let b_p = b_plus[k * (N/32) + UNIT_POS_X];
           let b_m = b_minus[k * (N/32) + UNIT_POS_X];
           
           // Popcount matches (warp_reduce for full)
           let pos = __popc(a_p & b_p) + __popc(a_m & b_m);
           let neg = __popc(a_p & b_m) + __popc(a_m & b_p);
           acc += F::from_i32(pos as i32 - neg as i32);
       }
       out[row * N + col..col+32] = acc * a_scale * b_scale;  // Vectorized store
   }
   ```
3. **Integration**:
   - Quantize Candle weights → +plane/-plane + scale (calibration like TWN).
   - Hybrid: FP activations, ternary weights.
4. **Testing**:
   - Small matrices: Verify vs CPU reference.
   - Profile: CUBECL_PROFILE=1 → expect >10x vs FP16 on sparse.

**Deliverable**: TernaryLinear/TernaryMatmul drop-in for unsloth-rs layers.

**Implications/Nuances**: Minor accuracy drop recoverable via short fine-tune; edge case dense weights → fallback FP.

---

## Phase 2: Ternary Flash Attention Integration (2–3 Weeks)

**Goal**: Extend Phase 1 to attention (QKV ternary projection, scoring via popcount).

**Steps**:
1. **Port embeddenator sparse-attention kernels**:
   - Ternary Q·K^T → popcount scores.
   - Threshold for top-k sparse (embeddenator has prototypes).
2. **CubeCL Enhancements**:
   - Use existing Flash skeleton.
   - Online popcount max/sum for stable "softmax" (approximate via counts).
   - Shared mem for tiles; skip zero planes.
3. **Hybrid Scoring**:
   - Ternary for speed, FP fallback for precision if needed.
4. **Testing**:
   - Numerical eq vs Candle Flash (tol 1e-3).
   - Seq=2048–4096 benchmarks.

**Priority Rationale**: Attention dominates LLM cost → biggest win.

---

## Phase 3: Sparsity and Plane Skipping (1–2 Weeks, Parallel to Phase 2)

**Goal**: Exploit 95–99% zeros (implicit in planes).

**Steps**:
1. **Metadata**: Bitmap per chunk (which planes/dims active).
2. **Kernel Skip**:
   - Comptime or runtime branch: If plane zero, skip load/compute.
3. **Storage**: CSR-like for ultra-sparse (embeddenator sparse.rs).
4. **Testing**: Pruned Llama weights → measure VRAM/speed.

**Nuance**: Overhead if sparsity <90%; auto-detect threshold.

---

## Phase 4: Algebraic Lens and Basic Holographic Binding (2–3 Weeks)

**Goal**: Enable embeddenator-style in-place edits + composition.

**Steps**:
1. **Lens Edits**:
   - Modify scalar: Flip bits in +plane/-plane (1–2 u32 touches).
2. **Binding**:
   - Element-wise: +out = +a XOR -b, etc. (gate tricks).
   - Bundle: OR or popcount accumulate.
3. **Moderate Nesting**:
   - 3-level bundle (27^3 slots).
   - Precomputed role vectors.
4. **Testing**: Key-value store demo (superpose 500 items, retrieve >90% acc).

**Math Note**: Capacity proof limits to ~D/10 items.

---

## Phase 5: Configurable Depth and Packed Fallback (2–4 Weeks)

**Goal**: Optional higher precision.

**Steps**:
1. **Packed Mode**: Holistic tryte scalars (ripple/lookup add).
2. **Config Per-Layer**: Trit depth 1–6, auto-tune via calibration.
3. **Hybrid Kernel Dispatch**: Bitsliced for matmul, packed for edits.
4. **Testing**: Ablation on accuracy vs density.

**Fallback Rationale**: For non-sparse layers.

---

## Phase 6: Polish, Ecosystem, and Validation (Ongoing)

**Steps**:
1. **Crate Structure**: `unsloth-ternary` with TernaryTensor trait.
2. **Full Model Quant**: Pipeline for Llama/Mistral.
3. **GPU Testing Plan** (Tonight/Soon):
   - Setup: CubeCL 0.8.1, CUDA device.
   - Benchmarks: Seq lens 512–4096, batch 1–8.
   - Profile memory (client queries), occupancy.
4. **Validation**: Perplexity on WikiText, zero-shot tasks.

**Overall Timeline**: 2–3 months to MVP ternary unsloth-rs.

**Risks/Implications**:
- Accuracy: Monitor degradation; fine-tune recovery essential.
- Portability: Vulkan slower → prioritize CUDA.
- Edge: Non-sparse models → minimal gain.

This roadmap is realistic, proven where possible, and builds incrementally. Knock out Phase 1 matmul first—it's the foundation that will "chooches" immediately. Let's get it slapped together.