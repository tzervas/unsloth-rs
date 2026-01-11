# Project Specification: Ternary Bitsliced Enhancements for unsloth-rs

This document defines a rigorous, phased plan for integrating single-trit bitsliced balanced ternary operations (with +plane/-plane emulation) into unsloth-rs using CubeCL kernels. The design prioritizes mathematically proven components (popcount-based matmul from ternary NN literature, bounded VSA superposition) while deferring speculative extensions (multi-trit depth, extreme nesting) until basics validate.

**Context and Rationale** (Multiple Angles):
- **Why Ternary Bitsliced?** Proven in TWN (2016) and follow-ups: Exact popcount dots for sparse weights, no carry overhead, 10–30x memory/speedup on pruned LLMs. Sparsity (95–99% zeros) amplifies via plane skipping.
- **Nuances/Implications**: Minor accuracy drop (~1–3% perplexity rise) recoverable via calibration/fine-tune. Fails on dense data (fallback needed). Edge: Non-power-of-2 dims require padding.
- **Objectivity**: No "infinite compression"—bounded by entropy (~1.58 bits/dim) and SNR (~sqrt(D/k) items superposed).
- **Optimization Focus**: Progressive—start minimal viable, measure, iterate (GPU profiling key).

**Scope Boundaries**:
- Core: Weights ternary; activations FP16/32 (hybrid like real ternary NNs).
- Exclude initially: Full holographic nesting (capacity limits prove marginal gains).

---

## Requirements

### Functional Requirements
1. **Ternary Quantization**: Convert FP weights to +plane/-plane bits + per-channel f32 scale (mutual exclusion enforced).
2. **Core Ops**: Bitsliced matmul, attention scoring (popcount dots), in-place edits (bit flips).
3. **Sparsity Handling**: Plane/dim skipping for >90% zeros.
4. **Hybrid Execution**: Seamless fallback to FP kernels.
5. **Configurability**: Per-layer sparsity threshold, scale calibration.
6. **Algebraic Lens**: Basic binding (XOR-like) and bundling (OR/popcount).

### Non-Functional Requirements
1. **Performance**: ≥5x speedup on matmul/attention vs FP16 baseline on sparse pruned models (seq=2048+).
2. **Memory**: ≥10x reduction for weights (target 20–30x with sparsity).
3. **Accuracy**: <2% perplexity degradation on WikiText/C4 vs FP16 (post-calibration).
4. **Portability**: CubeCL backends (CUDA priority, ROCm/Vulkan tested).
5. **Numerical Stability**: No NaN/Inf; tolerances match Flash Attention (1e-5 FP32).
6. **Usability**: Drop-in replacement for Linear/Attention layers.

**Edge Cases**:
- Dense layers: Auto-fallback.
- Batch>1, variable seq: Padding/masking handled.
- Precision mismatch: Activation requant if needed.

---

## Deliverables

### Overall
- `unsloth-ternary` crate (or module in unsloth-rs).
- Quantization pipeline script (calibrate + convert pretrained models).
- Benchmarks suite (speed, memory, perplexity).
- Documentation: API, math rationale, limitations.

### Per-Phase (Detailed Below)

---

## Success Criteria

**Quantitative**:
- **Speed**: Matmul/attention kernel >5x faster than Candle FP16 on pruned Llama-7B (measured via criterion, seq=2048, batch=4).
- **Memory**: Weights VRAM <10% of FP16 equivalent (e.g., 7B <2GB).
- **Accuracy**: Zero-shot perplexity on WikiText-103 <1.05x FP16 baseline (post optional fine-tune).
- **Capacity (if holographic)**: >90% retrieval accuracy for k=D/20 bundled items.

**Qualitative**:
- Numerical equivalence: All-reduce error <1e-4 vs FP reference on small tensors.
- Stability: No crashes/NaN across 1000 random inputs.
- Usability: <5 lines to quantize/load ternary model.

**Failure Thresholds**: If Phase 1 <3x speedup or >5% acc drop → pivot to hybrid/packed.

---

## Progressive Phases: Expected Characteristics and Optimization Strategies

Phases build incrementally: Validate math (popcount exactness, SNR bounds) via tests/benchmarks each step.

### Phase 1: Ternary Matmul Core (2–3 Weeks)
**Expected Characteristics**:
- Exact dots via popcount (math: dot = pos_matches - neg_matches × scale).
- Baseline speedup 5–15x on sparse (popcount warp-native).
- Memory: 16–32x weight reduction base.

**Optimization Strategies**:
- **Tiling**: Shared mem for A/B tiles (64–128 like cubek-matmul).
- **Vectorization**: Line<u32> for 4x u32 loads.
- **Profile-Driven**: CUBECL_PROFILE=1 → tune block dims (128–512 threads), occupancy >60%.
- **Sparsity Early**: Zero-plane branch (if all-zero bitmap, skip).
- **Measure**: Kernel time vs Candle matmul; target <10ms for 4096x4096 sparse.

**Success**: >5x speedup on pruned matrix; exact vs CPU ref.

### Phase 2: Ternary Attention Integration (3–4 Weeks)
**Expected Characteristics**:
- Scoring via popcount (approximate softmax stable via online max/count).
- Full forward pass hybrid (ternary proj, FP attn combine).
- Speedup 3–10x end-to-end attention.

**Optimization Strategies**:
- **Flash-Like Tiling**: Load Q/K tiles, popcount per row.
- **Masking**: Causal via zero-out planes.
- **Numerical Tuning**: Scale calibration (min-max or abs-max per channel).
- **Hybrid Probe**: If sparsity <80%, fallback FP scoring.
- **Measure**: Throughput (tokens/s) vs baseline Flash; VRAM during inference.

**Success**: <1.5% perplexity rise on small model; >4x faster attention.

### Phase 3: Advanced Sparsity and Edits (2 Weeks)
**Expected Characteristics**:
- Plane skipping → 50–100x effective compression on 99% sparse.
- In-place edits: O(1) bit flips.

**Optimization Strategies**:
- **Metadata**: 64-bit bitmap per 2048 dims.
- **Dynamic Skip**: Runtime check (or comptime if fixed).
- **Lens API**: modify_dim(idx, new_val) → mask & XOR planes.
- **Measure**: VRAM static load; edit latency.

**Success**: >20x weight memory on pruned 7B; edits <1μs per param.

### Phase 4: Basic Holographic Extensions (3 Weeks, Optional Post-Validation)
**Expected Characteristics**:
- Binding/bundling via gates (XOR/OR).
- Capacity ~D/20 items (per SNR proof).

**Optimization Strategies**:
- **Cleanup**: Iterative threshold/popcount.
- **Limit k**: Config cap based on D.
- **Measure**: Retrieval acc vs k bundled.

**Success**: >85% acc for k=500 in D=10k.

### Phase 5: Full Model Validation and Tuning (Ongoing)
**Expected Characteristics**:
- End-to-end ternary model (e.g., Llama-7B variant).
- Fine-tune recovery if needed.

**Optimization Strategies**:
- **Calibration Ablation**: Test abs-max vs percentile scaling.
- **Mixed Precision**: Ternary low layers, FP high.
- **Benchmark Suite**: Automate perplexity/speed across configs.

**Success**: Deployable model with >5x overall inference speedup, <2% acc hit.

This plan is realistic, measurable, and rooted in proven math. Start Phase 1—it's the proof point.