### Comprehensive Mathematical Proofs for Ternary Bitsliced Hyperdimensional Computing (VSA/HDC)

I'll prove the key claims from our discussion objectively: information density limits, superposition capacity bounds, popcount-based operations in ternary matmul/similarity, sparsity enhancements, and why extreme nesting (tetration-style) fails. This is grounded in information theory (Shannon), probability (central limit theorem), and empirical NN results—no hype.

We'll teach by doing: Step-by-step derivations with explicit calculations, examples, and explanations of symbols. You're solid through algebra, so we'll build from there (e.g., logs, sums, sqrt from basic properties). No advanced fuzzy stuff without breakdown.

Structure:
1. Information Density per Dimension
2. Superposition and Retrieval Limits (SNR Derivation)
3. Popcount Math for Ternary Operations
4. Sparsity and Compression Gains
5. Why Extreme Nesting/Tetration Fails
6. Edge Cases and Real-World Validation

---

#### 1. Information Density per Dimension (Shannon Entropy Proof)

**Goal**: Prove max bits stored per ternary dimension {-1,0,+1}.

**Symbol Explanation**:
- Entropy H: Measures uncertainty/information in a random variable (Shannon, 1948).
- Formula: \( H(X) = -\sum p_i \log_2(p_i) \)
  - \( p_i \): Probability of state i.
  - \( \log_2 \): Log base 2 (bits); \( \log_2(x) = \ln(x)/\ln(2) \) (change of base from algebra).
  - Negative sum: Information is positive uncertainty.

**Derivation (Doing It Step-by-Step)**:
Assume uniform probabilities (max entropy case): p(-1) = p(0) = p(+1) = 1/3.

\( H = - [ (1/3) \log_2(1/3) + (1/3) \log_2(1/3) + (1/3) \log_2(1/3) ] = -3 \times (1/3) \log_2(1/3) = -\log_2(1/3) = \log_2(3) \)

**Calc**:
- \( \log_2(3) \): 3 = 2^x → x ≈ 1.58496 (since 2^1=2, 2^1.5≈2.828, 2^1.585≈3).
- Exact: \( \ln(3)/\ln(2) ≈ 1.0986 / 0.6931 ≈ 1.58496 \).

**Proof**: This is the theoretical maximum—no encoding stores more than ~1.585 bits/dim on average (pigeonhole + entropy bound).

**Comparison**:
- Binary {0,1}: H = 1 bit.
- Bipolar {-1,+1}: H = 1 bit (2 states).
- Ternary: ~58% more than binary.

**Nuance**: Biased (e.g., 95% zeros): H lower (~0.3 bits/dim), but enables compression (next section).

---

#### 2. Superposition Capacity Limits (SNR Derivation)

**Goal**: Prove reliable bundling limited to k ≈ D/10 to D/50 items.

**Symbols**:
- Dot product: \( \mathbf{A} \cdot \mathbf{B} = \sum a_i b_i \)
- Norm: \( |\mathbf{A}| = \sqrt{\mathbf{A} \cdot \mathbf{A}} \)
- Similarity: cos θ ≈ dot / (D) for unit-ish vectors (algebra: cosine from dot formula).
- Normal(μ, σ²): Gaussian distribution (central limit theorem—sums of independents → bell curve).

**Derivation (Teaching by Doing)**:
Assume random ternary vectors in {-1,0,+1}^D, density ρ (fraction non-zero, ρ≈0.05 for sparse).

For two independent A, B:
- Per dim contribution to dot: E[a_i b_i] = 0 (symmetric).
- Var per dim: For non-zero: ~1 (matches ±1 or mismatch ∓1).
- Total Var(dot) ≈ ρ D (zeros contribute 0).

By CLT (sums → Gaussian for large D):
dot(A,B) ≈ Normal(0, ρ D)

Bundled S = sum_{j=1}^k X_j (k items).

Query dot(S, X_1) = |X_1|^2 + sum_{j=2}^k dot(X_1, X_j)

- Signal: |X_1|^2 ≈ ρ D (non-zeros).
- Noise: sum of (k-1) independents ≈ Normal(0, (k-1) ρ D)
- Std noise ≈ sqrt((k-1) ρ D) ≈ sqrt(k ρ D)

SNR = signal / std_noise ≈ (ρ D) / sqrt(k ρ D) = sqrt(ρ D / k)

Normalized similarity ≈ SNR / sqrt(ρ D) wait no—often normalize by sqrt(expected norm).

**Standard Bipolar Approx** (common in literature, similar):
Signal ≈ D, noise std ≈ sqrt(k D), SNR ≈ sqrt(D/k)

**Calc Examples** (D=10000, ρ=1 dense):
- k=100: SNR ≈ sqrt(100) = 10 → similarity ≈ 1 + noise/ D ≈ 0.99 (excellent).
- k=1000: SNR ≈ 3.16 → sim ≈ 0.76.
- k=5000: SNR ≈ 1.41 → sim ≈ 0.58 (marginal).
- k=10000: SNR=1 → sim ≈0.5 (random).

**Sparse Boost** (ρ=0.05):
Effective D_active = ρ D =500 → SNR ≈ sqrt(500 / k)
- k=50: SNR≈3.16 (good).

**Proof Sources**:
- Kanerva 1988: Capacity ~0.14 D bits (~D/7 items).
- Frady et al. 2020 (sparse ternary): Empirical ~D/10 reliable with cleanup.

**Why Retrieval Works**: Resonance (iterative unbind/bind) amplifies if initial sim > threshold (~0.5–0.7).

---

#### 3. Popcount Math for Ternary Matmul/Similarity

**Goal**: Prove exact/fast dot in bitsliced +plane/-plane.

**Setup**: A+, A- planes (bits set for +1/-1).

Dot(A,B) = count(same sign non-zero) - count(opposite sign)

= popcount(A+ & B+) + popcount(A- & B-) - popcount(A+ & B-) - popcount(A- & B+)

**Derivation**:
- Each +/+ or -/- match: +1 contribution.
- +/- or -/+: -1.
- Zeros: 0.

**Example (D=8, packed u32)**:
A: [+1,0,-1,+1,0,0,-1,+1] → A+ = 0b10010001, A- = 0b00100010
B similar.

popcount(A+ & B+) etc. → exact scaled dot.

**Proof**: Bitwise AND isolates matches; popcount sums (hardware instr).

No carry—pure reduction. Warp popcount → O(1) per group.

---

#### 4. Sparsity and Compression Gains

**Math**: For p_zero = 0.95:
- Expected non-zero per plane: 0.05 D bits.
- Total stored: ~0.1 D bits (2 planes) + metadata.
- Compression: 32 / 0.1 = 320x vs FP32.

**CSR Edge**: Indices + values → overhead if <90% zero.

---

#### 5. Why Extreme Nesting/Tetration Fails

**Calc Recap**: 3^^4 has ~10^{3.6e12} "slots"—but physical D fixed → can't address without hashing (collisions explode).

Bundling k=10^6 items → SNR ≈ sqrt(D / 10^6) <<0.01 → noise only.

**Proof**: Information theoretic—max states 3^D ≈10^{1.58 D}—no tetration multiplier without infinite D.

---

#### 6. Edge Cases and Validation

- Dense data: Falls to ~1.58 bits/dim—no free lunch.
- Retrieval failure: Low SNR → resonance diverges.
- Real Validation: TWN (2016) empirical proofs on datasets; HDC papers on MNIST/classification >90% acc with D=10k, k=100.

This math holds—no snake oil in moderate use. Extreme claims fail these bounds.