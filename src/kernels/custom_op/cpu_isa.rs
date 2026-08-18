// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! Host ISA helpers for CustomOp CPU paths.
//!
//! This desktop (akula-prime) is an i7-14700K: AVX2 + FMA + AVX-VNNI.
//! Consumer 14th-gen does **not** have AVX-512 or AMX. P-cores are
//! `/sys/devices/cpu_core/cpus` (`0-15` here); E-cores `16-27`. The crate
//! does not pin affinity — use `taskset -c 0-15` if you want P-only.

/// Runtime threads for row-parallel CPU kernels. Caps at available parallelism.
#[must_use]
pub fn cpu_worker_threads(rows: usize) -> usize {
    let hw = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
        .max(1);
    hw.min(rows).max(1)
}

/// `true` when a vocab-tile Candle GEMM is likely cheaper than the scalar
/// row CustomOp (GEMM already uses AVX2/FMA on this host).
#[must_use]
#[allow(dead_code)] // policy helper; used by tests and future CPU-tile dispatch
pub fn prefer_tile_gemm(rows: usize, vocab: usize, dim: usize) -> bool {
    rows >= 32 && vocab >= 64 && dim >= 32
}

/// Dot product. AVX2+FMA when the CPU advertises it; otherwise scalar.
#[must_use]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let len = a.len().min(b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if len >= 8 && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: AVX2+FMA advertised; slices have at least `len` f32s.
            return unsafe { dot_f32_avx2_fma(&a[..len], &b[..len]) };
        }
    }
    let mut sum = 0.0f32;
    for idx in 0..len {
        sum += a[idx] * b[idx];
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_avx2_fma(lhs: &[f32], rhs: &[f32]) -> f32 {
    use std::arch::x86_64::{
        _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_setzero_ps, _mm256_storeu_ps,
    };
    // SAFETY: caller checked AVX2+FMA; unaligned loads stay inside lhs/rhs.
    let len = lhs.len();
    let mut acc = _mm256_setzero_ps();
    let mut offset = 0usize;
    while offset + 8 <= len {
        let left = _mm256_loadu_ps(lhs.as_ptr().add(offset));
        let right = _mm256_loadu_ps(rhs.as_ptr().add(offset));
        acc = _mm256_fmadd_ps(left, right, acc);
        offset += 8;
    }
    let mut lanes = [0.0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
    let mut sum = 0.0f32;
    for lane in lanes {
        sum += lane;
    }
    while offset < len {
        sum += lhs[offset] * rhs[offset];
        offset += 1;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::dot_f32;

    #[test]
    fn dot_matches_scalar() {
        let a: Vec<f32> = (0..17).map(|i| i as f32 * 0.25).collect();
        let b: Vec<f32> = (0..17).map(|i| 1.5 - i as f32 * 0.1).collect();
        let got = dot_f32(&a, &b);
        let exp: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        assert!((got - exp).abs() < 1e-5, "got={got} exp={exp}");
    }

    #[test]
    fn prefer_tile_thresholds() {
        assert!(!super::prefer_tile_gemm(6, 20, 8));
        assert!(super::prefer_tile_gemm(32, 64, 32));
    }
}
