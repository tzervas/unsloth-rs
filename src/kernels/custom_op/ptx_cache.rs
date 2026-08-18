// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! Process-wide C→PTX string cache and host helpers (no NVRTC).
//!
//! Hits return `Arc<str>` so the PTX text is not cloned. A Mutex still
//! serializes lookups — `compile_cached` is not a launch-tax close.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// Next power of two, minimum 1.
#[must_use]
pub fn next_pow2(n: usize) -> usize {
    if n <= 1 {
        1
    } else {
        n.next_power_of_two()
    }
}

/// p50 = midpoint; p99 = ceil(0.99*(n-1)) into sorted samples.
#[must_use]
pub fn sorted_percentile(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if (p - 0.50).abs() < 1e-12 {
        if n % 2 == 1 {
            sorted[n / 2]
        } else {
            0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
        }
    } else {
        let idx = ((p * (n - 1) as f64).ceil() as usize).min(n - 1);
        sorted[idx]
    }
}

/// In-process PTX text cache keyed by module name.
pub struct PtxCache {
    map: Mutex<HashMap<String, Arc<str>>>,
    compiles: AtomicUsize,
}

impl Default for PtxCache {
    fn default() -> Self {
        Self {
            map: Mutex::new(HashMap::new()),
            compiles: AtomicUsize::new(0),
        }
    }
}

impl PtxCache {
    /// Empty cache.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// How many times a compile closure ran (cache misses).
    #[must_use]
    pub fn compile_count(&self) -> usize {
        self.compiles.load(Ordering::Relaxed)
    }

    /// Return cached PTX for `module_name`, compiling once on miss.
    pub fn get_or_insert<E>(
        &self,
        module_name: &str,
        compile: impl FnOnce() -> Result<String, E>,
    ) -> Result<Arc<str>, E> {
        {
            let guard = match self.map.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            if let Some(ptx) = guard.get(module_name) {
                return Ok(Arc::clone(ptx));
            }
        }
        let ptx: Arc<str> = compile()?.into();
        self.compiles.fetch_add(1, Ordering::Relaxed);
        let mut guard = match self.map.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        Ok(Arc::clone(
            guard
                .entry(module_name.to_string())
                .or_insert_with(|| Arc::clone(&ptx)),
        ))
    }
}

static PTX_CACHE: OnceLock<PtxCache> = OnceLock::new();

/// Process-wide cache used by `nvrtc::load_func`.
#[must_use]
pub fn global_ptx_cache() -> &'static PtxCache {
    PTX_CACHE.get_or_init(PtxCache::new)
}

/// Compile-miss count on the process-wide cache.
#[must_use]
pub fn ptx_compile_count() -> usize {
    global_ptx_cache().compile_count()
}

#[cfg(test)]
mod tests {
    use super::{next_pow2, sorted_percentile, PtxCache};
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn next_pow2_edges() {
        assert_eq!(next_pow2(0), 1);
        assert_eq!(next_pow2(1), 1);
        assert_eq!(next_pow2(3), 4);
        assert_eq!(next_pow2(1024), 1024);
    }

    #[test]
    fn percentile_midpoint_and_p99() {
        let even = [1.0, 2.0, 3.0, 4.0];
        assert!((sorted_percentile(&even, 0.50) - 2.5).abs() < 1e-12);
        let odd = [1.0, 2.0, 3.0];
        assert!((sorted_percentile(&odd, 0.50) - 2.0).abs() < 1e-12);
        let samples: Vec<f64> = (0..100).map(f64::from).collect();
        // ceil(0.99 * 99) = 99
        assert!((sorted_percentile(&samples, 0.99) - 99.0).abs() < 1e-12);
    }

    #[test]
    fn cache_key_skips_second_compile() {
        let cache = PtxCache::new();
        let n = AtomicUsize::new(0);
        let a = cache
            .get_or_insert("mod", || {
                n.fetch_add(1, Ordering::SeqCst);
                Ok::<_, ()>("ptx-src".to_string())
            })
            .unwrap();
        let b = cache
            .get_or_insert("mod", || {
                n.fetch_add(1, Ordering::SeqCst);
                Ok::<_, ()>("other".to_string())
            })
            .unwrap();
        assert_eq!(&*a, "ptx-src");
        assert_eq!(&*b, "ptx-src");
        assert_eq!(n.load(Ordering::SeqCst), 1);
        assert_eq!(cache.compile_count(), 1);
    }
}
