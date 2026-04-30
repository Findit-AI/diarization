//! Pairwise Euclidean distance dispatcher.

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
use crate::ops::arch;
#[cfg(target_arch = "aarch64")]
use crate::ops::neon_available;
use crate::ops::scalar;
#[cfg(target_arch = "x86_64")]
use crate::ops::{avx2_available, avx512_available};

/// Pairwise Euclidean distance over the rows of an `(n, d)` row-major
/// f64 matrix; condensed `pdist`-style ordering. See
/// [`crate::ops::scalar::pdist_euclidean`] for the contract.
#[inline]
pub fn pdist_euclidean(rows: &[f64], n: usize, d: usize, use_simd: bool) -> Vec<f64> {
  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: `neon_available()` confirmed NEON.
          return unsafe { arch::neon::pdist_euclidean(rows, n, d) };
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: `avx512_available()` confirmed AVX-512F.
          return unsafe { arch::x86_avx512::pdist_euclidean(rows, n, d) };
        }
        if avx2_available() {
          // SAFETY: `avx2_available()` confirmed AVX2 + FMA.
          return unsafe { arch::x86_avx2::pdist_euclidean(rows, n, d) };
        }
      },
      _ => {}
    }
  }
  scalar::pdist_euclidean(rows, n, d)
}
