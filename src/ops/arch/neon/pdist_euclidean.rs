//! NEON f64 pairwise Euclidean distance.
//!
//! Per row pair `(i, j)` with `j > i`, computes `||row_i - row_j||²`
//! with 2-lane SIMD `vsubq_f64 + vfmaq_f64` (squared accumulator),
//! then `sqrt` at the end. Output preserves `pdist`-style condensed
//! ordering identical to the scalar reference.
//!
//! The hot row-by-row inner loop dominates AHC cost; D = 192 / 256
//! production dims are 4-aligned, so the 4-wide unroll runs without
//! tail in production.

use core::arch::aarch64::{
  float64x2_t, vaddq_f64, vaddvq_f64, vdupq_n_f64, vfmaq_f64, vld1q_f64, vsubq_f64,
};

/// Pairwise Euclidean distance, condensed `pdist` ordering. See
/// [`crate::ops::scalar::pdist_euclidean`] for the contract.
///
/// # Safety
///
/// 1. NEON must be available (caller's obligation).
/// 2. `rows.len() == n * d` (debug-asserted).
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn pdist_euclidean(rows: &[f64], n: usize, d: usize) -> Vec<f64> {
  debug_assert_eq!(rows.len(), n * d, "neon::pdist_euclidean: shape mismatch");
  let mut out = Vec::with_capacity(n * (n - 1) / 2);

  // SAFETY: row indices are in `0..n` and pointer adds are bounded by
  // `i*d + d <= n*d == rows.len()`. Inner SIMD load offsets are bounded
  // by the `k + 4 <= d` / `k + 2 <= d` loop conditions.
  unsafe {
    for i in 0..n {
      let row_i_ptr = rows.as_ptr().add(i * d);
      for j in (i + 1)..n {
        let row_j_ptr = rows.as_ptr().add(j * d);
        let mut acc0: float64x2_t = vdupq_n_f64(0.0);
        let mut acc1: float64x2_t = vdupq_n_f64(0.0);
        let mut k = 0usize;
        while k + 4 <= d {
          let a0 = vld1q_f64(row_i_ptr.add(k));
          let b0 = vld1q_f64(row_j_ptr.add(k));
          let a1 = vld1q_f64(row_i_ptr.add(k + 2));
          let b1 = vld1q_f64(row_j_ptr.add(k + 2));
          let diff0 = vsubq_f64(a0, b0);
          let diff1 = vsubq_f64(a1, b1);
          acc0 = vfmaq_f64(acc0, diff0, diff0);
          acc1 = vfmaq_f64(acc1, diff1, diff1);
          k += 4;
        }
        if k + 2 <= d {
          let a0 = vld1q_f64(row_i_ptr.add(k));
          let b0 = vld1q_f64(row_j_ptr.add(k));
          let diff0 = vsubq_f64(a0, b0);
          acc0 = vfmaq_f64(acc0, diff0, diff0);
          k += 2;
        }
        let acc = vaddq_f64(acc0, acc1);
        let mut sq = vaddvq_f64(acc);
        // Scalar tail.
        while k < d {
          let diff = *row_i_ptr.add(k) - *row_j_ptr.add(k);
          sq += diff * diff;
          k += 1;
        }
        out.push(sq.sqrt());
      }
    }
  }

  out
}
