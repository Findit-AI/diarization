//! Scalar pairwise Euclidean distance, condensed `pdist` ordering.
//!
//! Implementation matches [`crate::ops::arch::neon::pdist_euclidean`]
//! bit-for-bit:
//! - Per-element squared accumulation via `f64::mul_add(diff, diff,
//!   acc)` (one IEEE 754 rounding, same as `vfmaq_f64`).
//! - Four partial accumulators over modulo-4 residue classes,
//!   mirroring NEON's two 2-lane registers.
//! - Final reduction tree `((s00 + s10) + (s01 + s11))` then `sqrt`.

/// Pairwise Euclidean distance over the rows of a `(n, d)` row-major
/// f64 matrix, returned in `pdist`-style condensed ordering:
/// `[d(0,1), d(0,2), ..., d(0,n-1), d(1,2), ..., d(n-2,n-1)]`,
/// length `n * (n - 1) / 2`. This is the format `kodama::linkage`
/// expects.
///
/// `rows` is a flat slice of length `n * d`, row-major: row `i`'s
/// d-vector starts at `&rows[i * d ..]`.
///
/// # Panics
///
/// - `debug_assert!` on `rows.len() == n * d`.
/// - Always panics if `d == 0` (zero-dim distance is undefined; without
///   this guard, `rows.len() == n * d == 0` would silently let any `n`
///   pass and `Vec::with_capacity(n * (n-1) / 2)` would OOM).
/// - Always panics if `n * (n - 1)` overflows `usize` — independent of
///   the `n * d` shape check; matters on 32-bit and any time `n` is
///   large relative to pointer width.
pub fn pdist_euclidean(rows: &[f64], n: usize, d: usize) -> Vec<f64> {
  assert!(d >= 1, "scalar::pdist_euclidean: d ({d}) must be >= 1");
  debug_assert_eq!(rows.len(), n * d, "pdist_euclidean: shape mismatch");
  // `n * (n-1)` overflow check, independent of `n * d`. Result is the
  // condensed output capacity; `/ 2` is safe (product is always even).
  let pair_count = if n >= 2 {
    n.checked_mul(n - 1)
      .expect("scalar::pdist_euclidean: n * (n - 1) overflows usize")
      / 2
  } else {
    0
  };
  let mut out = Vec::with_capacity(pair_count);
  for i in 0..n {
    let row_i = &rows[i * d..(i + 1) * d];
    for j in (i + 1)..n {
      let row_j = &rows[j * d..(j + 1) * d];
      let mut s00 = 0.0_f64;
      let mut s01 = 0.0_f64;
      let mut s10 = 0.0_f64;
      let mut s11 = 0.0_f64;
      let mut k = 0usize;
      while k + 4 <= d {
        let d0 = row_i[k] - row_j[k];
        let d1 = row_i[k + 1] - row_j[k + 1];
        let d2 = row_i[k + 2] - row_j[k + 2];
        let d3 = row_i[k + 3] - row_j[k + 3];
        s00 = f64::mul_add(d0, d0, s00);
        s01 = f64::mul_add(d1, d1, s01);
        s10 = f64::mul_add(d2, d2, s10);
        s11 = f64::mul_add(d3, d3, s11);
        k += 4;
      }
      if k + 2 <= d {
        let d0 = row_i[k] - row_j[k];
        let d1 = row_i[k + 1] - row_j[k + 1];
        s00 = f64::mul_add(d0, d0, s00);
        s01 = f64::mul_add(d1, d1, s01);
        k += 2;
      }
      let mut sq = (s00 + s10) + (s01 + s11);
      while k < d {
        let diff = row_i[k] - row_j[k];
        sq = f64::mul_add(diff, diff, sq);
        k += 1;
      }
      out.push(sq.sqrt());
    }
  }
  out
}
