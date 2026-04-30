//! Scalar pairwise Euclidean distance, condensed `pdist` ordering.

/// Pairwise Euclidean distance over the rows of a `(n, d)` row-major
/// f64 matrix, returned in `pdist`-style condensed ordering:
/// `[d(0,1), d(0,2), ..., d(0,n-1), d(1,2), ..., d(n-2,n-1)]`,
/// length `n * (n - 1) / 2`. This is the format `kodama::linkage`
/// expects.
///
/// `rows` is a flat slice of length `n * d`, row-major: row `i`'s
/// d-vector starts at `&rows[i * d ..]`. (We don't take a `DMatrix`
/// here because nalgebra is column-major by default, so its row
/// access is strided — fine for scalar but a non-starter for SIMD
/// in Step 3. Forcing a row-major slice at the boundary keeps the
/// scalar/SIMD contracts identical.)
///
/// # Panics (debug only)
///
/// Debug asserts on `rows.len() == n * d`.
pub fn pdist_euclidean(rows: &[f64], n: usize, d: usize) -> Vec<f64> {
  debug_assert_eq!(rows.len(), n * d, "pdist_euclidean: shape mismatch");
  let mut out = Vec::with_capacity(n * (n - 1) / 2);
  for i in 0..n {
    let row_i = &rows[i * d..(i + 1) * d];
    for j in (i + 1)..n {
      let row_j = &rows[j * d..(j + 1) * d];
      let mut sq = 0.0;
      for k in 0..d {
        let diff = row_i[k] - row_j[k];
        sq += diff * diff;
      }
      out.push(sq.sqrt());
    }
  }
  out
}
