//! Scalar AXPY: `y += alpha * x`.

/// In-place fused multiply-add over a slice: `y[i] += alpha * x[i]`
/// for each `i`.
///
/// Used by `centroid::weighted_centroids`'s
/// `centroids[k, d] += w * embeddings[t, d]` accumulator. The
/// k-by-d-by-t triple-nested loop reduces to repeated AXPY calls
/// (one per `(k, t)` pair, sized by `d = embed_dim`).
///
/// # Panics (debug only)
///
/// Debug asserts on `y.len() == x.len()`.
#[inline]
#[allow(dead_code)] // Step 2: scaffolded; centroid will route through this in Step 3.
pub fn axpy(y: &mut [f64], alpha: f64, x: &[f64]) {
  debug_assert_eq!(y.len(), x.len(), "axpy: length mismatch");
  for i in 0..y.len() {
    y[i] += alpha * x[i];
  }
}
