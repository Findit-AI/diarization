//! Weighted centroid computation: `W.T @ X / W.sum(0).T`, where
//! `W = q[:, sp > sp_threshold]`.

use crate::cluster::centroid::error::Error;
use nalgebra::{DMatrix, DVector};

/// Pyannote's hardcoded `sp > 1e-7` filter (clustering.py:619). Speakers
/// whose VBx prior `sp` falls below this floor are treated as
/// extinguished and their `q`-column is dropped before centroid
/// computation. Captured `sp_final` for the Phase-0 fixture has 2
/// surviving values (~0.85 + 0.15) and 17 squashed values at ~1.76e-14,
/// well below the threshold.
pub const SP_ALIVE_THRESHOLD: f64 = 1.0e-7;

/// Compute weighted centroids from VBx posterior responsibilities.
///
/// Mirrors `pyannote/audio/pipelines/clustering.py:618-621`:
///
/// ```python
/// W = q[:, sp > 1e-7]
/// centroids = W.T @ train_embeddings.reshape(-1, dimension) / W.sum(0, keepdims=True).T
/// ```
///
/// # Inputs
///
/// - `q`: VBx posterior responsibilities, shape `(num_train,
///   num_init_clusters)` (the `q_final` returned by `vbx_iterate`).
/// - `sp`: VBx final speaker priors, shape `(num_init_clusters,)`
///   (the `pi` returned by `vbx_iterate`).
/// - `embeddings`: raw `(num_train, embed_dim)` x-vectors that pyannote
///   averages with `q` weights — *not* the post-PLDA features.
/// - `sp_threshold`: drop columns where `sp[k] <= threshold`. Pass
///   [`SP_ALIVE_THRESHOLD`] for pyannote parity.
///
/// # Output
///
/// `(num_alive, embed_dim)` matrix of weighted-mean embeddings.
/// `num_alive = (sp > threshold).count()`.
///
/// # Errors
///
/// - [`Error::Shape`] for any dimension mismatch, no surviving clusters,
///   or a surviving cluster with zero total weight (would produce a
///   `NaN` centroid).
/// - [`Error::NonFinite`] if any input contains a NaN/`±inf`.
pub fn weighted_centroids(
  q: &DMatrix<f64>,
  sp: &DVector<f64>,
  embeddings: &DMatrix<f64>,
  sp_threshold: f64,
) -> Result<DMatrix<f64>, Error> {
  // SIMD always on; see `cluster::vbx::vbx_iterate` for the
  // rationale (matrixmultiply GEMM already arch-dispatches inside
  // VBx, so a use_simd gate here was never sufficient for cross-
  // arch bit-equality, and algorithm robustness is validated
  // empirically by parity tests).
  weighted_centroids_inner(q, sp, embeddings, sp_threshold, true)
}

/// Test-only entrypoint: identical to [`weighted_centroids`] but with
/// the `use_simd` flag exposed so end-to-end differential tests can
/// A/B both backends on identical inputs. Production code goes
/// through [`weighted_centroids`] which always passes `true`.
#[cfg(test)]
pub(crate) fn weighted_centroids_with_simd(
  q: &DMatrix<f64>,
  sp: &DVector<f64>,
  embeddings: &DMatrix<f64>,
  sp_threshold: f64,
  use_simd: bool,
) -> Result<DMatrix<f64>, Error> {
  weighted_centroids_inner(q, sp, embeddings, sp_threshold, use_simd)
}

fn weighted_centroids_inner(
  q: &DMatrix<f64>,
  sp: &DVector<f64>,
  embeddings: &DMatrix<f64>,
  sp_threshold: f64,
  use_simd: bool,
) -> Result<DMatrix<f64>, Error> {
  let (num_train, num_init) = q.shape();
  if num_train == 0 {
    return Err(Error::Shape("q must have at least one row"));
  }
  if num_init == 0 {
    return Err(Error::Shape("q must have at least one column"));
  }
  if sp.len() != num_init {
    return Err(Error::Shape("sp.len() must equal q.ncols()"));
  }
  if embeddings.nrows() != num_train {
    return Err(Error::Shape("embeddings.nrows() must equal q.nrows()"));
  }
  let embed_dim = embeddings.ncols();
  if embed_dim == 0 {
    return Err(Error::Shape("embeddings must have at least one column"));
  }
  if !sp_threshold.is_finite() {
    return Err(Error::Shape("sp_threshold must be finite"));
  }
  // Validate finite values across all inputs.
  for v in q.iter() {
    if !v.is_finite() {
      return Err(Error::NonFinite("q"));
    }
  }
  for v in sp.iter() {
    if !v.is_finite() {
      return Err(Error::NonFinite("sp"));
    }
  }
  for v in embeddings.iter() {
    if !v.is_finite() {
      return Err(Error::NonFinite("embeddings"));
    }
  }

  // Identify surviving clusters (sp > threshold).
  let alive: Vec<usize> = (0..num_init).filter(|&k| sp[k] > sp_threshold).collect();
  if alive.is_empty() {
    return Err(Error::Shape(
      "no clusters survive the sp threshold (would produce empty centroid set)",
    ));
  }

  // Compute weighted sums + total weight per surviving cluster.
  // nalgebra is column-major so `embeddings.row(t)` is strided. We
  // pre-pack `embeddings` into a row-major scratch buffer once, and
  // accumulate centroids into a row-major buffer too, so the inner
  // `centroid[k] += w * embedding[t]` reduces to `ops::axpy` over
  // contiguous f64 slices. Final write-back fills the column-major
  // `DMatrix` output. The `w_total <= 0` validation deferred to after
  // accumulation — wasted work on bad input is bounded by the input
  // shape and the error is the same either way.
  let num_alive = alive.len();
  let mut embed_buf: Vec<f64> = Vec::with_capacity(num_train * embed_dim);
  for t in 0..num_train {
    for d in 0..embed_dim {
      embed_buf.push(embeddings[(t, d)]);
    }
  }
  let mut centroid_buf: Vec<f64> = vec![0.0; num_alive * embed_dim];
  let mut w_totals: Vec<f64> = vec![0.0; num_alive];
  // SIMD AXPY: scalar and NEON produce bit-identical results
  // (scalar uses `f64::mul_add`, NEON uses `vfmaq_f64` — both single-
  // rounding FMA, no inter-element reduction so no order-divergence
  // either). Centroid coordinates downstream are bit-stable across
  // backends. Cross-arch (AVX2/AVX-512) is also bit-identical for
  // axpy specifically — see `ops::differential_tests::axpy_byte_identical`.
  for (alive_idx, &k) in alive.iter().enumerate() {
    let centroid_slice = &mut centroid_buf[alive_idx * embed_dim..(alive_idx + 1) * embed_dim];
    for t in 0..num_train {
      let w = q[(t, k)];
      w_totals[alive_idx] += w;
      let emb_slice = &embed_buf[t * embed_dim..(t + 1) * embed_dim];
      crate::ops::axpy(centroid_slice, w, emb_slice, use_simd);
    }
  }
  for &w_total in &w_totals {
    if w_total <= 0.0 {
      return Err(Error::Shape(
        "surviving cluster has non-positive total weight; \
         cannot normalize without producing NaN",
      ));
    }
  }
  // Normalize: row-wise divide by w_total. The axpy primitive doesn't
  // cover this shape (per-row scalar); a small scalar loop is fine —
  // num_alive · embed_dim is at most ~20 · 256 = 5120 ops per session.
  for (alive_idx, &w_total) in w_totals.iter().enumerate() {
    let inv_w = 1.0 / w_total;
    let centroid_slice = &mut centroid_buf[alive_idx * embed_dim..(alive_idx + 1) * embed_dim];
    for v in centroid_slice.iter_mut() {
      *v *= inv_w;
    }
  }
  let mut centroids = DMatrix::<f64>::zeros(num_alive, embed_dim);
  for k in 0..num_alive {
    for d in 0..embed_dim {
      centroids[(k, d)] = centroid_buf[k * embed_dim + d];
    }
  }

  Ok(centroids)
}
