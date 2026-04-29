//! Weighted centroid computation: `W.T @ X / W.sum(0).T`, where
//! `W = q[:, sp > sp_threshold]`.

use crate::centroid::error::Error;
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
  let num_alive = alive.len();
  let mut centroids = DMatrix::<f64>::zeros(num_alive, embed_dim);
  for (alive_idx, &k) in alive.iter().enumerate() {
    let mut w_total = 0.0;
    for t in 0..num_train {
      let w = q[(t, k)];
      w_total += w;
      for d in 0..embed_dim {
        centroids[(alive_idx, d)] += w * embeddings[(t, d)];
      }
    }
    if w_total <= 0.0 {
      return Err(Error::Shape(
        "surviving cluster has non-positive total weight; \
         cannot normalize without producing NaN",
      ));
    }
    for d in 0..embed_dim {
      centroids[(alive_idx, d)] /= w_total;
    }
  }

  Ok(centroids)
}
