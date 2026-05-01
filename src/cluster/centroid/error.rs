//! Errors for `diarization::cluster::centroid`.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
  /// Input shape is invalid (mismatched dims, no surviving clusters,
  /// non-positive `sp_threshold`, etc.).
  #[error("centroid: shape error: {0}")]
  Shape(&'static str),
  /// A NaN/`±inf` entry was found in `q`, `sp`, or `embeddings`.
  #[error("centroid: non-finite value in {0}")]
  NonFinite(&'static str),
  /// A `sp[k]` value lands inside the SIMD-vs-scalar guard band around
  /// `sp_threshold`. The discrete alive/squashed decision could differ
  /// across CPU backends (NEON ↔ AVX2 ↔ AVX-512 reductions diverge by
  /// O(1e-15) relative). Caller must rerun on a deterministic path or
  /// surface the input as ambiguous. See `weighted_centroids` for
  /// the band definition. Codex review HIGH round 10.
  #[error(
    "centroid: sp[{cluster}] = {value:.3e} lands within the SIMD guard band \
     [{lo:.0e}, {hi:.0e}] around sp_threshold = {threshold:.0e}; \
     alive/squashed decision is non-deterministic across CPU backends"
  )]
  AmbiguousAliveCluster {
    /// The cluster index whose `sp` lands in the guard band.
    cluster: usize,
    /// The actual `sp[cluster]` value.
    value: f64,
    /// The configured `sp_threshold`.
    threshold: f64,
    /// Lower bound of the guard band (exclusive).
    lo: f64,
    /// Upper bound of the guard band (exclusive).
    hi: f64,
  },
}
