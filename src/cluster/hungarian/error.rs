//! Errors for `diarization::cluster::hungarian`.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
  /// Input shape is invalid (e.g., 0 speakers or 0 clusters).
  #[error("hungarian: shape error: {0}")]
  Shape(#[from] ShapeError),
  /// A NaN/`±inf` entry was found in the cost matrix.
  #[error("hungarian: non-finite value: {0}")]
  NonFinite(#[from] NonFiniteError),
}

/// Specific shape-violation reasons for [`Error::Shape`].
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum ShapeError {
  #[error("chunks must contain at least one chunk")]
  EmptyChunks,
  #[error("num_speakers must be at least 1")]
  ZeroSpeakers,
  #[error("num_clusters must be at least 1")]
  ZeroClusters,
  #[error("all chunks must share the same shape")]
  InconsistentChunkShape,
}

/// Specific non-finite reasons for [`Error::NonFinite`].
#[derive(Debug, Error, Clone, Copy, PartialEq)]
pub enum NonFiniteError {
  #[error("soft_clusters contains +inf or -inf")]
  InfInSoftClusters,
  #[error("soft_clusters has no finite entries; cannot compute nanmin replacement")]
  NoFiniteEntries,
  /// A finite cost magnitude exceeds [`MAX_COST_MAGNITUDE`]. The
  /// `kuhn_munkres` solver internally accumulates `lx[i] + ly[j] -
  /// weight[i,j]` and label sums; values approaching `f64::MAX`
  /// overflow to `±inf` after one or two additions, which can wedge
  /// the solver per the crate's own docs and reintroduce the failure
  /// mode the upstream `±inf` guard exists to prevent.
  ///
  /// `MAX_COST_MAGNITUDE = 1e15` is the documented safe range:
  /// production cosine distances and PLDA log-likelihoods are bounded
  /// by O(1)–O(100), so any value beyond `1e15` indicates upstream
  /// corruption rather than a legitimate cost matrix.
  ///
  /// [`MAX_COST_MAGNITUDE`]: crate::cluster::hungarian::MAX_COST_MAGNITUDE
  #[error(
    "soft_clusters contains finite value {value:e} with |value| > MAX_COST_MAGNITUDE ({max:e})"
  )]
  WeightOutOfBounds { value: f64, max: f64 },
}
