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
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum NonFiniteError {
  #[error("soft_clusters contains +inf or -inf")]
  InfInSoftClusters,
  #[error("soft_clusters has no finite entries; cannot compute nanmin replacement")]
  NoFiniteEntries,
}
