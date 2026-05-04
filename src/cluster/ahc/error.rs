//! Errors for `diarization::cluster::ahc`.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
  /// Input shape is invalid (empty embeddings, zero-norm row, bad threshold).
  #[error("ahc: shape error: {0}")]
  Shape(#[from] ShapeError),
  /// A NaN/`±inf` entry was found in the embeddings.
  #[error("ahc: non-finite value in {0}")]
  NonFinite(#[from] NonFiniteField),
  /// Failed to allocate the condensed pdist buffer. On large
  /// `num_train`, the buffer can exceed
  /// `SpillOptions::threshold_bytes` and route through the file-
  /// backed mmap path; surface tempfile / mmap failures here.
  ///
  /// [`SpillOptions::threshold_bytes`]: crate::ops::spill::SpillOptions::threshold_bytes
  #[error("ahc: failed to allocate condensed pdist buffer: {0}")]
  Spill(#[from] crate::ops::spill::SpillError),
}

/// Specific shape-violation reasons for [`Error::Shape`].
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum ShapeError {
  #[error("embeddings must have at least one row")]
  EmptyEmbeddings,
  #[error("embeddings must have at least one column")]
  ZeroEmbeddingDim,
  #[error("threshold must be a positive finite scalar")]
  InvalidThreshold,
  #[error("embeddings row has zero L2 norm; cannot normalize")]
  ZeroNormRow,
  #[error(
    "embeddings row's squared-norm accumulator overflowed to +inf \
     (sum of v*v exceeded f64::MAX); the normalize step would collapse \
     it to all-zeros and silently corrupt the clustering"
  )]
  RowNormOverflow,
}

/// Field that contained a non-finite value.
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum NonFiniteField {
  #[error("embeddings")]
  Embeddings,
}
