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
}

/// Field that contained a non-finite value.
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum NonFiniteField {
  #[error("embeddings")]
  Embeddings,
}
