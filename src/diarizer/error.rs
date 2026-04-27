//! Error type for [`crate::diarizer::Diarizer`]. Filled in by Task 34.

use thiserror::Error;

/// Errors returned by [`crate::diarizer::Diarizer`] APIs.
#[derive(Debug, Error)]
pub enum Error {
  /// Surface a [`crate::segment::Error`].
  #[error(transparent)]
  Segment(#[from] crate::segment::Error),
  /// Surface a [`crate::embed::Error`].
  #[error(transparent)]
  Embed(#[from] crate::embed::Error),
  /// Surface a [`crate::cluster::Error`].
  #[error(transparent)]
  Cluster(#[from] crate::cluster::Error),
  /// Internal invariant violation.
  #[error(transparent)]
  Internal(#[from] InternalError),
}

/// Internal invariant violation (Diarizer state machine bug or unsupported input).
#[derive(Debug, Error)]
pub enum InternalError {
  /// Placeholder until Task 34 enumerates real internal errors.
  #[error("placeholder")]
  Placeholder,
}
