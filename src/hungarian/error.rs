//! Errors for `diarization::hungarian`.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
  /// Input shape is invalid (e.g., 0 speakers or 0 clusters).
  #[error("hungarian: shape error: {0}")]
  Shape(&'static str),
  /// A NaN/`±inf` entry was found in the cost matrix.
  #[error("hungarian: non-finite value in {0}")]
  NonFinite(&'static str),
}
