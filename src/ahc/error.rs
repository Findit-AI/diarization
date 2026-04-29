//! Errors for `dia::ahc`.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
  /// Input shape is invalid (empty embeddings, zero-norm row, bad threshold).
  #[error("ahc: shape error: {0}")]
  Shape(&'static str),
  /// A NaN/`±inf` entry was found in the embeddings.
  #[error("ahc: non-finite value in {0}")]
  NonFinite(&'static str),
}
