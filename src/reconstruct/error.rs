//! Errors for `dia::reconstruct`.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
  #[error("reconstruct: shape error: {0}")]
  Shape(&'static str),
  #[error("reconstruct: non-finite value in {0}")]
  NonFinite(&'static str),
  #[error("reconstruct: invalid sliding-window timing: {0}")]
  Timing(&'static str),
}
