//! Error type for the segmentation module.

use core::fmt;

#[cfg(feature = "std")]
use std::path::PathBuf;

use crate::segment::types::WindowId;

/// All errors produced by `dia::segment`.
#[derive(Debug)]
pub enum Error {
  /// Construction-time validation failure for `SegmentOptions`.
  InvalidOptions(&'static str),

  /// `push_inference` received a `scores` slice of the wrong length.
  InferenceShapeMismatch {
    /// Expected element count: `FRAMES_PER_WINDOW * POWERSET_CLASSES`.
    expected: usize,
    /// Actual length received.
    got: usize,
  },

  /// `push_inference` was called with a `WindowId` that wasn't yielded by
  /// `poll` (or that has already been consumed).
  UnknownWindow {
    /// The unknown id.
    id: WindowId,
  },

  /// The `ort::Session` failed to load the model file.
  #[cfg(feature = "ort")]
  LoadModel {
    /// Path passed to `from_file`.
    path: PathBuf,
    /// Underlying ort error.
    source: ort::Error,
  },

  /// Generic ort runtime error from `SegmentModel::infer` or session ops.
  #[cfg(feature = "ort")]
  Ort(ort::Error),
}

impl fmt::Display for Error {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Self::InvalidOptions(msg) => write!(f, "invalid segment options: {msg}"),
      Self::InferenceShapeMismatch { expected, got } => {
        write!(f, "inference scores length {got}, expected {expected}")
      }
      Self::UnknownWindow { id } => {
        write!(f, "inference scores received for unknown WindowId {id:?}")
      }
      #[cfg(feature = "ort")]
      Self::LoadModel { path, source } => {
        write!(f, "failed to load model from {}: {source}", path.display())
      }
      #[cfg(feature = "ort")]
      Self::Ort(e) => write!(f, "ort runtime error: {e}"),
    }
  }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {
  fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
    match self {
      #[cfg(feature = "ort")]
      Self::LoadModel { source, .. } => Some(source),
      #[cfg(feature = "ort")]
      Self::Ort(e) => Some(e),
      _ => None,
    }
  }
}

#[cfg(feature = "ort")]
impl From<ort::Error> for Error {
  fn from(e: ort::Error) -> Self {
    Self::Ort(e)
  }
}
