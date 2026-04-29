//! Errors for `dia::pipeline`.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
  /// Input shape is invalid (e.g., zero chunks, mismatched dims, etc.).
  #[error("pipeline: shape error: {0}")]
  Shape(&'static str),
  /// A NaN/`±inf` entry was found where finite values are required.
  #[error("pipeline: non-finite value in {0}")]
  NonFinite(&'static str),
  /// `min_active_ratio` falls outside `(0.0, 1.0]`.
  #[error("pipeline: invalid min_active_ratio (must be in (0, 1]): {0}")]
  InvalidActiveRatio(f64),
  /// Fewer than 2 active embeddings — pyannote takes a separate
  /// fast-path here. The Rust port surfaces this as a typed error so
  /// the caller can decide (e.g. assign all to a single cluster).
  #[error("pipeline: only {0} active embeddings; AHC needs >= 2 to cluster")]
  TooFewActiveEmbeddings(usize),
  /// Propagated from `dia::ahc`.
  #[error("pipeline: ahc: {0}")]
  Ahc(#[from] crate::ahc::Error),
  /// Propagated from `dia::vbx`.
  #[error("pipeline: vbx: {0}")]
  Vbx(#[from] crate::vbx::Error),
  /// Propagated from `dia::centroid`.
  #[error("pipeline: centroid: {0}")]
  Centroid(#[from] crate::centroid::Error),
  /// Propagated from `dia::hungarian`.
  #[error("pipeline: hungarian: {0}")]
  Hungarian(#[from] crate::hungarian::Error),
  /// Propagated from `dia::plda`.
  #[error("pipeline: plda: {0}")]
  Plda(#[from] crate::plda::Error),
}
