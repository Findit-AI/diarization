//! Errors for `diarization::pipeline`.

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
  /// Propagated from `diarization::cluster::ahc`.
  #[error("pipeline: ahc: {0}")]
  Ahc(#[from] crate::cluster::ahc::Error),
  /// Propagated from `diarization::cluster::vbx`.
  #[error("pipeline: vbx: {0}")]
  Vbx(#[from] crate::cluster::vbx::Error),
  /// Propagated from `diarization::cluster::centroid`.
  #[error("pipeline: centroid: {0}")]
  Centroid(#[from] crate::cluster::centroid::Error),
  /// Propagated from `diarization::cluster::hungarian`.
  #[error("pipeline: hungarian: {0}")]
  Hungarian(#[from] crate::cluster::hungarian::Error),
  /// Propagated from `diarization::plda`.
  #[error("pipeline: plda: {0}")]
  Plda(#[from] crate::plda::Error),
}
