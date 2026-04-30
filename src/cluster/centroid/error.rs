//! Errors for `diarization::cluster::centroid`.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
  /// Input shape is invalid (mismatched dims, no surviving clusters,
  /// non-positive `sp_threshold`, etc.).
  #[error("centroid: shape error: {0}")]
  Shape(&'static str),
  /// A NaN/`±inf` entry was found in `q`, `sp`, or `embeddings`.
  #[error("centroid: non-finite value in {0}")]
  NonFinite(&'static str),
}
