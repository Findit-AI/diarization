//! Errors for `diarization::pipeline`.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
  /// Input shape is invalid (e.g., zero chunks, mismatched dims, etc.).
  #[error("pipeline: shape error: {0}")]
  Shape(#[from] ShapeError),
  /// A NaN/`±inf` entry was found where finite values are required.
  #[error("pipeline: non-finite value in {0}")]
  NonFinite(#[from] NonFiniteField),
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

/// Specific shape-violation reasons for [`Error::Shape`].
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum ShapeError {
  #[error("num_chunks must be at least 1")]
  ZeroNumChunks,
  #[error("num_speakers must equal MAX_SPEAKER_SLOTS (segmentation-3.0 / community-1 = 3)")]
  WrongNumSpeakers,
  #[error("embeddings must have at least one column")]
  ZeroEmbeddingDim,
  #[error("num_chunks * num_speakers overflows usize")]
  EmbeddingsRowsOverflow,
  #[error("embeddings.nrows() must equal num_chunks * num_speakers")]
  EmbeddingsRowMismatch,
  #[error("num_frames must be at least 1")]
  ZeroNumFrames,
  #[error("num_chunks * num_frames * num_speakers overflows usize")]
  SegmentationsOverflow,
  #[error("segmentations.len() must equal num_chunks * num_frames * num_speakers")]
  SegmentationsLenMismatch,
  #[error("train_chunk_idx and train_speaker_idx must have the same length")]
  TrainIndexLenMismatch,
  #[error("post_plda.nrows() must equal num_train")]
  PostPldaRowMismatch,
  #[error("post_plda must have at least one column (PLDA dimension)")]
  ZeroPldaDim,
  #[error("phi.len() must equal post_plda.ncols()")]
  PhiPldaDimMismatch,
  #[error("train_chunk_idx[i] out of range")]
  TrainChunkIdxOutOfRange,
  #[error("train_speaker_idx[i] out of range")]
  TrainSpeakerIdxOutOfRange,
  #[error("threshold must be a positive finite scalar")]
  InvalidThreshold,
  #[error("max_iters must be at least 1")]
  ZeroMaxIters,
}

/// Field that contained a non-finite value.
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum NonFiniteField {
  #[error("embeddings")]
  Embeddings,
  #[error("segmentations")]
  Segmentations,
}
