//! Errors for `diarization::reconstruct`.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
  #[error("reconstruct: shape error: {0}")]
  Shape(#[from] ShapeError),
  #[error("reconstruct: non-finite value in {0}")]
  NonFinite(#[from] NonFiniteField),
  #[error("reconstruct: invalid sliding-window timing: {0}")]
  Timing(#[from] TimingError),
}

/// Specific shape-violation reasons for [`Error::Shape`].
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum ShapeError {
  #[error("num_chunks must be at least 1")]
  ZeroNumChunks,
  #[error("num_frames_per_chunk must be at least 1")]
  ZeroNumFramesPerChunk,
  #[error("num_speakers must be at least 1")]
  ZeroNumSpeakers,
  #[error("num_speakers must be <= MAX_SPEAKER_SLOTS (3)")]
  TooManySpeakers,
  #[error("segmentations.len() != num_chunks * num_frames_per_chunk * num_speakers")]
  SegmentationsLenMismatch,
  #[error("hard_clusters.len() != num_chunks")]
  HardClustersLenMismatch,
  #[error("num_output_frames must be at least 1")]
  ZeroNumOutputFrames,
  #[error("count.len() != num_output_frames")]
  CountLenMismatch,
  #[error("count entry exceeds MAX_COUNT_PER_FRAME (64)")]
  CountAboveMax,
  #[error("hard_clusters contains a negative id other than UNMATCHED")]
  HardClustersNegativeId,
  #[error("hard_clusters id exceeds MAX_CLUSTER_ID (1023)")]
  HardClustersIdAboveMax,
}

/// Field that contained a non-finite value.
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum NonFiniteField {
  #[error("segmentations")]
  Segmentations,
}

/// Specific timing-validation failures for [`Error::Timing`].
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum TimingError {
  #[error("non-finite sliding-window parameter")]
  NonFiniteParameter,
  #[error("non-positive duration or step")]
  NonPositiveDurationOrStep,
}
