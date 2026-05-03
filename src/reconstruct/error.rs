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
#[derive(Debug, Error, Clone, Copy, PartialEq)]
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
  #[error("num_chunks * num_frames_per_chunk * num_speakers overflows usize")]
  SegmentationsSizeOverflow,
  #[error("num_chunks * num_frames_per_chunk * num_clusters overflows usize")]
  ClusteredSizeOverflow,
  #[error("num_output_frames * num_clusters overflows usize")]
  OutputGridSizeOverflow,
  #[error(
    "hard_clusters[c] has a non-UNMATCHED id in a slot beyond num_speakers; \
     slots num_speakers..MAX_SPEAKER_SLOTS must all be UNMATCHED"
  )]
  HardClustersTrailingSlotNotUnmatched,
  #[error("grid.len() must equal num_frames * num_clusters")]
  GridLenMismatch,
  #[error("num_frames * num_clusters overflows usize")]
  GridSizeOverflow,
  /// `smoothing_epsilon` is `Some(NaN/±inf)` or `Some(< 0)`. The
  /// per-frame top-k pass compares activation differences against
  /// this epsilon; `Some(+inf)` collapses every comparison
  /// (every pair is "within epsilon"), making selection fall back
  /// to stable cluster index order rather than activation order.
  /// `Some(NaN)` makes every comparison false. `None` is the bit-
  /// exact pyannote argmax path and is always valid.
  ///
  /// Mirrors the same predicate the offline / streaming entrypoints
  /// enforce, but checked at the lower-level `reconstruct` boundary
  /// so direct callers cannot bypass it.
  #[error("smoothing_epsilon ({value:?}) must be None or Some(finite >= 0)")]
  SmoothingEpsilonOutOfRange { value: Option<f32> },
  /// `min_duration_off` is NaN/±inf or negative. RTTM span-merge
  /// reads this as a non-negative seconds quantity; `+inf` merges
  /// every same-cluster gap, `NaN` silently disables merging
  /// (every comparison becomes false), and negative values are
  /// nonsensical. Catches direct callers of [`try_discrete_to_spans`]
  /// that bypass the offline-entrypoint validation.
  ///
  /// [`try_discrete_to_spans`]: crate::reconstruct::try_discrete_to_spans
  #[error("min_duration_off ({value}) must be finite and >= 0")]
  MinDurationOffOutOfRange { value: f64 },
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
