//! Error type for [`crate::diarizer::Diarizer`] APIs. Spec §4.4 / §5.12.

use thiserror::Error;

/// Errors returned by [`crate::diarizer::Diarizer`].
///
/// Wraps the per-module error types ([`crate::segment::Error`],
/// [`crate::embed::Error`], [`crate::cluster::Error`]) plus
/// diarizer-specific [`InternalError`] cases for integration-glue
/// invariant violations.
#[derive(Debug, Error)]
pub enum Error {
  /// Surface a [`crate::segment::Error`] (segmentation model failure,
  /// shape mismatch, etc.).
  #[error(transparent)]
  Segment(#[from] crate::segment::Error),

  /// Surface a [`crate::embed::Error`] (degenerate embedding, model
  /// load failure, fbank initialization, etc.).
  #[error(transparent)]
  Embed(#[from] crate::embed::Error),

  /// Surface a [`crate::cluster::Error`] (online speaker cap reached
  /// with `Reject`, etc.).
  #[error(transparent)]
  Cluster(#[from] crate::cluster::Error),

  /// An invariant of the Diarizer's internal state was violated.
  /// Distinct from the wrapped variants because those wrap errors
  /// FROM the underlying state machines; `Internal` covers the
  /// integration glue itself (audio buffer indexing, activity range
  /// reconciliation). Almost certainly a bug in dia or a pathological
  /// caller (e.g., a custom mid-level composition supplying out-of-
  /// order activities).
  #[error(transparent)]
  Internal(#[from] InternalError),

  /// The Diarizer encountered a non-recoverable error in a prior call
  /// (e.g., an embedding failure mid-window). Subsequent
  /// `process_samples` and `finish_stream` calls return this until
  /// [`Diarizer::clear`](crate::diarizer::Diarizer::clear) resets the
  /// state.
  ///
  /// This guards against silent partial-state corruption when a window's
  /// activities are partially embedded/clustered before an error
  /// propagates: rather than continue with an inconsistent reconstruction
  /// state, the Diarizer refuses further work until the caller explicitly
  /// recovers via `clear()`.
  ///
  /// Codex review HIGH.
  #[error("Diarizer is poisoned by a prior error; call clear() to reset")]
  Poisoned,

  /// [`finish_stream`](crate::diarizer::Diarizer::finish_stream) has
  /// already completed for this session. Once finished, the audio
  /// buffer is locked: further [`process_samples`] / [`finish_stream`]
  /// calls return this error until [`clear`](crate::diarizer::Diarizer::clear)
  /// starts a new session.
  ///
  /// Without this terminal-state check, [`process_samples`] would still
  /// append to the audio buffer and increment the public sample
  /// counter, but the inner [`Segmenter`](crate::segment::Segmenter)
  /// silently drops post-finish samples — they would never reach
  /// segmentation, producing a counter/segmenter divergence with no
  /// surfaced error. Codex review HIGH.
  ///
  /// [`process_samples`]: crate::diarizer::Diarizer::process_samples
  /// [`finish_stream`]: crate::diarizer::Diarizer::finish_stream
  #[error("Diarizer stream is already finished; call clear() to start a new session")]
  Finished,
}

/// Internal Diarizer invariant violations. Surfaced by
/// [`Error::Internal`].
///
/// These represent integration-glue inconsistencies (audio buffer
/// underflow / overrun) that should be unreachable per the §5.7
/// segment-contract verification. Defense-in-depth.
///
/// `#[non_exhaustive]` — a v0.X bump may add new variants. Match
/// exhaustively only inside `dia::diarizer`; downstream callers
/// should use a catch-all `_` arm.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum InternalError {
  /// An emitted activity's range start is older than the audio
  /// buffer's earliest retained sample. Should never fire; defensive.
  #[error("activity range start {activity_start} is below audio buffer base {audio_base} (delta = {} samples)", audio_base - activity_start)]
  AudioBufferUnderflow {
    /// Start of the activity range (absolute samples).
    activity_start: u64,
    /// Earliest retained sample in the audio buffer.
    audio_base: u64,
  },

  /// An emitted activity's range end exceeds the audio buffer's
  /// latest sample.
  #[error("activity range end {activity_end} exceeds audio buffer end {audio_end}")]
  AudioBufferOverrun {
    /// End of the activity range (absolute samples).
    activity_end: u64,
    /// Latest sample in the audio buffer (`audio_base + buffered_samples`).
    audio_end: u64,
  },
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn underflow_message_includes_delta() {
    let e = InternalError::AudioBufferUnderflow {
      activity_start: 100,
      audio_base: 500,
    };
    let s = format!("{e}");
    assert!(s.contains("100"));
    assert!(s.contains("500"));
    assert!(s.contains("400"), "delta 500-100=400 must appear: {s}");
  }

  #[test]
  fn overrun_message_format() {
    let e = InternalError::AudioBufferOverrun {
      activity_end: 1500,
      audio_end: 1000,
    };
    let s = format!("{e}");
    assert!(s.contains("1500"));
    assert!(s.contains("1000"));
  }

  #[test]
  fn from_segment_error_compiles() {
    // Verifies #[from] works for crate::segment::Error.
    fn _accepts(e: crate::segment::Error) -> Error {
      e.into()
    }
  }

  #[test]
  fn from_embed_error_compiles() {
    fn _accepts(e: crate::embed::Error) -> Error {
      e.into()
    }
  }

  #[test]
  fn from_cluster_error_compiles() {
    fn _accepts(e: crate::cluster::Error) -> Error {
      e.into()
    }
  }
}
