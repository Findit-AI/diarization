//! Error type for [`crate::diarizer::Diarizer`] APIs. Spec §4.4 / §5.12.

use thiserror::Error;

/// Errors returned by [`crate::diarizer::Diarizer`].
///
/// Wraps the per-module error types ([`crate::segment::Error`],
/// [`crate::embed::Error`], [`crate::cluster::Error`]) plus
/// diarizer-specific [`InternalError`] cases for state-machine
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

  /// Internal Diarizer invariant violation. These should be
  /// unreachable in practice; if you see one, file a bug.
  #[error(transparent)]
  Internal(#[from] InternalError),
}

/// Internal Diarizer invariant violations. Surfaced by
/// [`Error::Internal`].
///
/// These represent state-machine inconsistencies that should be
/// unreachable on well-formed input. The variant names are public
/// but stable; a v0.X bump may add new variants. Match exhaustively
/// only inside `dia::diarizer`; downstream callers should use a
/// catch-all `_` arm.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum InternalError {
  /// `push_inference` called with a `WindowId` not pending in the
  /// segmenter — typically a caller bug (mixing two sessions or
  /// stale ids).
  #[error("unknown WindowId from upstream segmenter")]
  UnknownWindow,

  /// Per-frame reconstruction received scores for a window already
  /// finalized. Indicates an out-of-order push_inference or a
  /// double-finalize bug.
  #[error("inference scores arrived after window already finalized")]
  LateScores,

  /// `process_samples` extracted an embedding clip that, after
  /// gathering by keep_mask, was below the embed::MIN_CLIP_SAMPLES
  /// threshold AND the fallback path also failed.
  ///
  /// (This is distinct from `embed::Error::InvalidClip` which fires
  /// at the embed layer; this variant is a reconstruction-layer
  /// guard that we should always be able to handle gracefully.)
  #[error("embedding clip too short after applying both clean and fallback masks")]
  ClipTooShortAfterMasks,

  /// Reconstruction state machine reached an impossible state
  /// (e.g., negative chunk count). Indicates a Diarizer bug.
  #[error("reconstruction state machine reached an unreachable state: {0}")]
  Unreachable(&'static str),
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn segment_error_round_trip() {
    let e: Error = crate::segment::Error::InferenceShapeMismatch {
      expected: 4123,
      got: 100,
    }
    .into();
    let s = format!("{e}");
    assert!(s.contains("4123") || s.contains("100"));
  }

  #[test]
  fn internal_unknown_window() {
    let e: Error = InternalError::UnknownWindow.into();
    let s = format!("{e}");
    assert!(s.contains("WindowId") || s.contains("segmenter"));
  }

  #[test]
  fn internal_unreachable_carries_message() {
    let e = InternalError::Unreachable("test marker");
    let s = format!("{e}");
    assert!(s.contains("test marker"));
  }
}
