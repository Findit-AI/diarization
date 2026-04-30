//! Error type for the segmentation module.

#[cfg(feature = "ort")]
use std::path::PathBuf;

use thiserror::Error;

use crate::segment::types::WindowId;

/// All errors produced by `diarization::segment`.
#[derive(Debug, Error)]
pub enum Error {
  /// Construction-time validation failure for [`SegmentOptions`].
  ///
  /// Reserved for future eager validation; not currently emitted by
  /// v0.1.0 (which stores option values verbatim).
  ///
  /// [`SegmentOptions`]: crate::segment::SegmentOptions
  #[error("invalid segment options: {0}")]
  InvalidOptions(&'static str),

  /// `push_inference` received a `scores` slice of the wrong length.
  ///
  /// Expected length is [`FRAMES_PER_WINDOW`] × [`POWERSET_CLASSES`] = 4123.
  ///
  /// [`FRAMES_PER_WINDOW`]: crate::segment::FRAMES_PER_WINDOW
  /// [`POWERSET_CLASSES`]: crate::segment::POWERSET_CLASSES
  #[error("inference scores length {got}, expected {expected}")]
  InferenceShapeMismatch {
    /// Expected element count.
    expected: usize,
    /// Actual length received.
    got: usize,
  },

  /// `push_inference` was called with a [`WindowId`] that is not in the
  /// pending set.
  ///
  /// See [`Segmenter::push_inference`] rustdoc for the four scenarios this
  /// covers (never-yielded, already-consumed, stale-after-`clear`,
  /// cross-segmenter-collision).
  ///
  /// [`Segmenter::push_inference`]: crate::segment::Segmenter::push_inference
  #[error("inference scores received for unknown WindowId {id:?}")]
  UnknownWindow {
    /// The unknown id.
    id: WindowId,
  },

  /// `push_inference` received a `scores` slice containing one or more
  /// non-finite values (`NaN`, `+inf`, or `-inf`).
  ///
  /// The [`WindowId`] is left in the pending set so the caller can
  /// re-run inference (e.g. retry on a transient backend failure that
  /// produced bad logits) without losing the window.
  #[error("inference scores for WindowId {id:?} contain non-finite values")]
  NonFiniteScores {
    /// The window whose scores were rejected. Still pending; safe to
    /// retry `push_inference` after producing valid logits.
    id: WindowId,
  },

  /// A loaded ONNX model's input or output dimensions don't match what
  /// `diarization::segment` expects (`[*, 1, 160000]` for input, `[*, 589, 7]` for
  /// output, where `*` is a free batch dimension).
  #[cfg(feature = "ort")]
  #[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
  #[error("model {tensor} dims {got:?}, expected {expected:?}")]
  IncompatibleModel {
    /// Which tensor (`"input"` or `"output"`).
    tensor: &'static str,
    /// Expected dimension list. `-1` indicates a dynamic dimension.
    expected: &'static [i64],
    /// Actual dimensions reported by the loaded model.
    got: Vec<i64>,
  },

  /// The `ort::Session` failed to load the model file.
  #[cfg(feature = "ort")]
  #[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
  #[error("failed to load model from {path}: {source}", path = path.display())]
  LoadModel {
    /// Path passed to `from_file`.
    path: PathBuf,
    /// Underlying ort error.
    #[source]
    source: ort::Error,
  },

  /// Generic ort runtime error from `SegmentModel::infer` or session ops.
  #[cfg(feature = "ort")]
  #[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
  #[error(transparent)]
  Ort(#[from] ort::Error),
}
