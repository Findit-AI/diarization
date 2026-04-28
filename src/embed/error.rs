//! Error type for `dia::embed`.

#[cfg(feature = "ort")]
use std::path::PathBuf;

use thiserror::Error;

/// Errors returned by `dia::embed` APIs.
#[derive(Debug, Error)]
pub enum Error {
  /// Input clip too short. Either `samples.len() < MIN_CLIP_SAMPLES`
  /// (for `embed`/`embed_weighted`) or the gathered length after
  /// applying a keep_mask in `embed_masked` was below the threshold.
  #[error("clip too short: {len} samples (need at least {min})")]
  InvalidClip { len: usize, min: usize },

  /// `voice_probs.len() != samples.len()` for `embed_weighted`.
  #[error("voice_probs.len() = {weights_len} must equal samples.len() = {samples_len}")]
  WeightShapeMismatch {
    samples_len: usize,
    weights_len: usize,
  },

  /// Rev-8: `keep_mask.len() != samples.len()` for `embed_masked`.
  #[error("keep_mask.len() = {mask_len} must equal samples.len() = {samples_len}")]
  MaskShapeMismatch { samples_len: usize, mask_len: usize },

  /// All windows had near-zero voice-probability weight; the weighted
  /// average is undefined. Almost always caller error.
  #[error("all windows had effectively zero voice-activity weight")]
  AllSilent,

  /// Input contains NaN or infinity.
  #[error("input contains non-finite values (NaN or infinity)")]
  NonFiniteInput,

  /// Input contains a zero-norm (or near-zero-norm, `< NORM_EPSILON`)
  /// embedding. Zero IS finite — kept distinct from `NonFiniteInput`
  /// so callers debugging real NaN/inf cases aren't misled.
  #[error("input contains a zero-norm or degenerate embedding")]
  DegenerateEmbedding,

  /// `kaldi-native-fbank` initialization failed with this message.
  /// `FbankComputer::new` returns `Result<Self, String>`; we wrap
  /// the message verbatim. This is effectively unreachable with our
  /// fixed configuration but kept as a fallible escape hatch in case
  /// a future kaldi-native-fbank version starts validating fields we
  /// currently rely on as no-ops.
  #[error("fbank computer initialization failed: {0}")]
  Fbank(String),

  /// ONNX inference output had an unexpected shape.
  #[error("inference scores length {got}, expected {expected}")]
  InferenceShapeMismatch { expected: usize, got: usize },

  /// Load-time model shape verification failed.
  #[cfg(feature = "ort")]
  #[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
  #[error("model {tensor} dims {got:?}, expected {expected:?}")]
  IncompatibleModel {
    tensor: &'static str,
    expected: &'static [i64],
    got: Vec<i64>,
  },

  /// Failed to load the ONNX model from disk.
  #[cfg(feature = "ort")]
  #[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
  #[error("failed to load model from {path}: {source}", path = path.display())]
  LoadModel {
    path: PathBuf,
    #[source]
    source: ort::Error,
  },

  /// Wrap an `ort::Error` from session/inference.
  #[cfg(feature = "ort")]
  #[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
  #[error(transparent)]
  Ort(#[from] ort::Error),
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn invalid_clip_message() {
    let e = Error::InvalidClip { len: 100, min: 400 };
    let s = format!("{e}");
    assert!(s.contains("100"));
    assert!(s.contains("400"));
  }

  #[test]
  fn mask_shape_mismatch_message() {
    let e = Error::MaskShapeMismatch {
      samples_len: 1000,
      mask_len: 999,
    };
    let s = format!("{e}");
    assert!(s.contains("1000"));
    assert!(s.contains("999"));
  }

  #[test]
  fn fbank_message() {
    let e = Error::Fbank("bad mel config".to_string());
    let s = format!("{e}");
    assert!(s.contains("fbank computer initialization failed"));
    assert!(s.contains("bad mel config"));
  }
}
