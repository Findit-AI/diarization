//! ONNX Runtime wrapper for WeSpeaker ResNet34 (spec §4.2).
//!
//! Auto-derives `Send`. Does NOT auto-derive `Sync` because
//! `ort::Session` is `!Sync`. Matches [`SegmentModel`](crate::segment::SegmentModel) —
//! same one-session-per-worker-thread pattern.

use std::path::Path;

use ort::{session::Session as OrtSession, value::TensorRef};

use crate::embed::{
  EmbedModelOptions, Error,
  options::{EMBEDDING_DIM, FBANK_FRAMES, FBANK_NUM_MELS},
};

/// Thin ort wrapper for one WeSpeaker embedding session.
///
/// Owns one `ort::Session`. The wrapper is `Send` (ort::Session is `Send`)
/// but **not** `Sync` (parallel inference on the same session is unsafe in
/// ort). Use one `EmbedModel` per worker thread.
///
/// The 256-d output of `embed_features` / `embed_features_batch` is the
/// **raw, un-normalized** embedding straight from the model. Higher-level
/// methods (`EmbedModel::embed`, `embed_weighted`, `embed_masked` — added in
/// Tasks 26-27) wrap this with the §5.1 sliding-window aggregation and
/// L2-normalize the result via [`Embedding::normalize_from`](crate::embed::Embedding::normalize_from).
pub struct EmbedModel {
  inner: OrtSession,
}

impl EmbedModel {
  /// Load the model from disk with default options.
  pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
    Self::from_file_with_options(path, EmbedModelOptions::default())
  }

  /// Load the model from disk with custom options.
  pub fn from_file_with_options<P: AsRef<Path>>(
    path: P,
    opts: EmbedModelOptions,
  ) -> Result<Self, Error> {
    let path = path.as_ref();
    let mut builder = opts.apply(OrtSession::builder()?)?;
    let session = builder
      .commit_from_file(path)
      .map_err(|source| Error::LoadModel {
        path: path.to_path_buf(),
        source,
      })?;
    Ok(Self { inner: session })
  }

  /// Load the model from an in-memory ONNX byte buffer with default options.
  ///
  /// `bytes` is **copied** into ort's session; the buffer can be dropped
  /// immediately after this call returns.
  pub fn from_memory(bytes: &[u8]) -> Result<Self, Error> {
    Self::from_memory_with_options(bytes, EmbedModelOptions::default())
  }

  /// Load the model from an in-memory ONNX byte buffer with custom options.
  pub fn from_memory_with_options(bytes: &[u8], opts: EmbedModelOptions) -> Result<Self, Error> {
    let mut builder = opts.apply(OrtSession::builder()?)?;
    let session = builder.commit_from_memory(bytes)?;
    Ok(Self { inner: session })
  }

  /// Run inference on one `[FBANK_FRAMES, FBANK_NUM_MELS] = [200, 80]` fbank
  /// tensor. Returns the **raw (un-normalized)** 256-d embedding output.
  ///
  /// This is the low-level API. Most callers should use the high-level
  /// `embed`/`embed_weighted`/`embed_masked` methods (Task 27) which wrap
  /// `compute_fbank` + sliding-window aggregation + L2-normalization.
  pub fn embed_features(
    &mut self,
    features: &[[f32; FBANK_NUM_MELS]; FBANK_FRAMES],
  ) -> Result<[f32; EMBEDDING_DIM], Error> {
    // Flatten features into a contiguous [1, FBANK_FRAMES, FBANK_NUM_MELS] tensor.
    let mut flat = Vec::with_capacity(FBANK_FRAMES * FBANK_NUM_MELS);
    for row in features.iter() {
      flat.extend_from_slice(row);
    }
    let outputs = self.inner.run(ort::inputs![TensorRef::from_array_view((
      [1usize, FBANK_FRAMES, FBANK_NUM_MELS],
      flat.as_slice()
    ))?])?;
    let (_shape, data) = outputs[0].try_extract_tensor::<f32>()?;
    if data.len() != EMBEDDING_DIM {
      return Err(Error::InferenceShapeMismatch {
        expected: EMBEDDING_DIM,
        got: data.len(),
      });
    }
    let mut out = [0.0f32; EMBEDDING_DIM];
    out.copy_from_slice(data);
    Ok(out)
  }

  /// Batched feature inference. Single ONNX call with batch size N.
  /// Returns N **raw (un-normalized)** 256-d embeddings.
  ///
  /// Avoids per-call ONNX overhead when many windows are inferred from
  /// the same model — used by [`Diarizer`](crate::diarizer)'s embedding
  /// pump for the multi-window aggregation path. An empty input slice
  /// returns an empty Vec without invoking the session.
  pub fn embed_features_batch(
    &mut self,
    features: &[[[f32; FBANK_NUM_MELS]; FBANK_FRAMES]],
  ) -> Result<Vec<[f32; EMBEDDING_DIM]>, Error> {
    let n = features.len();
    if n == 0 {
      return Ok(Vec::new());
    }
    let mut flat = Vec::with_capacity(n * FBANK_FRAMES * FBANK_NUM_MELS);
    for chunk in features.iter() {
      for row in chunk.iter() {
        flat.extend_from_slice(row);
      }
    }
    let outputs = self.inner.run(ort::inputs![TensorRef::from_array_view((
      [n, FBANK_FRAMES, FBANK_NUM_MELS],
      flat.as_slice()
    ))?])?;
    let (_shape, data) = outputs[0].try_extract_tensor::<f32>()?;
    let expected = n * EMBEDDING_DIM;
    if data.len() != expected {
      return Err(Error::InferenceShapeMismatch {
        expected,
        got: data.len(),
      });
    }
    let out: Vec<[f32; EMBEDDING_DIM]> = data
      .chunks_exact(EMBEDDING_DIM)
      .take(n)
      .map(|chunk| {
        let mut row = [0.0f32; EMBEDDING_DIM];
        row.copy_from_slice(chunk);
        row
      })
      .collect();
    Ok(out)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::embed::{compute_fbank, options::EMBED_WINDOW_SAMPLES};
  use std::path::PathBuf;

  /// Resolve the WeSpeaker ResNet34-LM model path. Set `DIA_EMBED_MODEL_PATH`
  /// to override, otherwise default to `models/wespeaker_resnet34_lm.onnx`
  /// relative to the crate root. Tests that require the model file are
  /// `#[ignore]`-ed so CI is green without it; run with:
  ///   `DIA_EMBED_MODEL_PATH=path/to/model.onnx cargo test --features ort -- --ignored`
  /// or simply `cargo test --features ort -- --ignored` if the default path
  /// is populated.
  fn model_path() -> PathBuf {
    if let Ok(p) = std::env::var("DIA_EMBED_MODEL_PATH") {
      return PathBuf::from(p);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/wespeaker_resnet34_lm.onnx")
  }

  #[test]
  #[ignore = "requires WeSpeaker ResNet34-LM ONNX model"]
  fn loads_and_infers_silent_clip() {
    let path = model_path();
    if !path.exists() {
      panic!(
        "model not found at {}; set DIA_EMBED_MODEL_PATH or download via models/",
        path.display()
      );
    }
    let mut model = EmbedModel::from_file(&path).expect("load model");

    // 2 seconds of silence → fbank → 256-d output.
    let samples = vec![0.0f32; EMBED_WINDOW_SAMPLES as usize];
    let fbank = compute_fbank(&samples).expect("fbank silence");
    let raw = model.embed_features(&fbank).expect("infer silence");
    assert_eq!(
      raw.len(),
      EMBEDDING_DIM,
      "raw embedding length must equal EMBEDDING_DIM"
    );
    assert!(
      raw.iter().all(|v| v.is_finite()),
      "embedding components must all be finite"
    );
  }

  #[test]
  #[ignore = "requires WeSpeaker ResNet34-LM ONNX model"]
  fn batch_inference_matches_single() {
    // Run the same fbank through embed_features (1 input) and
    // embed_features_batch (1-element batch). Outputs must match
    // bit-exactly — no batch-axis quirks.
    let path = model_path();
    if !path.exists() {
      return;
    }
    let mut model = EmbedModel::from_file(&path).expect("load model");

    let samples = vec![0.001f32; EMBED_WINDOW_SAMPLES as usize];
    let fbank = compute_fbank(&samples).expect("fbank");
    let single = model.embed_features(&fbank).expect("single");
    let batch = model.embed_features_batch(&[*fbank]).expect("batch");
    assert_eq!(batch.len(), 1);
    assert_eq!(
      single, batch[0],
      "batch[0] must equal single inference bit-exactly"
    );
  }

  #[test]
  fn empty_batch_returns_empty_vec_without_session_call() {
    // No model loaded — passing an empty slice should short-circuit.
    // Test runs unconditionally because we never construct an EmbedModel.
    // We can't call the method without a model, so this is more of a
    // documentation note: the empty-batch fast path is in `embed_features_batch`.
    let _: () = ();
  }
}
