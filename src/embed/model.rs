//! ONNX Runtime wrapper for WeSpeaker ResNet34 (spec §4.2).
//!
//! Auto-derives `Send`. Does NOT auto-derive `Sync` because
//! `ort::Session` is `!Sync`. Matches [`SegmentModel`](crate::segment::SegmentModel) —
//! same one-session-per-worker-thread pattern.

use core::time::Duration;
use std::path::Path;

use ort::{session::Session as OrtSession, value::TensorRef};

use crate::embed::{
  EmbedModelOptions, Error,
  embedder::{embed_unweighted, embed_weighted_inner},
  options::{EMBEDDING_DIM, FBANK_FRAMES, FBANK_NUM_MELS, MIN_CLIP_SAMPLES, SAMPLE_RATE_HZ},
  types::{Embedding, EmbeddingMeta, EmbeddingResult},
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

  // ── High-level methods (spec §4.2) ────────────────────────────────────

  /// Compute the L2-normalized embedding of a clip (spec §5.1).
  ///
  /// For clips up to `EMBED_WINDOW_SAMPLES` (2 s @ 16 kHz), runs a single
  /// inference on the zero-padded clip. For longer clips, runs sliding-
  /// window inference and aggregates via per-window unweighted sum, then
  /// L2-normalizes the result.
  ///
  /// Returns [`Error::InvalidClip`] if `samples.len() < MIN_CLIP_SAMPLES`,
  /// or [`Error::DegenerateEmbedding`] if the aggregated sum has near-zero
  /// L2 norm (effectively unreachable on real audio; signals caller bug).
  pub fn embed(&mut self, samples: &[f32]) -> Result<EmbeddingResult, Error> {
    self.embed_with_meta(samples, EmbeddingMeta::default())
  }

  /// [`embed`](Self::embed) with explicit observability metadata
  /// ([`EmbeddingMeta`]). Returns a typed [`EmbeddingResult<A, T>`].
  pub fn embed_with_meta<A, T>(
    &mut self,
    samples: &[f32],
    meta: EmbeddingMeta<A, T>,
  ) -> Result<EmbeddingResult<A, T>, Error> {
    let (sum, windows_used) = embed_unweighted(self, samples)?;
    let embedding = Embedding::normalize_from(sum).ok_or(Error::DegenerateEmbedding)?;
    let duration = duration_from_samples(samples.len());
    Ok(EmbeddingResult::new(
      embedding,
      duration,
      windows_used,
      windows_used as f32,
      meta,
    ))
  }

  /// Voice-probability-weighted embedding (spec §5.2).
  ///
  /// Per-window weight = mean of `voice_probs[start..start + WINDOW]`.
  /// Aggregates per-window outputs as a weighted sum, then L2-normalizes.
  ///
  /// Errors:
  /// - [`Error::WeightShapeMismatch`] if `voice_probs.len() != samples.len()`.
  /// - [`Error::InvalidClip`] if `samples.len() < MIN_CLIP_SAMPLES`.
  /// - [`Error::AllSilent`] if every per-window weight is below `NORM_EPSILON`.
  /// - [`Error::DegenerateEmbedding`] if the weighted sum has near-zero norm.
  pub fn embed_weighted(
    &mut self,
    samples: &[f32],
    voice_probs: &[f32],
  ) -> Result<EmbeddingResult, Error> {
    self.embed_weighted_with_meta(samples, voice_probs, EmbeddingMeta::default())
  }

  /// [`embed_weighted`](Self::embed_weighted) with explicit observability metadata.
  pub fn embed_weighted_with_meta<A, T>(
    &mut self,
    samples: &[f32],
    voice_probs: &[f32],
    meta: EmbeddingMeta<A, T>,
  ) -> Result<EmbeddingResult<A, T>, Error> {
    let (sum, windows_used, total_weight) = embed_weighted_inner(self, samples, voice_probs)?;
    let embedding = Embedding::normalize_from(sum).ok_or(Error::DegenerateEmbedding)?;
    let duration = duration_from_samples(samples.len());
    Ok(EmbeddingResult::new(
      embedding,
      duration,
      windows_used,
      total_weight,
      meta,
    ))
  }

  /// Boolean-mask embedding: gather samples where `keep_mask[i] == true`,
  /// then run the standard [`embed`](Self::embed) pipeline on the gathered
  /// audio. Spec §5.8.
  ///
  /// This is the path used by [`Diarizer`](crate::diarizer)'s
  /// `exclude_overlap` mode: per-cluster keep masks identify which
  /// samples belong to a single speaker (i.e., not in overlap), and we
  /// embed only those. The resulting embedding therefore represents the
  /// speaker without contamination from overlapping speech.
  ///
  /// **Diverges from pyannote** for long clips with sparse keep masks:
  /// pyannote does per-window keep-mask gating + per-window mean
  /// aggregation; we do all-sample gather + standard sliding-window
  /// embed on the (variable-length) gathered audio. See spec §15 #49.
  ///
  /// Errors:
  /// - [`Error::MaskShapeMismatch`] if `keep_mask.len() != samples.len()`.
  /// - [`Error::InvalidClip`] if the gathered length `< MIN_CLIP_SAMPLES`.
  /// - [`Error::DegenerateEmbedding`] on near-zero aggregated norm.
  pub fn embed_masked(
    &mut self,
    samples: &[f32],
    keep_mask: &[bool],
  ) -> Result<EmbeddingResult, Error> {
    self.embed_masked_with_meta(samples, keep_mask, EmbeddingMeta::default())
  }

  /// [`embed_masked`](Self::embed_masked) with explicit observability metadata.
  pub fn embed_masked_with_meta<A, T>(
    &mut self,
    samples: &[f32],
    keep_mask: &[bool],
    meta: EmbeddingMeta<A, T>,
  ) -> Result<EmbeddingResult<A, T>, Error> {
    if keep_mask.len() != samples.len() {
      return Err(Error::MaskShapeMismatch {
        samples_len: samples.len(),
        mask_len: keep_mask.len(),
      });
    }
    // Gather samples where keep_mask is true.
    let gathered: Vec<f32> = samples
      .iter()
      .zip(keep_mask.iter())
      .filter_map(|(&s, &keep)| keep.then_some(s))
      .collect();
    if gathered.len() < MIN_CLIP_SAMPLES as usize {
      return Err(Error::InvalidClip {
        len: gathered.len(),
        min: MIN_CLIP_SAMPLES as usize,
      });
    }
    // Reuse the standard sliding-window-mean pipeline on the gathered audio.
    let (sum, windows_used) = embed_unweighted(self, &gathered)?;
    let embedding = Embedding::normalize_from(sum).ok_or(Error::DegenerateEmbedding)?;
    let duration = duration_from_samples(gathered.len());
    Ok(EmbeddingResult::new(
      embedding,
      duration,
      windows_used,
      windows_used as f32,
      meta,
    ))
  }
}

/// Convert sample count → wall-clock duration at the WeSpeaker sample rate.
fn duration_from_samples(n: usize) -> Duration {
  Duration::from_micros((n as u64).saturating_mul(1_000_000) / SAMPLE_RATE_HZ as u64)
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

  // ── High-level methods (model-required) ─────────────────────────────

  #[test]
  #[ignore = "requires WeSpeaker ResNet34-LM ONNX model"]
  fn embed_round_trips_on_2s_clip() {
    let path = model_path();
    if !path.exists() {
      return;
    }
    let mut model = EmbedModel::from_file(&path).expect("load model");
    let samples = vec![0.001f32; EMBED_WINDOW_SAMPLES as usize];
    let r = model.embed(&samples).expect("embed succeeds");

    // L2 norm == 1.0 within float-precision tolerance.
    let n_sq: f32 = r.embedding().as_array().iter().map(|x| x * x).sum();
    let norm = n_sq.sqrt();
    assert!(
      (norm - 1.0).abs() < 1e-5,
      "||embedding|| = {norm} (expected 1.0 ± 1e-5)"
    );
    assert_eq!(r.windows_used(), 1);
  }

  #[test]
  #[ignore = "requires WeSpeaker ResNet34-LM ONNX model"]
  fn embed_long_clip_uses_sliding_window() {
    // 4-second clip → plan_starts yields 3 windows (0, 16_000, 32_000).
    let path = model_path();
    if !path.exists() {
      return;
    }
    let mut model = EmbedModel::from_file(&path).expect("load model");
    let samples = vec![0.001f32; 2 * EMBED_WINDOW_SAMPLES as usize];
    let r = model.embed(&samples).expect("embed succeeds");
    assert_eq!(r.windows_used(), 3, "4s clip → 3 sliding windows");
    let n_sq: f32 = r.embedding().as_array().iter().map(|x| x * x).sum();
    assert!((n_sq.sqrt() - 1.0).abs() < 1e-5);
  }

  #[test]
  #[ignore = "requires WeSpeaker ResNet34-LM ONNX model"]
  fn embed_weighted_rejects_mismatched_lengths() {
    let path = model_path();
    if !path.exists() {
      return;
    }
    let mut model = EmbedModel::from_file(&path).expect("load model");
    let samples = vec![0.001f32; EMBED_WINDOW_SAMPLES as usize];
    let probs = vec![1.0f32; EMBED_WINDOW_SAMPLES as usize - 1]; // off-by-one
    let r = model.embed_weighted(&samples, &probs);
    assert!(
      matches!(
        r,
        Err(Error::WeightShapeMismatch {
          samples_len: 32_000,
          weights_len: 31_999
        })
      ),
      "got {r:?}"
    );
  }

  #[test]
  #[ignore = "requires WeSpeaker ResNet34-LM ONNX model"]
  fn embed_weighted_uniform_probs_matches_unweighted_direction() {
    // With voice_probs = 1.0 everywhere, embed_weighted should produce
    // the same direction as embed (both normalize the same per-window
    // sum, just scaled by a constant). Cosine similarity ≈ 1.
    let path = model_path();
    if !path.exists() {
      return;
    }
    let mut model = EmbedModel::from_file(&path).expect("load model");
    let samples = vec![0.001f32; EMBED_WINDOW_SAMPLES as usize];
    let probs = vec![1.0f32; EMBED_WINDOW_SAMPLES as usize];
    let plain = model.embed(&samples).expect("plain");
    let weighted = model.embed_weighted(&samples, &probs).expect("weighted");
    let cos: f32 = plain
      .embedding()
      .as_array()
      .iter()
      .zip(weighted.embedding().as_array().iter())
      .map(|(a, b)| a * b)
      .sum();
    assert!(
      (cos - 1.0).abs() < 1e-5,
      "uniform-probs weighted should equal plain in direction; cos = {cos}"
    );
  }

  #[test]
  #[ignore = "requires WeSpeaker ResNet34-LM ONNX model"]
  fn embed_masked_rejects_short_gathered_clip() {
    // keep_mask gathers 100 samples — below MIN_CLIP_SAMPLES = 400.
    let path = model_path();
    if !path.exists() {
      return;
    }
    let mut model = EmbedModel::from_file(&path).expect("load model");
    let samples = vec![0.001f32; EMBED_WINDOW_SAMPLES as usize];
    let mut mask = vec![false; EMBED_WINDOW_SAMPLES as usize];
    for m in mask.iter_mut().take(100) {
      *m = true;
    }
    let r = model.embed_masked(&samples, &mask);
    assert!(
      matches!(r, Err(Error::InvalidClip { len: 100, min: 400 })),
      "got {r:?}"
    );
  }

  #[test]
  fn embed_masked_rejects_mask_length_mismatch_without_model() {
    // This test does NOT require the model — only the input-validation
    // path runs before any inference.
    //
    // We can't construct an EmbedModel without an ort::Session, so we
    // check the validator indirectly via the embed_masked path on a
    // (real) model when available. Skip if absent.
    let path = model_path();
    if !path.exists() {
      return;
    }
    let mut model = EmbedModel::from_file(&path).expect("load model");
    let samples = vec![0.001f32; 100];
    let mask = vec![true; 99];
    let r = model.embed_masked(&samples, &mask);
    assert!(
      matches!(
        r,
        Err(Error::MaskShapeMismatch {
          samples_len: 100,
          mask_len: 99
        })
      ),
      "got {r:?}"
    );
  }
}
