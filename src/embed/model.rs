//! WeSpeaker ResNet34 embedding inference (spec §4.2).
//!
//! Multi-backend wrapper. The same `EmbedModel` API supports two
//! inference engines:
//!
//! - **ONNX (default)**: pulls in `ort` (ONNX Runtime). Fast, no
//!   dynamic linking. Constructed via [`EmbedModel::from_file`] /
//!   [`EmbedModel::from_memory`].
//! - **TorchScript** (feature `tch`): pulls in `tch` (libtorch C++
//!   bindings). Heavier (libtorch shared lib at runtime) but matches
//!   pyannote's PyTorch inference bit-exactly on hard cases — useful
//!   when ONNX→ORT diverges from PyTorch numerically. Constructed via
//!   [`EmbedModel::from_torchscript_file`].
//!
//! `Send` but **not** `Sync` (single-session-per-thread for both ort
//! and tch). Matches [`SegmentModel`](crate::segment::SegmentModel).
//!
//! The 256-d output of `embed_features` / `embed_features_batch` is
//! the **raw, un-normalized** embedding straight from the model.
//! Higher-level methods (`embed`, `embed_weighted`, `embed_masked`)
//! wrap this with the §5.1 sliding-window aggregation and L2-normalize
//! the result via [`Embedding::normalize_from`].

use core::time::Duration;
use std::path::Path;

use crate::embed::{
  Error,
  embedder::{embed_unweighted, embed_weighted_inner},
  options::{EMBEDDING_DIM, FBANK_FRAMES, FBANK_NUM_MELS, MIN_CLIP_SAMPLES, SAMPLE_RATE_HZ},
  types::{Embedding, EmbeddingMeta, EmbeddingResult},
};

#[cfg(feature = "ort")]
use crate::embed::EmbedModelOptions;

// ── Backend trait ───────────────────────────────────────────────────

/// Backend-agnostic interface for embedding inference.
///
/// Implementations: `OrtBackend` (ONNX via ort), `TchBackend`
/// (TorchScript via tch). Both produce raw, un-normalized 256-d
/// embeddings.
///
/// Two methods cover the two pyannote use cases:
///
/// 1. [`embed_audio_clips_batch`] — bare audio clips (no mask). Used
///    by the high-level `embed`, `embed_weighted`, `embed_masked`
///    helpers for variable-length clips with sliding-window
///    aggregation.
/// 2. [`embed_chunk_with_frame_mask`] — pyannote-style 10s chunk +
///    589-frame segmentation mask. The mask is interpreted as
///    pooling weights: the WeSpeaker statistics-pooling layer
///    ignores frames with zero weight. This is the call that
///    [`crate::offline::OwnedDiarizationPipeline`] uses per
///    (chunk, slot) to extract a speaker-specific embedding from
///    a multi-speaker chunk.
///
/// `embed_audio_clips_batch` and `embed_chunk_with_frame_mask` differ
/// in how they handle the segmentation mask:
/// - The audio-clips path masks via audio zeroing (ORT) — the model
///   sees a "filtered" audio with silence in inactive frames.
/// - The frame-mask path uses pyannote's exact `forward(waveforms,
///   weights)` — the model sees the raw audio, and the pooling
///   layer integrates only over active frames. This matches
///   pyannote's bit-exact embedding extraction; the audio-zeroing
///   approach is an approximation that diverges by O(1) per element
///   on overlap-heavy chunks.
///
/// [`embed_audio_clips_batch`]: EmbedBackend::embed_audio_clips_batch
/// [`embed_chunk_with_frame_mask`]: EmbedBackend::embed_chunk_with_frame_mask
pub(crate) trait EmbedBackend: Send {
  /// Embed a batch of audio clips. Each clip must be exactly
  /// `EMBED_WINDOW_SAMPLES = 32_000` samples long (2 s @ 16 kHz);
  /// the Rust embedder zero-pads shorter clips before calling.
  fn embed_audio_clips_batch(
    &mut self,
    clips: &[&[f32]],
  ) -> Result<Vec<[f32; EMBEDDING_DIM]>, Error>;

  /// Embed a 10-second chunk (160_000 samples) using a 589-frame
  /// per-frame mask as pooling weights. Pyannote's exact embedding
  /// extraction call.
  ///
  /// The default implementation **gathers** samples in the
  /// mask-active frames (drops inactive regions entirely) and runs
  /// sliding-window inference on the gathered audio. This is what
  /// the ORT backend uses — the bundled ONNX model doesn't accept a
  /// weights input, so we fall back to the audio-zeroing
  /// approximation that was the previous behavior. The tch backend
  /// overrides to pass weights directly to the TorchScript module
  /// (bit-exact pyannote).
  fn embed_chunk_with_frame_mask(
    &mut self,
    chunk_samples: &[f32],
    frame_mask: &[bool],
  ) -> Result<[f32; EMBEDDING_DIM], Error> {
    use crate::embed::options::{EMBED_WINDOW_SAMPLES, HOP_SAMPLES, MIN_CLIP_SAMPLES};
    let total_samples = chunk_samples.len();
    let frame_count = frame_mask.len();
    if frame_count == 0 {
      return Err(Error::InvalidClip {
        len: 0,
        min: MIN_CLIP_SAMPLES as usize,
      });
    }

    // Build per-sample mask from per-frame mask, then GATHER active
    // samples (matching the previous `embed_masked_raw` semantics).
    let samples_per_frame = total_samples as f64 / frame_count as f64;
    let mut sample_mask = vec![false; total_samples];
    for (f, &active) in frame_mask.iter().enumerate() {
      if !active {
        continue;
      }
      let s0 = (f as f64 * samples_per_frame).round() as usize;
      let s1 = ((f + 1) as f64 * samples_per_frame).round() as usize;
      let lo = s0.min(total_samples);
      let hi = s1.min(total_samples);
      for v in &mut sample_mask[lo..hi] {
        *v = true;
      }
    }
    let gathered: Vec<f32> = chunk_samples
      .iter()
      .zip(sample_mask.iter())
      .filter_map(|(&s, &keep)| keep.then_some(s))
      .collect();
    if gathered.len() < MIN_CLIP_SAMPLES as usize {
      return Err(Error::InvalidClip {
        len: gathered.len(),
        min: MIN_CLIP_SAMPLES as usize,
      });
    }

    let win = EMBED_WINDOW_SAMPLES as usize;
    let mut sum = [0.0_f32; EMBEDDING_DIM];
    if gathered.len() <= win {
      let mut padded = vec![0.0_f32; win];
      padded[..gathered.len()].copy_from_slice(&gathered);
      let raws = self.embed_audio_clips_batch(&[padded.as_slice()])?;
      sum.copy_from_slice(&raws[0]);
      return Ok(sum);
    }
    let hop = HOP_SAMPLES as usize;
    let k_max = (gathered.len() - win) / hop;
    let mut starts: Vec<usize> = (0..=k_max).map(|k| k * hop).collect();
    starts.push(gathered.len() - win);
    starts.sort_unstable();
    starts.dedup();
    let clips: Vec<&[f32]> = starts.iter().map(|&s| &gathered[s..s + win]).collect();
    let raws = self.embed_audio_clips_batch(&clips)?;
    for raw in &raws {
      for (s, r) in sum.iter_mut().zip(raw.iter()) {
        *s += r;
      }
    }
    Ok(sum)
  }
}

// ── ORT (ONNX) backend ──────────────────────────────────────────────

#[cfg(feature = "ort")]
mod ort_backend {
  use super::*;
  use crate::embed::fbank::compute_fbank;
  use ort::{session::Session as OrtSession, value::TensorRef};

  pub(crate) struct OrtBackend {
    pub(crate) session: OrtSession,
  }

  /// Number of segmentation frames per 10s chunk in pyannote's
  /// community-1 config. Used as the default `weights` length when
  /// the high-level audio-clips path doesn't carry a per-frame mask
  /// (we pass all-ones to disable weighted pooling).
  const SEG_FRAMES_PER_CHUNK: usize = 589;

  fn run_inference(
    session: &mut OrtSession,
    n: usize,
    fbank_flat: &[f32],
    fbank_frames: usize,
    weights_flat: &[f32],
    num_weights: usize,
  ) -> Result<Vec<[f32; EMBEDDING_DIM]>, Error> {
    let outputs = session.run(ort::inputs![
      "fbank" => TensorRef::from_array_view((
        [n, fbank_frames, FBANK_NUM_MELS],
        fbank_flat,
      ))?,
      "weights" => TensorRef::from_array_view((
        [n, num_weights],
        weights_flat,
      ))?,
    ])?;
    let (shape, data) = outputs[0].try_extract_tensor::<f32>()?;
    // Per-call shape contract: the ResNet's output must be exactly
    // `[n, EMBEDDING_DIM]`. Validating only the element count (`n *
    // EMBEDDING_DIM`) lets a custom/exporter-drifted model that emits
    // `[EMBEDDING_DIM, n]`, `[1, n * EMBEDDING_DIM]`, or any rank-1
    // flattening pass through. Each chunk would then be silently
    // mis-stridden into PLDA/clustering as if it were `[n, 256]` — the
    // resulting embeddings are corrupted but finite, so no downstream
    // validation catches it. We reject any shape divergence at the ABI
    // boundary before reading rows.
    let dims: &[i64] = shape.as_ref();
    let expected_n = n as i64;
    let expected_dim = EMBEDDING_DIM as i64;
    if dims.len() != 2 || dims[0] != expected_n || dims[1] != expected_dim {
      return Err(Error::InferenceOutputShape {
        got: dims.to_vec(),
        n,
        embedding_dim: EMBEDDING_DIM,
      });
    }
    let expected = n * EMBEDDING_DIM;
    if data.len() != expected {
      return Err(Error::InferenceShapeMismatch {
        expected,
        got: data.len(),
      });
    }
    Ok(
      data
        .chunks_exact(EMBEDDING_DIM)
        .take(n)
        .map(|chunk| {
          let mut row = [0.0f32; EMBEDDING_DIM];
          row.copy_from_slice(chunk);
          row
        })
        .collect(),
    )
  }

  impl super::EmbedBackend for OrtBackend {
    fn embed_audio_clips_batch(
      &mut self,
      clips: &[&[f32]],
    ) -> Result<Vec<[f32; EMBEDDING_DIM]>, Error> {
      let n = clips.len();
      if n == 0 {
        return Ok(Vec::new());
      }
      // 2s clips → 200-frame fbank. Pass all-ones weights at the
      // same length so the resnet's pooling layer treats every frame
      // equally. Length matches `FBANK_FRAMES = 200`; pyannote's
      // pooling layer accepts mismatched fbank/weights lengths via
      // resampling but the trivial all-ones case avoids that path.
      let mut flat = Vec::with_capacity(n * FBANK_FRAMES * FBANK_NUM_MELS);
      for clip in clips.iter() {
        let fbank = compute_fbank(clip)?;
        for row in fbank.iter() {
          flat.extend_from_slice(row);
        }
      }
      let weights_flat = vec![1.0_f32; n * FBANK_FRAMES];
      run_inference(
        &mut self.session,
        n,
        &flat,
        FBANK_FRAMES,
        &weights_flat,
        FBANK_FRAMES,
      )
    }

    fn embed_chunk_with_frame_mask(
      &mut self,
      chunk_samples: &[f32],
      frame_mask: &[bool],
    ) -> Result<[f32; EMBEDDING_DIM], Error> {
      // Pyannote's exact embedding extraction: 10s chunk → fbank →
      // resnet+pool with frame_mask as weights → embedding. We
      // compute the fbank in Rust (kaldi-native-fbank) since
      // torchaudio's kaldi.fbank doesn't export to ONNX.
      use crate::embed::fbank::compute_full_fbank;
      let fbank = compute_full_fbank(chunk_samples)?;
      let num_frames = fbank.len() / FBANK_NUM_MELS;
      let weights_flat: Vec<f32> = frame_mask
        .iter()
        .map(|&b| if b { 1.0 } else { 0.0 })
        .collect();
      let _ = SEG_FRAMES_PER_CHUNK; // doc reference
      let mut out = run_inference(
        &mut self.session,
        1,
        &fbank,
        num_frames,
        &weights_flat,
        frame_mask.len(),
      )?;
      Ok(out.pop().expect("n=1 batch"))
    }
  }
}

// ── tch (TorchScript) backend ───────────────────────────────────────

#[cfg(feature = "tch")]
mod tch_backend {
  use super::*;
  use tch::{CModule, Device, Kind, Tensor};

  pub(crate) struct TchBackend {
    pub(crate) module: CModule,
  }

  impl super::EmbedBackend for TchBackend {
    fn embed_audio_clips_batch(
      &mut self,
      clips: &[&[f32]],
    ) -> Result<Vec<[f32; EMBEDDING_DIM]>, Error> {
      // The TorchScript module signature is `forward(waveforms,
      // weights)`. For unweighted aggregation, pass an all-ones
      // weights tensor of the matching frame count. Pyannote's
      // segmentation model emits 589 frames per 10s window; for
      // 2s windows the resnet's pooling layer interpolates the
      // weights as needed. We pass `(seg_frames * window_secs / 10)`
      // weights — the wrapper was traced at 589, so we always pass
      // 589-element ones here for batch=1.
      let n = clips.len();
      if n == 0 {
        return Ok(Vec::new());
      }
      let mut out = Vec::with_capacity(n);
      for clip in clips.iter() {
        let len = clip.len();
        let input = Tensor::from_slice(clip).reshape([1, len as i64]);
        let weights = Tensor::ones([1, 589], (Kind::Float, Device::Cpu));
        let output = self.module.forward_ts(&[input, weights])?;
        let expected_shape = [1_i64, EMBEDDING_DIM as i64];
        if output.size() != expected_shape {
          return Err(Error::InferenceShapeMismatch {
            expected: EMBEDDING_DIM,
            got: output.numel(),
          });
        }
        let mut row = [0.0_f32; EMBEDDING_DIM];
        output.copy_data(&mut row, EMBEDDING_DIM);
        out.push(row);
      }
      Ok(out)
    }

    fn embed_chunk_with_frame_mask(
      &mut self,
      chunk_samples: &[f32],
      frame_mask: &[bool],
    ) -> Result<[f32; EMBEDDING_DIM], Error> {
      // Pyannote's exact embedding extraction: pass the full chunk
      // audio + the per-frame mask as pooling weights. The
      // TorchScript wrapper handles fbank + resnet + statistics
      // pooling internally; the weights drive the pooling layer
      // (active frames count, inactive frames are skipped).
      let len = chunk_samples.len();
      let input = Tensor::from_slice(chunk_samples).reshape([1, len as i64]);
      let weights_data: Vec<f32> = frame_mask
        .iter()
        .map(|&b| if b { 1.0 } else { 0.0 })
        .collect();
      let weights = Tensor::from_slice(&weights_data).reshape([1, frame_mask.len() as i64]);
      let output = self.module.forward_ts(&[input, weights])?;
      let expected_shape = [1_i64, EMBEDDING_DIM as i64];
      if output.size() != expected_shape {
        return Err(Error::InferenceShapeMismatch {
          expected: EMBEDDING_DIM,
          got: output.numel(),
        });
      }
      let mut row = [0.0_f32; EMBEDDING_DIM];
      output.copy_data(&mut row, EMBEDDING_DIM);
      Ok(row)
    }
  }
}

// ── EmbedModel — public wrapper ─────────────────────────────────────

/// WeSpeaker ResNet34 embedding inference. Holds one backend session
/// (ORT or tch). `Send`-only; one instance per worker thread.
pub struct EmbedModel {
  backend: Box<dyn EmbedBackend>,
}

impl EmbedModel {
  /// Load the ONNX model from disk with default options.
  ///
  /// Available with the `ort` feature (on by default).
  #[cfg(feature = "ort")]
  #[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
  pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
    Self::from_file_with_options(path, EmbedModelOptions::default())
  }

  /// Load the ONNX model from disk with custom options.
  #[cfg(feature = "ort")]
  #[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
  pub fn from_file_with_options<P: AsRef<Path>>(
    path: P,
    opts: EmbedModelOptions,
  ) -> Result<Self, Error> {
    use ort::session::Session as OrtSession;
    let path = path.as_ref();
    let mut builder = opts.apply(OrtSession::builder()?)?;
    let session = builder
      .commit_from_file(path)
      .map_err(|source| Error::LoadModel {
        path: path.to_path_buf(),
        source,
      })?;
    Ok(Self {
      backend: Box::new(ort_backend::OrtBackend { session }),
    })
  }

  /// Load the ONNX model from an in-memory byte buffer (default options).
  #[cfg(feature = "ort")]
  #[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
  pub fn from_memory(bytes: &[u8]) -> Result<Self, Error> {
    Self::from_memory_with_options(bytes, EmbedModelOptions::default())
  }

  /// Load the ONNX model from an in-memory byte buffer with custom options.
  #[cfg(feature = "ort")]
  #[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
  pub fn from_memory_with_options(bytes: &[u8], opts: EmbedModelOptions) -> Result<Self, Error> {
    use ort::session::Session as OrtSession;
    let mut builder = opts.apply(OrtSession::builder()?)?;
    let session = builder.commit_from_memory(bytes)?;
    Ok(Self {
      backend: Box::new(ort_backend::OrtBackend { session }),
    })
  }

  /// Load a TorchScript module from disk.
  ///
  /// Available with the `tch` feature. The module must accept a single
  /// `[N, FBANK_FRAMES, FBANK_NUM_MELS] = [N, 200, 80]` f32 tensor and
  /// return `[N, EMBEDDING_DIM] = [N, 256]` raw embeddings. See
  /// `scripts/export-wespeaker-torchscript.py` for the conversion from
  /// pyannote's PyTorch model.
  #[cfg(feature = "tch")]
  #[cfg_attr(docsrs, doc(cfg(feature = "tch")))]
  pub fn from_torchscript_file<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
    let path = path.as_ref();
    let module = tch::CModule::load(path).map_err(|source| Error::LoadTorchScript {
      path: path.to_path_buf(),
      source,
    })?;
    Ok(Self {
      backend: Box::new(tch_backend::TchBackend { module }),
    })
  }

  /// Embed a single 2-second audio clip. Returns the raw (un-normalized)
  /// 256-d embedding. `samples.len()` must be exactly
  /// `EMBED_WINDOW_SAMPLES = 32_000`; the high-level methods
  /// (`embed`, `embed_weighted`, `embed_masked`) handle padding and
  /// sliding-window aggregation automatically.
  pub(crate) fn embed_audio_clip(
    &mut self,
    samples: &[f32],
  ) -> Result<[f32; EMBEDDING_DIM], Error> {
    let mut out = self.backend.embed_audio_clips_batch(&[samples])?;
    let raw = out
      .pop()
      .expect("backend returned a non-empty batch for n=1 input");
    if raw.iter().any(|v| !v.is_finite()) {
      return Err(Error::NonFiniteOutput);
    }
    Ok(raw)
  }

  /// Batched audio-clip inference. Returns N raw (un-normalized)
  /// 256-d embeddings. An empty input returns `Vec::new()` without
  /// invoking the backend.
  pub(crate) fn embed_audio_clips_batch(
    &mut self,
    clips: &[&[f32]],
  ) -> Result<Vec<[f32; EMBEDDING_DIM]>, Error> {
    let raws = self.backend.embed_audio_clips_batch(clips)?;
    // Centralized finite check at the EmbedModel boundary: neither the
    // ORT nor tch backend validates per-element finiteness on its own,
    // and the high-level `embed`/`embed_weighted`/`embed_masked`
    // helpers go straight from this batch into per-window axpy
    // accumulation. A NaN/inf raw row would propagate through the L2
    // normalize and feed PLDA/clustering as a "valid" speaker vector.
    for raw in raws.iter() {
      if raw.iter().any(|v| !v.is_finite()) {
        return Err(Error::NonFiniteOutput);
      }
    }
    Ok(raws)
  }

  /// Pyannote-style speaker embedding for a 10-second chunk + per-
  /// frame segmentation mask. Returns the raw (un-normalized) 256-d
  /// embedding for the speaker whose activity is in `frame_mask`.
  ///
  /// Backend dispatches:
  /// - **ORT**: zeroes audio in inactive frames, runs sliding-window
  ///   inference, sums the per-window outputs. Approximate (the
  ///   bundled ONNX model doesn't accept a weights input).
  /// - **tch**: passes `(audio, frame_mask)` directly to the
  ///   TorchScript wrapper, which delegates to pyannote's
  ///   `WeSpeakerResNet34.forward(waveforms, weights=mask)` —
  ///   bit-exact pyannote.
  pub fn embed_chunk_with_frame_mask(
    &mut self,
    chunk_samples: &[f32],
    frame_mask: &[bool],
  ) -> Result<[f32; EMBEDDING_DIM], Error> {
    // Centralized boundary validation that cannot be bypassed by a
    // backend's `embed_chunk_with_frame_mask` override. The `EmbedBackend`
    // trait provides default empty/short-mask guards via its
    // gather-then-window fallback, but the ORT and tch overrides skip
    // them and pass `frame_mask` straight to the model. An empty or
    // all-false mask becomes all-zero pooling weights, division-by-zero
    // in statistics pooling, and NaN/inf rows the public API has no
    // other place to catch.
    if frame_mask.is_empty() || !frame_mask.iter().any(|&b| b) {
      return Err(Error::EmptyOrInactiveMask);
    }
    let raw = self
      .backend
      .embed_chunk_with_frame_mask(chunk_samples, frame_mask)?;
    if raw.iter().any(|v| !v.is_finite()) {
      return Err(Error::NonFiniteOutput);
    }
    Ok(raw)
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
    if voice_probs.len() != samples.len() {
      return Err(Error::WeightShapeMismatch {
        samples_len: samples.len(),
        weights_len: voice_probs.len(),
      });
    }
    let (sum, windows_used, weight_sum) = embed_weighted_inner(self, samples, voice_probs)?;
    let embedding = Embedding::normalize_from(sum).ok_or(Error::DegenerateEmbedding)?;
    let duration = duration_from_samples(samples.len());
    Ok(EmbeddingResult::new(
      embedding,
      duration,
      windows_used,
      weight_sum,
      meta,
    ))
  }

  /// Mask-gated embedding: same windowing as
  /// [`embed`](Self::embed), but each fbank row is **zeroed out**
  /// where `keep_mask` is `false` for the corresponding sample window.
  /// Equivalent to running pyannote's masked-clip embedding.
  pub fn embed_masked(
    &mut self,
    samples: &[f32],
    keep_mask: &[bool],
  ) -> Result<EmbeddingResult, Error> {
    self.embed_masked_with_meta(samples, keep_mask, EmbeddingMeta::default())
  }

  /// Raw masked embedding — returns the un-normalized 256-d output.
  /// Useful for downstream PLDA stages that consume raw embeddings.
  ///
  /// Gathers samples where `keep_mask` is true (drops the rest), then
  /// runs the standard sliding-window pipeline on the gathered audio.
  pub fn embed_masked_raw(
    &mut self,
    samples: &[f32],
    keep_mask: &[bool],
  ) -> Result<[f32; EMBEDDING_DIM], Error> {
    if keep_mask.len() != samples.len() {
      return Err(Error::MaskShapeMismatch {
        samples_len: samples.len(),
        mask_len: keep_mask.len(),
      });
    }
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
    let (sum, _windows_used) = embed_unweighted(self, &gathered)?;
    Ok(sum)
  }

  /// Mask-gated embedding with metadata.
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

#[inline]
fn duration_from_samples(samples: usize) -> Duration {
  Duration::from_secs_f64(samples as f64 / SAMPLE_RATE_HZ as f64)
}

#[cfg(all(test, feature = "ort"))]
mod tests {
  use super::*;
  use crate::embed::options::EMBED_WINDOW_SAMPLES;
  use std::path::PathBuf;

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
    let samples = vec![0.0f32; EMBED_WINDOW_SAMPLES as usize];
    let raw = model.embed_audio_clip(&samples).expect("infer silence");
    assert_eq!(raw.len(), EMBEDDING_DIM);
    assert!(raw.iter().all(|v| v.is_finite()));
  }

  #[test]
  #[ignore = "requires WeSpeaker ResNet34-LM ONNX model"]
  fn batch_inference_matches_single() {
    let path = model_path();
    if !path.exists() {
      return;
    }
    let mut model = EmbedModel::from_file(&path).expect("load model");
    let samples = vec![0.001f32; EMBED_WINDOW_SAMPLES as usize];
    let single = model.embed_audio_clip(&samples).expect("single");
    let batch = model.embed_audio_clips_batch(&[&samples]).expect("batch");
    assert_eq!(batch.len(), 1);
    assert_eq!(single, batch[0]);
  }

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
    let n_sq: f32 = r.embedding().as_array().iter().map(|x| x * x).sum();
    let norm = n_sq.sqrt();
    assert!((norm - 1.0).abs() < 1e-5);
    assert_eq!(r.windows_used(), 1);
  }

  #[test]
  #[ignore = "requires WeSpeaker ResNet34-LM ONNX model"]
  fn embed_long_clip_uses_sliding_window() {
    let path = model_path();
    if !path.exists() {
      return;
    }
    let mut model = EmbedModel::from_file(&path).expect("load model");
    let samples = vec![0.001f32; 2 * EMBED_WINDOW_SAMPLES as usize];
    let r = model.embed(&samples).expect("embed succeeds");
    assert_eq!(r.windows_used(), 3);
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
    let probs = vec![1.0f32; EMBED_WINDOW_SAMPLES as usize - 1];
    let r = model.embed_weighted(&samples, &probs);
    assert!(matches!(
      r,
      Err(Error::WeightShapeMismatch {
        samples_len: 32_000,
        weights_len: 31_999,
      })
    ));
  }

  #[test]
  #[ignore = "requires WeSpeaker ResNet34-LM ONNX model"]
  fn embed_masked_rejects_short_gathered_clip() {
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
    assert!(matches!(r, Err(Error::InvalidClip { len: 100, min: 400 })));
  }

  /// `EmbedModel::embed_chunk_with_frame_mask` rejects an empty
  /// `frame_mask` at the public boundary BEFORE invoking the backend.
  /// This guard cannot be bypassed by an ORT/tch backend override that
  /// elides the trait default's frame-count check.
  #[test]
  #[ignore = "requires WeSpeaker ResNet34-LM ONNX model"]
  fn embed_chunk_with_frame_mask_rejects_empty_mask() {
    let path = model_path();
    if !path.exists() {
      return;
    }
    let mut model = EmbedModel::from_file(&path).expect("load model");
    let samples = vec![0.001f32; EMBED_WINDOW_SAMPLES as usize];
    let mask: Vec<bool> = Vec::new();
    let r = model.embed_chunk_with_frame_mask(&samples, &mask);
    assert!(matches!(r, Err(Error::EmptyOrInactiveMask)), "got {r:?}");
  }

  /// All-false `frame_mask` produces all-zero pooling weights →
  /// division-by-zero in statistics pooling → NaN/inf raw vector
  /// downstream. We reject it at the EmbedModel boundary instead.
  #[test]
  #[ignore = "requires WeSpeaker ResNet34-LM ONNX model"]
  fn embed_chunk_with_frame_mask_rejects_all_false_mask() {
    let path = model_path();
    if !path.exists() {
      return;
    }
    let mut model = EmbedModel::from_file(&path).expect("load model");
    let samples = vec![0.001f32; EMBED_WINDOW_SAMPLES as usize];
    let mask = vec![false; 589];
    let r = model.embed_chunk_with_frame_mask(&samples, &mask);
    assert!(matches!(r, Err(Error::EmptyOrInactiveMask)), "got {r:?}");
  }
}
