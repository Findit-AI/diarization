//! End-to-end audio→RTTM offline diarization.
//!
//! `OwnedDiarizationPipeline` is the speakrs-comparable batch
//! entrypoint: take owned 16 kHz mono samples, run segmentation +
//! embedding ONNX inference internally, project through PLDA, run
//! `cluster_vbx`, reconstruct frame-level diarization, and return
//! spans / RTTM. Pyannote `community-1` algorithm.
//!
//! ## Status
//!
//! MVP. End-to-end orchestration works on the captured fixtures.
//! Cross-chunk speaker permutation alignment is *not* performed —
//! `assign_embeddings` (AHC) handles cross-chunk pairing
//! algorithmically via embedding similarity, so the slot ordering
//! within each chunk being arbitrary doesn't break the pipeline.
//! However, the per-output-frame `count` aggregation uses simple
//! averaging-then-binarize across covering chunks, *not* pyannote's
//! PIT-permutation-aware aggregation. This is a known divergence
//! that affects the discrete diarization grid (and thus the
//! reconstruction step's choice of which speakers to emit per
//! frame). DER target: ≤5% on community-1 evaluation sets; bit-
//! exact pyannote parity is reserved for the offline-from-captures
//! path (`offline::diarize_offline`).

use crate::{
  aggregate::try_count_pyannote,
  embed::{EMBEDDING_DIM, EmbedModel},
  offline::{Error, OfflineInput, OfflineOutput, diarize_offline},
  plda::PldaTransform,
  reconstruct::SlidingWindow,
  segment::{
    FRAMES_PER_WINDOW, POWERSET_CLASSES, PYANNOTE_FRAME_DURATION_S, PYANNOTE_FRAME_STEP_S,
    SAMPLE_RATE_HZ, SegmentModel, WINDOW_SAMPLES,
    powerset::{powerset_to_speakers_hard, softmax_row},
  },
};

/// Number of speaker slots per chunk. Pyannote `segmentation-3.0`
/// trains on 3 simultaneous speakers (the 7 powerset classes).
pub const SLOTS_PER_CHUNK: usize = 3;

/// Configuration for [`OwnedDiarizationPipeline`].
///
/// Defaults match pyannote `speaker-diarization-community-1`:
/// 1-second chunk step, 0.5 onset/offset binarization, threshold/Fa/Fb
/// from the community-1 config.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct OwnedPipelineOptions {
  #[cfg_attr(feature = "serde", serde(default = "default_step_samples"))]
  step_samples: u32,
  #[cfg_attr(feature = "serde", serde(default = "default_onset"))]
  onset: f32,
  #[cfg_attr(feature = "serde", serde(default = "default_threshold"))]
  threshold: f64,
  #[cfg_attr(feature = "serde", serde(default = "default_fa"))]
  fa: f64,
  #[cfg_attr(feature = "serde", serde(default = "default_fb"))]
  fb: f64,
  #[cfg_attr(feature = "serde", serde(default = "default_max_iters"))]
  max_iters: usize,
  #[cfg_attr(feature = "serde", serde(default))]
  min_duration_off: f64,
  #[cfg_attr(feature = "serde", serde(default = "default_smoothing_epsilon"))]
  smoothing_epsilon: Option<f32>,
}

#[cfg(feature = "serde")]
const fn default_step_samples() -> u32 {
  16_000
}
#[cfg(feature = "serde")]
const fn default_onset() -> f32 {
  0.5
}
#[cfg(feature = "serde")]
const fn default_threshold() -> f64 {
  0.6
}
#[cfg(feature = "serde")]
const fn default_fa() -> f64 {
  0.07
}
#[cfg(feature = "serde")]
const fn default_fb() -> f64 {
  0.8
}
#[cfg(feature = "serde")]
const fn default_max_iters() -> usize {
  20
}
#[cfg(feature = "serde")]
const fn default_smoothing_epsilon() -> Option<f32> {
  Some(0.1)
}

impl OwnedPipelineOptions {
  /// Construct with `community-1` defaults.
  pub const fn new() -> Self {
    Self {
      step_samples: 16_000, // 1 s — community-1 config
      onset: 0.5,
      threshold: 0.6,
      fa: 0.07,
      fb: 0.8,
      max_iters: 20,
      min_duration_off: 0.0,
      smoothing_epsilon: Some(0.1),
    }
  }

  // ── Getters ─────────────────────────────────────────────────────

  /// Sliding-window step in samples. Community-1 uses 16_000 (1 s).
  pub const fn step_samples(&self) -> u32 {
    self.step_samples
  }
  /// Frame-level binarization onset (default: 0.5).
  pub const fn onset(&self) -> f32 {
    self.onset
  }
  /// AHC linkage threshold (community-1: 0.6).
  pub const fn threshold(&self) -> f64 {
    self.threshold
  }
  /// VBx Fa (community-1: 0.07).
  pub const fn fa(&self) -> f64 {
    self.fa
  }
  /// VBx Fb (community-1: 0.8).
  pub const fn fb(&self) -> f64 {
    self.fb
  }
  /// VBx max iterations (community-1 hardcodes 20).
  pub const fn max_iters(&self) -> usize {
    self.max_iters
  }
  /// Span post-processing min_duration_off (seconds).
  pub const fn min_duration_off(&self) -> f64 {
    self.min_duration_off
  }
  /// Temporal smoothing epsilon for top-k reconstruction.
  pub const fn smoothing_epsilon(&self) -> Option<f32> {
    self.smoothing_epsilon
  }

  // ── Builders ────────────────────────────────────────────────────

  /// Builder: sliding-window step in samples.
  #[must_use]
  pub const fn with_step_samples(mut self, v: u32) -> Self {
    self.step_samples = v;
    self
  }
  /// Builder: frame-level binarization onset.
  #[must_use]
  pub const fn with_onset(mut self, v: f32) -> Self {
    self.onset = v;
    self
  }
  /// Builder: AHC linkage threshold.
  #[must_use]
  pub const fn with_threshold(mut self, v: f64) -> Self {
    self.threshold = v;
    self
  }
  /// Builder: VBx Fa.
  #[must_use]
  pub const fn with_fa(mut self, v: f64) -> Self {
    self.fa = v;
    self
  }
  /// Builder: VBx Fb.
  #[must_use]
  pub const fn with_fb(mut self, v: f64) -> Self {
    self.fb = v;
    self
  }
  /// Builder: VBx max iterations.
  #[must_use]
  pub const fn with_max_iters(mut self, v: usize) -> Self {
    self.max_iters = v;
    self
  }
  /// Builder: span post-processing `min_duration_off` (seconds).
  #[must_use]
  pub const fn with_min_duration_off(mut self, v: f64) -> Self {
    self.min_duration_off = v;
    self
  }
  /// Builder: temporal smoothing epsilon. Pass `None` for bit-exact
  /// pyannote argmax behavior, `Some(0.1)` for `community-1` smoothed
  /// reconstruction.
  #[must_use]
  pub const fn with_smoothing_epsilon(mut self, v: Option<f32>) -> Self {
    self.smoothing_epsilon = v;
    self
  }
}

impl Default for OwnedPipelineOptions {
  fn default() -> Self {
    Self::new()
  }
}

/// End-to-end audio→RTTM offline diarization pipeline.
///
/// Borrows `&mut SegmentModel`, `&mut EmbedModel`, and `&PldaTransform`
/// per [`run`](Self::run) call (they're caller-owned because both
/// model types are `!Sync` — see [`crate::diarizer::Diarizer`] for
/// the same pattern). Configuration is held in [`OwnedPipelineOptions`].
pub struct OwnedDiarizationPipeline {
  options: OwnedPipelineOptions,
}

impl OwnedDiarizationPipeline {
  /// Construct with the community-1 default options.
  pub const fn new() -> Self {
    Self {
      options: OwnedPipelineOptions::new(),
    }
  }

  /// Construct with explicit options.
  pub fn with_options(options: OwnedPipelineOptions) -> Self {
    Self { options }
  }

  /// Borrow the options.
  pub fn options(&self) -> &OwnedPipelineOptions {
    &self.options
  }

  /// Run diarization on owned 16 kHz mono samples.
  ///
  /// Returns the same [`OfflineOutput`] shape as
  /// [`diarize_offline`](super::diarize_offline) — `(hard_clusters,
  /// discrete_diarization, num_clusters, spans)`.
  ///
  /// # Errors
  ///
  /// - [`Error::Shape`] if `samples` is empty or shorter than one
  ///   segmentation window (`WINDOW_SAMPLES = 160_000` = 10 s).
  /// - All other errors propagate from the underlying ONNX inference,
  ///   PLDA, AHC, VBx, centroid, Hungarian, or reconstruct stages.
  pub fn run(
    &self,
    seg_model: &mut SegmentModel,
    embed_model: &mut EmbedModel,
    plda: &PldaTransform,
    samples: &[f32],
  ) -> Result<OfflineOutput, Error> {
    let cfg = &self.options;
    if samples.is_empty() {
      return Err(crate::offline::algo::ShapeError::EmptySamples.into());
    }
    let win = WINDOW_SAMPLES as usize;
    let step = cfg.step_samples() as usize;
    if step == 0 {
      return Err(crate::offline::algo::ShapeError::ZeroStepSamples.into());
    }

    // ── Stage 1: chunked sliding-window segmentation ───────────────
    // Last-chunk zero-pad if `samples` doesn't align with the grid.
    let num_chunks = if samples.len() <= win {
      1
    } else {
      (samples.len() - win).div_ceil(step) + 1
    };

    let mut padded_chunk = vec![0.0_f32; win];
    let mut segmentations: Vec<f64> = vec![0.0; num_chunks * FRAMES_PER_WINDOW * SLOTS_PER_CHUNK];

    for c in 0..num_chunks {
      let start = c * step;
      // Build the (possibly zero-padded) 10s window.
      padded_chunk.fill(0.0);
      let end = (start + win).min(samples.len());
      let lo = start.min(samples.len());
      let n = end - lo;
      if n > 0 {
        padded_chunk[..n].copy_from_slice(&samples[lo..end]);
      }

      let logits = seg_model.infer(&padded_chunk)?;
      // logits is [FRAMES_PER_WINDOW * POWERSET_CLASSES] row-major.
      for f in 0..FRAMES_PER_WINDOW {
        let mut row = [0.0_f32; POWERSET_CLASSES];
        for k in 0..POWERSET_CLASSES {
          row[k] = logits[f * POWERSET_CLASSES + k];
        }
        let probs = softmax_row(&row);
        // Pyannote's `to_multilabel(powerset, soft=False)` picks the
        // argmax powerset class, then maps to the speaker mask. This
        // is the conversion captured `segmentations.npz` reflects —
        // every entry is exactly 0.0 or 1.0. Soft marginals followed
        // by `>= onset` would disagree on 3-way overlap chunks where
        // the marginal sum exceeds 0.5 but argmax picks a different
        // class. Critical for `filter_embeddings`'s `single_active`
        // mask (frames where sum_speakers == 1) and for `count`,
        // both of which assume hard argmax binarization.
        let speakers = powerset_to_speakers_hard(&probs);
        for s in 0..SLOTS_PER_CHUNK {
          segmentations[(c * FRAMES_PER_WINDOW + f) * SLOTS_PER_CHUNK + s] = speakers[s] as f64;
        }
      }
    }

    // ── Stage 2: per-(chunk, slot) masked embedding ────────────────
    let mut raw_embeddings: Vec<f32> = vec![0.0; num_chunks * SLOTS_PER_CHUNK * EMBEDDING_DIM];

    for c in 0..num_chunks {
      let start = c * step;
      // Re-slice the same padded window we used for segmentation so
      // mask offsets line up. Zero-pad samples outside the audio range.
      padded_chunk.fill(0.0);
      let end = (start + win).min(samples.len());
      let lo = start.min(samples.len());
      let n = end - lo;
      if n > 0 {
        padded_chunk[..n].copy_from_slice(&samples[lo..end]);
      }

      for s in 0..SLOTS_PER_CHUNK {
        // Build per-frame binary mask: speaker active iff seg > onset.
        let mut frame_mask = [false; FRAMES_PER_WINDOW];
        let mut any_active = false;
        for f in 0..FRAMES_PER_WINDOW {
          let active =
            segmentations[(c * FRAMES_PER_WINDOW + f) * SLOTS_PER_CHUNK + s] >= cfg.onset() as f64;
          frame_mask[f] = active;
          any_active |= active;
        }
        if !any_active {
          // Zero the segmentation column so filter_embeddings drops
          // this (c, s) pair. Without this, sub-onset segmentation
          // sums (e.g. 0.0001 from ONNX softmax noise) would still
          // satisfy `sum > 0` and admit a zero-embedding into PLDA,
          // failing `RawEmbedding::from_raw_array`'s norm guard.
          for f in 0..FRAMES_PER_WINDOW {
            segmentations[(c * FRAMES_PER_WINDOW + f) * SLOTS_PER_CHUNK + s] = 0.0;
          }
          continue;
        }

        // Run pyannote-style chunk + frame-mask embedding. The
        // EmbedModel's `embed_chunk_with_frame_mask` dispatches based
        // on the active backend: ORT zeroes audio + sliding-window
        // aggregates (approximate); tch passes (audio, mask) directly
        // to the TorchScript wrapper which delegates to pyannote's
        // `WeSpeakerResNet34.forward(waveforms, weights=mask)` —
        // bit-exact pyannote.
        let raw = match embed_model.embed_chunk_with_frame_mask(&padded_chunk, &frame_mask) {
          Ok(v) => v,
          Err(crate::embed::Error::InvalidClip { .. })
          | Err(crate::embed::Error::DegenerateEmbedding) => {
            for f in 0..FRAMES_PER_WINDOW {
              segmentations[(c * FRAMES_PER_WINDOW + f) * SLOTS_PER_CHUNK + s] = 0.0;
            }
            continue;
          }
          Err(e) => return Err(e.into()),
        };
        // Pre-validate: if the raw norm is below the PLDA min, drop.
        // PLDA min is 0.01 (RawEmbedding::from_raw_array). Computing
        // the L2 norm here lets us drop the slot before
        // `diarize_offline` rejects it later.
        let norm_sq: f64 = raw.iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
        if !norm_sq.is_finite() || norm_sq.sqrt() < 0.01 {
          for f in 0..FRAMES_PER_WINDOW {
            segmentations[(c * FRAMES_PER_WINDOW + f) * SLOTS_PER_CHUNK + s] = 0.0;
          }
          continue;
        }
        let dst = (c * SLOTS_PER_CHUNK + s) * EMBEDDING_DIM;
        raw_embeddings[dst..dst + EMBEDDING_DIM].copy_from_slice(&raw);
      }
    }

    // ── Stage 3: build count tensor + sliding-window timing ────────
    //
    // Bit-exact to pyannote 4.0.4's
    // `SpeakerDiarizationMixin.speaker_count` →
    // `Inference.aggregate(hamming=False, skip_average=False,
    // missing=0.0)` with `warm_up=(0.0, 0.0)` (community-1's explicit
    // override of the default `(0.1, 0.1)`).
    //
    // Critical algorithmic property: per-frame count is uniform-
    // averaged across non-NaN contributing chunks, NOT
    // hamming-weighted. The previous implementation used hamming
    // weights and divided by total weight rather than overlap count;
    // see `aggregate::count_pyannote` source for the algorithm and
    // `aggregate::parity_tests` for the bit-exact fixture parity.
    let chunk_duration_s = WINDOW_SAMPLES as f64 / SAMPLE_RATE_HZ as f64;
    let chunk_step_s = cfg.step_samples() as f64 / SAMPLE_RATE_HZ as f64;
    let chunks_sw = SlidingWindow::new(0.0, chunk_duration_s, chunk_step_s);
    let frames_sw_template =
      SlidingWindow::new(0.0, PYANNOTE_FRAME_DURATION_S, PYANNOTE_FRAME_STEP_S);
    // Use the fallible variant: a malformed `onset` (NaN/inf via the
    // public `with_onset` builder) would panic the infallible
    // `count_pyannote` wrapper at `try_count_pyannote.expect(...)`.
    // Surface it as a typed `Error::Aggregate` instead so untrusted
    // config can never crash the process.
    let (count, frames_sw) = try_count_pyannote(
      &segmentations,
      num_chunks,
      FRAMES_PER_WINDOW,
      SLOTS_PER_CHUNK,
      cfg.onset() as f64,
      chunks_sw,
      frames_sw_template,
    )?
    .into_parts();
    let num_output_frames = count.len();

    // ── Stage 4: dispatch to diarize_offline ───────────────────────
    let input = OfflineInput::new(
      &raw_embeddings,
      num_chunks,
      SLOTS_PER_CHUNK,
      &segmentations,
      FRAMES_PER_WINDOW,
      &count,
      num_output_frames,
      chunks_sw,
      frames_sw,
      plda,
    )
    .with_threshold(cfg.threshold())
    .with_fa(cfg.fa())
    .with_fb(cfg.fb())
    .with_max_iters(cfg.max_iters())
    .with_min_duration_off(cfg.min_duration_off())
    .with_smoothing_epsilon(cfg.smoothing_epsilon());
    diarize_offline(&input)
  }
}

impl Default for OwnedDiarizationPipeline {
  fn default() -> Self {
    Self::new()
  }
}

#[cfg(all(test, feature = "serde"))]
mod serde_tests {
  use super::*;

  #[test]
  fn owned_pipeline_config_default_roundtrip() {
    let cfg = OwnedPipelineOptions::new();
    let json = serde_json::to_string(&cfg).expect("serialize");
    let back: OwnedPipelineOptions = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(cfg.step_samples(), back.step_samples());
    assert_eq!(cfg.threshold(), back.threshold());
    assert_eq!(cfg.fa(), back.fa());
    assert_eq!(cfg.fb(), back.fb());
    assert_eq!(cfg.max_iters(), back.max_iters());
    assert_eq!(cfg.smoothing_epsilon(), back.smoothing_epsilon());
  }

  /// Empty JSON object → all defaults filled in.
  #[test]
  fn owned_pipeline_config_empty_json_uses_defaults() {
    let cfg: OwnedPipelineOptions = serde_json::from_str("{}").expect("deserialize");
    let want = OwnedPipelineOptions::new();
    assert_eq!(cfg.step_samples(), want.step_samples());
    assert_eq!(cfg.onset(), want.onset());
    assert_eq!(cfg.threshold(), want.threshold());
    assert_eq!(cfg.smoothing_epsilon(), want.smoothing_epsilon());
  }
}
