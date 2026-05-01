//! Phase 5d: end-to-end audio→RTTM offline diarization.
#![allow(
  clippy::manual_div_ceil,
  clippy::needless_range_loop,
  clippy::useless_vec
)]
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
  aggregate::count_pyannote,
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
pub struct OwnedPipelineConfig {
  /// Sliding-window step in samples. Community-1 uses 16_000 (1 s).
  pub step_samples: u32,
  /// Frame-level binarization onset (default: 0.5).
  pub onset: f32,
  /// AHC linkage threshold (community-1: 0.6).
  pub threshold: f64,
  /// VBx Fa (community-1: 0.07).
  pub fa: f64,
  /// VBx Fb (community-1: 0.8).
  pub fb: f64,
  /// VBx max iterations (community-1 hardcodes 20).
  pub max_iters: usize,
  /// Span post-processing min_duration_off (seconds). Community-1: 0.0.
  pub min_duration_off: f64,
  /// Temporal smoothing epsilon for top-k reconstruction. Community-1
  /// default `Some(0.1)` matches speakrs's
  /// `ReconstructMethod::Smoothed { epsilon: 0.1 }` — reduces flicker
  /// between near-tied speakers. Set to `None` for bit-exact
  /// pyannote argmax behavior.
  pub smoothing_epsilon: Option<f32>,
}

impl Default for OwnedPipelineConfig {
  fn default() -> Self {
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
}

/// End-to-end audio→RTTM offline diarization pipeline.
///
/// Borrows `&mut SegmentModel`, `&mut EmbedModel`, and `&PldaTransform`
/// per [`run`](Self::run) call (they're caller-owned because both
/// model types are `!Sync` — see [`crate::diarizer::Diarizer`] for
/// the same pattern). Configuration is held in [`OwnedPipelineConfig`].
pub struct OwnedDiarizationPipeline {
  config: OwnedPipelineConfig,
}

impl OwnedDiarizationPipeline {
  /// Construct with the community-1 default config.
  pub const fn new() -> Self {
    Self {
      config: OwnedPipelineConfig {
        step_samples: 16_000,
        onset: 0.5,
        threshold: 0.6,
        fa: 0.07,
        fb: 0.8,
        max_iters: 20,
        min_duration_off: 0.0,
        smoothing_epsilon: Some(0.1),
      },
    }
  }

  /// Construct with explicit config.
  pub fn with_config(config: OwnedPipelineConfig) -> Self {
    Self { config }
  }

  /// Borrow the configuration.
  pub fn config(&self) -> &OwnedPipelineConfig {
    &self.config
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
    let cfg = &self.config;
    if samples.is_empty() {
      return Err(Error::Shape("samples is empty"));
    }
    let win = WINDOW_SAMPLES as usize;
    let step = cfg.step_samples as usize;
    if step == 0 {
      return Err(Error::Shape("step_samples must be > 0"));
    }

    // ── Stage 1: chunked sliding-window segmentation ───────────────
    // Last-chunk zero-pad if `samples` doesn't align with the grid.
    let num_chunks = if samples.len() <= win {
      1
    } else {
      ((samples.len() - win) + step - 1) / step + 1
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

      let logits = seg_model
        .infer(&padded_chunk)
        .map_err(|e| Error::Shape(box_leak_segment_err(e)))?;
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
            segmentations[(c * FRAMES_PER_WINDOW + f) * SLOTS_PER_CHUNK + s] >= cfg.onset as f64;
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
          Err(e) => return Err(Error::Shape(box_leak_embed_err(e))),
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
    let chunk_step_s = cfg.step_samples as f64 / SAMPLE_RATE_HZ as f64;
    let (count, frames_sw) = count_pyannote(
      &segmentations,
      num_chunks,
      FRAMES_PER_WINDOW,
      SLOTS_PER_CHUNK,
      cfg.onset as f64,
      chunk_duration_s,
      chunk_step_s,
      PYANNOTE_FRAME_DURATION_S,
      PYANNOTE_FRAME_STEP_S,
    );
    let num_output_frames = count.len();

    let chunks_sw = SlidingWindow {
      start: 0.0,
      duration: chunk_duration_s,
      step: chunk_step_s,
    };

    // ── Stage 4: dispatch to diarize_offline ───────────────────────
    let input = OfflineInput {
      raw_embeddings: &raw_embeddings,
      num_chunks,
      num_speakers: SLOTS_PER_CHUNK,
      segmentations: &segmentations,
      num_frames_per_chunk: FRAMES_PER_WINDOW,
      count: &count,
      num_output_frames,
      chunks_sw,
      frames_sw,
      plda,
      threshold: cfg.threshold,
      fa: cfg.fa,
      fb: cfg.fb,
      max_iters: cfg.max_iters,
      min_duration_off: cfg.min_duration_off,
      smoothing_epsilon: cfg.smoothing_epsilon,
    };
    diarize_offline(&input)
  }
}

impl Default for OwnedDiarizationPipeline {
  fn default() -> Self {
    Self::new()
  }
}

// Error-conversion helpers. The `offline::Error` enum is reused for
// MVP; future revisions can add typed variants for segmentation /
// embedding failures specifically.
fn box_leak_segment_err(e: crate::segment::Error) -> &'static str {
  // For MVP we just stringify; a typed variant is a follow-up.
  Box::leak(format!("segment: {e}").into_boxed_str())
}
fn box_leak_embed_err(e: crate::embed::Error) -> &'static str {
  Box::leak(format!("embed: {e}").into_boxed_str())
}
