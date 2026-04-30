//! Phase 5d: end-to-end audio→RTTM offline diarization.
#![allow(
  clippy::manual_div_ceil,
  clippy::needless_range_loop,
  clippy::useless_vec,
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
  embed::{EMBEDDING_DIM, EmbedModel},
  offline::{Error, OfflineInput, OfflineOutput, diarize_offline},
  plda::PldaTransform,
  reconstruct::SlidingWindow,
  segment::{
    FRAMES_PER_WINDOW, POWERSET_CLASSES, SAMPLE_RATE_HZ, SegmentModel, WINDOW_SAMPLES,
    powerset::{powerset_to_speakers, softmax_row},
  },
};

/// Per-frame audio-rate step. Pyannote community-1's segmentation
/// model emits 589 frames per 10-second window, so each output frame
/// covers `WINDOW_SAMPLES / FRAMES_PER_WINDOW ≈ 271.6` samples — i.e.
/// 16.97 ms at 16 kHz. The per-frame timing is documented as
/// `frame_duration ≈ 0.01697 s` and `frame_step ≈ 0.01697 s` in
/// pyannote's `Inference.aggregate` metadata.
const SAMPLES_PER_OUTPUT_FRAME: f64 = WINDOW_SAMPLES as f64 / FRAMES_PER_WINDOW as f64;

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
        let speakers = powerset_to_speakers(&probs);
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

        // Expand per-frame mask to per-sample mask.
        // Frame f covers samples [f * SAMPLES_PER_OUTPUT_FRAME .. (f+1) * SAMPLES_PER_OUTPUT_FRAME).
        let mut sample_mask = vec![false; win];
        for f in 0..FRAMES_PER_WINDOW {
          if !frame_mask[f] {
            continue;
          }
          let s0 = (f as f64 * SAMPLES_PER_OUTPUT_FRAME).round() as usize;
          let s1 = ((f + 1) as f64 * SAMPLES_PER_OUTPUT_FRAME).round() as usize;
          let lo = s0.min(win);
          let hi = s1.min(win);
          for i in lo..hi {
            sample_mask[i] = true;
          }
        }

        // Run masked embedding on the un-zeroed portion of the window
        // (zero-padding samples have keep=false anyway; for the last
        // chunk, n < win means the padding tail won't have been masked
        // in by speaker activity since pyannote's segmentation will
        // not flag silence as active speech).
        //
        // Skip-on-degenerate policy: when the masked clip produces an
        // unusable embedding (mask gathered too few samples, all-NaN
        // ONNX output, or sub-threshold L2 norm that would fail
        // `RawEmbedding::from_raw_array`'s degenerate-input guard),
        // we zero out BOTH `raw_embeddings[c, s]` and `segmentations
        // [c, :, s]`. The downstream `filter_embeddings` in
        // `diarize_offline` keys off `sum(seg[c, :, s]) > 0`, so
        // zeroing the segmentation column drops the (c, s) pair from
        // the PLDA / VBx training set — matching pyannote's
        // `filter_embeddings` behavior on degraded slots.
        let raw = match embed_model.embed_masked_raw(&padded_chunk, &sample_mask) {
          Ok(v) => v,
          Err(crate::embed::Error::InvalidClip { .. })
          | Err(crate::embed::Error::DegenerateEmbedding) => {
            // Drop this (chunk, slot): zero its segmentation column
            // so filter_embeddings skips it.
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
    // Pyannote's algorithm (per `reconstruct::algo` comment + the
    // pyannote source `pipelines/speaker_diarization.py`):
    //   1. Per chunk, per frame: binarize seg > onset and sum across
    //      slots → per-(chunk, frame) speaker count (integer 0..3).
    //   2. Aggregate this per-chunk count tensor across overlapping
    //      chunks via hamming-weighted average → per-output-frame
    //      float count.
    //   3. Round to integer.
    //
    // Why this is permutation-invariant: the sum-across-slots in step 1
    // collapses speaker identity. Two chunks where slot_0 is different
    // speakers still agree on "how many speakers are speaking now"
    // (assuming the segmentation model is well-trained, which it is
    // for community-1). No PIT alignment needed for the count.
    //
    // Output-frame grid: spans `[0, total_samples)` at
    // `SAMPLES_PER_OUTPUT_FRAME` step. Each chunk c covers output
    // frames `[c * step / spf, (c * step + win) / spf)`.
    let total_samples = ((num_chunks - 1) * step + win) as f64;
    let num_output_frames = (total_samples / SAMPLES_PER_OUTPUT_FRAME).ceil() as usize;

    // Precompute per-chunk per-frame integer counts (slots active).
    let mut chunk_counts: Vec<u8> = vec![0; num_chunks * FRAMES_PER_WINDOW];
    for c in 0..num_chunks {
      for f in 0..FRAMES_PER_WINDOW {
        let mut n = 0u8;
        for s in 0..SLOTS_PER_CHUNK {
          if segmentations[(c * FRAMES_PER_WINDOW + f) * SLOTS_PER_CHUNK + s] >= cfg.onset as f64
          {
            n += 1;
          }
        }
        chunk_counts[c * FRAMES_PER_WINDOW + f] = n;
      }
    }

    let mut count = vec![0u8; num_output_frames];
    let win_f = win as f64;
    for ofr in 0..num_output_frames {
      let sample_idx = (ofr as f64 + 0.5) * SAMPLES_PER_OUTPUT_FRAME;
      // Find chunks whose [start, start+win) covers sample_idx.
      let lo_chunk = if sample_idx >= win_f {
        ((sample_idx - win_f) / step as f64).ceil() as usize
      } else {
        0
      };
      let hi_chunk = ((sample_idx / step as f64).floor() as usize + 1).min(num_chunks);
      let mut weighted_sum = 0.0_f64;
      let mut weight_total = 0.0_f64;
      for cc in lo_chunk..hi_chunk {
        let chunk_start_sample = (cc * step) as f64;
        let chunk_relative_sample = sample_idx - chunk_start_sample;
        if chunk_relative_sample < 0.0 || chunk_relative_sample >= win_f {
          continue;
        }
        let chunk_frame = (chunk_relative_sample / SAMPLES_PER_OUTPUT_FRAME) as usize;
        if chunk_frame >= FRAMES_PER_WINDOW {
          continue;
        }
        // Hamming weight: w(τ) = 0.54 − 0.46·cos(2π·τ) where
        // τ = chunk_relative_sample / win ∈ [0, 1). Edges of a chunk
        // get less weight than the center, matching pyannote's
        // `Inference.aggregate(hamming=True)`. Reduces the Gibbs
        // ringing at chunk-boundary frames.
        let tau = chunk_relative_sample / win_f;
        let w = 0.54 - 0.46 * (std::f64::consts::TAU * tau).cos();
        weighted_sum += chunk_counts[cc * FRAMES_PER_WINDOW + chunk_frame] as f64 * w;
        weight_total += w;
      }
      let avg = if weight_total > 0.0 {
        weighted_sum / weight_total
      } else {
        0.0
      };
      count[ofr] = avg.round().clamp(0.0, u8::MAX as f64) as u8;
    }

    let chunks_sw = SlidingWindow {
      start: 0.0,
      duration: WINDOW_SAMPLES as f64 / SAMPLE_RATE_HZ as f64,
      step: cfg.step_samples as f64 / SAMPLE_RATE_HZ as f64,
    };
    let frames_sw = SlidingWindow {
      start: 0.0,
      duration: SAMPLES_PER_OUTPUT_FRAME / SAMPLE_RATE_HZ as f64,
      step: SAMPLES_PER_OUTPUT_FRAME / SAMPLE_RATE_HZ as f64,
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
