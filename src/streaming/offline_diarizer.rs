//! Voice-range-driven streaming diarizer that produces pyannote-
//! equivalent global speaker assignments.
//!
//! Architecture: [`StreamingOfflineDiarizer::push_voice_range`] runs
//! the heavy stages 1+2 (sliding-window segmentation + masked
//! embedding) on each VAD-emitted voice range and accumulates the
//! derived tensors. [`StreamingOfflineDiarizer::finalize`] runs the
//! single global pyannote `cluster_vbx` pass (PLDA + AHC + VBx +
//! centroid + Hungarian) on the union of accumulated chunks, then
//! reconstructs per-range frame-level diarization and maps spans
//! back to the original timeline.
//!
//! ## Why not per-range clustering with cross-range bank
//!
//! The previous `StreamingDiarizationPipeline` ran full pyannote
//! offline diarization on each voice range independently and matched
//! cluster centroids across ranges via cosine bank. Two problems:
//!
//! 1. **Per-range AHC has no cross-range context.** A speaker who
//!    appears only briefly in range A and dominantly in range B can
//!    be merged with a different speaker in A (because A doesn't
//!    have enough evidence) and become a separate cluster from B.
//! 2. **Cosine bank in raw-embedding space is noisier than PLDA**.
//!    Pyannote clusters in PLDA-projected space because PLDA
//!    suppresses channel/session variance. Raw cosine bank inherits
//!    the unsuppressed variance and over- or under-merges.
//!
//! Running global AHC + VBx on the union of all voice ranges' chunks
//! mirrors what pyannote does on the full recording — each voice
//! range contributes its (chunk, slot) embeddings to one global
//! clustering, so cross-range identity is established by the same
//! algorithm pyannote uses, not a side-channel cosine bank.
//!
//! ## Memory & latency
//!
//! Per chunk: 589 frames × 3 slots × 8 B (segmentations) + 3 slots
//! × 256 dims × 4 B (raw embeddings) + ~10 KB count tensor ≈ 17 KB.
//! For 1 hour of audio with the community-1 1 s chunk step that's
//! ~3600 chunks ≈ 60 MB of accumulated tensors — bounded and small
//! relative to the PCM buffer the previous pipeline retained.
//!
//! Latency is `finalize`-bound: the offline clustering pass scales
//! roughly as O(num_train²) for AHC and O(num_train · plda_dim²) for
//! VBx, where `num_train` ≈ active (chunk, slot) pairs. For a 1 h
//! conversation that's ~10 000 pairs — multi-second clustering. For
//! near-realtime indexing this is acceptable; for sub-range latency
//! see [`crate::diarizer::Diarizer`].

use std::sync::Arc;

use crate::{
  aggregate::try_count_pyannote,
  embed::{EMBEDDING_DIM, EmbedModel},
  offline::{OfflineInput, OwnedPipelineOptions, diarize_offline},
  plda::PldaTransform,
  reconstruct::{
    ReconstructInput, RttmSpan, SlidingWindow, discrete_to_spans, reconstruct as reconstruct_grid,
  },
  segment::{
    FRAMES_PER_WINDOW, POWERSET_CLASSES, PYANNOTE_FRAME_DURATION_S, PYANNOTE_FRAME_STEP_S,
    SAMPLE_RATE_HZ, SegmentModel, WINDOW_SAMPLES,
    powerset::{powerset_to_speakers_hard, softmax_row},
  },
};

/// Number of speaker slots per chunk. Same as
/// [`crate::offline::SLOTS_PER_CHUNK`]; duplicated here for module
/// independence.
const SLOTS_PER_CHUNK: usize = 3;

/// Errors from the streaming offline diarizer.
#[derive(Debug, thiserror::Error)]
pub enum StreamingError {
  #[error("streaming: shape: {0}")]
  Shape(#[from] StreamingShapeError),
  #[error("streaming: segment: {0}")]
  Segment(String),
  #[error("streaming: embed: {0}")]
  Embed(String),
  #[error("streaming: offline: {0}")]
  Offline(#[from] crate::offline::Error),
  #[error("streaming: reconstruct: {0}")]
  Reconstruct(#[from] crate::reconstruct::Error),
  /// Propagated from `aggregate::try_count_pyannote` when the count
  /// tensor cannot be computed (e.g. NaN/inf `onset` from a
  /// misconfigured `OwnedPipelineOptions`). Replaces a panic path
  /// through the infallible `count_pyannote` wrapper.
  #[error("streaming: aggregate: {0}")]
  Aggregate(#[from] crate::aggregate::Error),
}

/// Specific shape-violation reasons for [`StreamingError::Shape`].
#[derive(Debug, thiserror::Error, Clone, Copy, PartialEq, Eq)]
pub enum StreamingShapeError {
  #[error("voice range samples is empty")]
  EmptyVoiceRange,
  #[error("step_samples must be > 0")]
  ZeroStepSamples,
  #[error("all accumulated voice ranges are empty")]
  AllRangesEmpty,
}

/// Configuration for [`StreamingOfflineDiarizer`].
#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StreamingOfflineOptions {
  #[cfg_attr(feature = "serde", serde(default))]
  diarization: OwnedPipelineOptions,
}

impl StreamingOfflineOptions {
  /// Construct with `community-1` diarization defaults.
  pub const fn new() -> Self {
    Self {
      diarization: OwnedPipelineOptions::new(),
    }
  }

  /// Borrow the inner diarization parameters.
  pub const fn diarization(&self) -> &OwnedPipelineOptions {
    &self.diarization
  }

  /// Builder: replace the diarization parameters.
  #[must_use]
  pub const fn with_diarization(mut self, diarization: OwnedPipelineOptions) -> Self {
    self.diarization = diarization;
    self
  }
}

/// One diarized span in the original audio timeline.
#[derive(Debug, Clone)]
pub struct DiarizedSpan {
  start_sample: u64,
  end_sample: u64,
  speaker_id: u32,
}

impl DiarizedSpan {
  /// Construct.
  pub const fn new(start_sample: u64, end_sample: u64, speaker_id: u32) -> Self {
    Self {
      start_sample,
      end_sample,
      speaker_id,
    }
  }

  /// Absolute start sample (relative to the start of the input
  /// audio stream that drove `push_voice_range`).
  pub const fn start_sample(&self) -> u64 {
    self.start_sample
  }

  /// Absolute end sample.
  pub const fn end_sample(&self) -> u64 {
    self.end_sample
  }

  /// Globally-tracked speaker id, consistent across all voice
  /// ranges pushed before `finalize`.
  pub const fn speaker_id(&self) -> u32 {
    self.speaker_id
  }
}

/// Voice-range-driven streaming diarizer.
///
/// Caller drives VAD externally and pushes one voice range per VAD
/// segment. At end-of-stream, [`finalize`](Self::finalize) runs the
/// global clustering pass and returns spans on the original
/// timeline.
pub struct StreamingOfflineDiarizer {
  options: StreamingOfflineOptions,
  ranges: Vec<AccumulatedRange>,
}

/// Per-voice-range derived tensors plus original-timeline anchor.
struct AccumulatedRange {
  /// Absolute sample at which this voice range starts in the
  /// original audio stream. Used to re-anchor output spans.
  abs_start_sample: u64,
  /// Number of segmentation chunks emitted within this range.
  num_chunks: usize,
  /// Per-(chunk, frame, slot) segmentation activity, flattened
  /// `[c][f][s]`. Length `num_chunks * FRAMES_PER_WINDOW *
  /// SLOTS_PER_CHUNK`. f64 to match pyannote internals.
  segmentations: Vec<f64>,
  /// Per-(chunk, slot) raw f32 embeddings, flattened `[c][s][d]`.
  /// Length `num_chunks * SLOTS_PER_CHUNK * EMBEDDING_DIM`.
  raw_embeddings: Vec<f32>,
  /// Per-output-frame instantaneous speaker count, computed by
  /// `aggregate::count_pyannote` on this range's segmentations.
  /// `Arc<[u8]>` to avoid a copy from `count_pyannote`'s output;
  /// also lets `finalize` cheaply hand the per-range buffer to
  /// downstream stages.
  count: Arc<[u8]>,
  /// Output-frame sliding window (local to this range, start = 0).
  frames_sw_local: SlidingWindow,
  /// Chunk-level sliding window (local to this range, start = 0).
  chunks_sw_local: SlidingWindow,
}

impl StreamingOfflineDiarizer {
  pub fn new(options: StreamingOfflineOptions) -> Self {
    Self {
      options,
      ranges: Vec::new(),
    }
  }

  /// Borrow the options.
  pub fn options(&self) -> &StreamingOfflineOptions {
    &self.options
  }

  /// Number of voice ranges accumulated so far.
  pub fn num_ranges(&self) -> usize {
    self.ranges.len()
  }

  /// Push one voice range. Runs segmentation + embedding + count
  /// tensor computation on the supplied PCM and stores the derived
  /// tensors. Does NOT cluster — that happens at
  /// [`finalize`](Self::finalize).
  ///
  /// `abs_start_sample` is the absolute sample index in the
  /// original audio stream where this range starts; it's used at
  /// `finalize` to remap output spans back to the original timeline.
  ///
  /// # Errors
  ///
  /// - [`StreamingError::Shape`] if `samples.is_empty()` or
  ///   `step_samples == 0`.
  /// - [`StreamingError::Segment`] / [`StreamingError::Embed`] for
  ///   ONNX inference failures on the range.
  pub fn push_voice_range(
    &mut self,
    seg_model: &mut SegmentModel,
    embed_model: &mut EmbedModel,
    abs_start_sample: u64,
    samples: &[f32],
  ) -> Result<(), StreamingError> {
    let cfg = &self.options.diarization;
    if samples.is_empty() {
      return Err(StreamingShapeError::EmptyVoiceRange.into());
    }
    let win = WINDOW_SAMPLES as usize;
    let step = cfg.step_samples() as usize;
    if step == 0 {
      return Err(StreamingShapeError::ZeroStepSamples.into());
    }

    let num_chunks = if samples.len() <= win {
      1
    } else {
      (samples.len() - win).div_ceil(step) + 1
    };

    let mut padded_chunk = vec![0.0_f32; win];
    let mut segmentations: Vec<f64> = vec![0.0; num_chunks * FRAMES_PER_WINDOW * SLOTS_PER_CHUNK];

    // ── Stage 1: chunked sliding-window segmentation ───────────────
    for c in 0..num_chunks {
      let chunk_start = c * step;
      padded_chunk.fill(0.0);
      let end = (chunk_start + win).min(samples.len());
      let lo = chunk_start.min(samples.len());
      let n = end - lo;
      if n > 0 {
        padded_chunk[..n].copy_from_slice(&samples[lo..end]);
      }

      let logits = seg_model
        .infer(&padded_chunk)
        .map_err(|e| StreamingError::Segment(format!("{e}")))?;
      for f in 0..FRAMES_PER_WINDOW {
        let mut row = [0.0_f32; POWERSET_CLASSES];
        for k in 0..POWERSET_CLASSES {
          row[k] = logits[f * POWERSET_CLASSES + k];
        }
        let probs = softmax_row(&row);
        // Pyannote's `to_multilabel(soft=False)` — see the long
        // comment in `crate::offline::owned::OwnedDiarizationPipeline
        // ::run` stage 1 for the rationale.
        let speakers = powerset_to_speakers_hard(&probs);
        for s in 0..SLOTS_PER_CHUNK {
          segmentations[(c * FRAMES_PER_WINDOW + f) * SLOTS_PER_CHUNK + s] = speakers[s] as f64;
        }
      }
    }

    // ── Stage 2: per-(chunk, slot) masked embedding ────────────────
    let mut raw_embeddings: Vec<f32> = vec![0.0; num_chunks * SLOTS_PER_CHUNK * EMBEDDING_DIM];

    for c in 0..num_chunks {
      let chunk_start = c * step;
      padded_chunk.fill(0.0);
      let end = (chunk_start + win).min(samples.len());
      let lo = chunk_start.min(samples.len());
      let n = end - lo;
      if n > 0 {
        padded_chunk[..n].copy_from_slice(&samples[lo..end]);
      }

      for s in 0..SLOTS_PER_CHUNK {
        let mut frame_mask = [false; FRAMES_PER_WINDOW];
        let mut any_active = false;
        for f in 0..FRAMES_PER_WINDOW {
          let active =
            segmentations[(c * FRAMES_PER_WINDOW + f) * SLOTS_PER_CHUNK + s] >= cfg.onset() as f64;
          frame_mask[f] = active;
          any_active |= active;
        }
        if !any_active {
          for f in 0..FRAMES_PER_WINDOW {
            segmentations[(c * FRAMES_PER_WINDOW + f) * SLOTS_PER_CHUNK + s] = 0.0;
          }
          continue;
        }

        let raw = match embed_model.embed_chunk_with_frame_mask(&padded_chunk, &frame_mask) {
          Ok(v) => v,
          Err(crate::embed::Error::InvalidClip { .. })
          | Err(crate::embed::Error::DegenerateEmbedding) => {
            for f in 0..FRAMES_PER_WINDOW {
              segmentations[(c * FRAMES_PER_WINDOW + f) * SLOTS_PER_CHUNK + s] = 0.0;
            }
            continue;
          }
          Err(e) => return Err(StreamingError::Embed(format!("{e}"))),
        };
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

    // ── Stage 3: count tensor (local to this range) ────────────────
    let chunk_duration_s = WINDOW_SAMPLES as f64 / SAMPLE_RATE_HZ as f64;
    let chunk_step_s = cfg.step_samples() as f64 / SAMPLE_RATE_HZ as f64;
    let chunks_sw_local = SlidingWindow::new(0.0, chunk_duration_s, chunk_step_s);
    let frames_sw_template =
      SlidingWindow::new(0.0, PYANNOTE_FRAME_DURATION_S, PYANNOTE_FRAME_STEP_S);
    // Use the fallible variant: a malformed `onset` (NaN/inf via the
    // public `with_onset` builder) would panic the infallible wrapper
    // at `try_count_pyannote.expect(...)`. Surface it as a typed
    // `StreamingError::Aggregate` so untrusted config can never crash.
    let (count, frames_sw_local) = try_count_pyannote(
      &segmentations,
      num_chunks,
      FRAMES_PER_WINDOW,
      SLOTS_PER_CHUNK,
      cfg.onset() as f64,
      chunks_sw_local,
      frames_sw_template,
    )?
    .into_parts();

    self.ranges.push(AccumulatedRange {
      abs_start_sample,
      num_chunks,
      segmentations,
      raw_embeddings,
      count,
      frames_sw_local,
      chunks_sw_local,
    });

    Ok(())
  }

  /// Run global clustering on the union of accumulated voice ranges
  /// and return original-timeline spans.
  ///
  /// Operationally:
  /// 1. Concatenate all ranges' segmentations / raw_embeddings into
  ///    a single `(total_chunks, FRAMES_PER_WINDOW, SLOTS_PER_CHUNK)`
  ///    tensor and a single `(total_chunks, SLOTS_PER_CHUNK,
  ///    EMBEDDING_DIM)` embedding tensor.
  /// 2. Concatenate count tensors. The chunks_sw passed to
  ///    `diarize_offline` is irrelevant for the clustering stages
  ///    (they ignore timing); we pass the first range's chunks_sw
  ///    so the output's reconstruct stage sees a valid SlidingWindow.
  ///    We then re-run reconstruct PER RANGE with each range's local
  ///    timing and the corresponding slice of `hard_clusters`.
  /// 3. Per range, build spans via `discrete_to_spans` and offset
  ///    by `abs_start_sample / SR`.
  ///
  /// # Errors
  ///
  /// - [`StreamingError::Shape`] if no voice ranges have been
  ///   pushed or any range's chunk count is zero.
  /// - All other errors propagate from `diarize_offline` /
  ///   `reconstruct`.
  pub fn finalize(&self, plda: &PldaTransform) -> Result<Arc<[DiarizedSpan]>, StreamingError> {
    if self.ranges.is_empty() {
      return Ok(Arc::from([] as [DiarizedSpan; 0]));
    }
    let total_chunks: usize = self.ranges.iter().map(|r| r.num_chunks).sum();
    if total_chunks == 0 {
      return Err(StreamingShapeError::AllRangesEmpty.into());
    }

    // ── 1. Concatenate per-range tensors ───────────────────────────
    let mut all_segs: Vec<f64> =
      Vec::with_capacity(total_chunks * FRAMES_PER_WINDOW * SLOTS_PER_CHUNK);
    let mut all_emb: Vec<f32> = Vec::with_capacity(total_chunks * SLOTS_PER_CHUNK * EMBEDDING_DIM);
    for r in &self.ranges {
      all_segs.extend_from_slice(&r.segmentations);
      all_emb.extend_from_slice(&r.raw_embeddings);
    }

    // ── 2. Concatenate count tensors (per-range adjacent in output) ─
    let total_output_frames: usize = self.ranges.iter().map(|r| r.count.len()).sum();
    let mut all_count: Vec<u8> = Vec::with_capacity(total_output_frames);
    for r in &self.ranges {
      all_count.extend_from_slice(&r.count);
    }

    // ── 3. Run global cluster_vbx via diarize_offline ──────────────
    //
    // `diarize_offline`'s reconstruct stage uses `chunks_sw` /
    // `frames_sw` to map per-chunk frames onto the global output
    // grid. With our concatenated chunks (which have non-uniform
    // gaps in absolute time), this global reconstruct would emit
    // garbage timings. So we ignore its reconstruct output and
    // re-reconstruct per range below.
    let cfg = &self.options.diarization;
    let chunks_sw_global = self.ranges[0].chunks_sw_local;
    let frames_sw_global = self.ranges[0].frames_sw_local;
    let input = OfflineInput::new(
      &all_emb,
      total_chunks,
      SLOTS_PER_CHUNK,
      &all_segs,
      FRAMES_PER_WINDOW,
      &all_count,
      total_output_frames,
      chunks_sw_global,
      frames_sw_global,
      plda,
    )
    .with_threshold(cfg.threshold())
    .with_fa(cfg.fa())
    .with_fb(cfg.fb())
    .with_max_iters(cfg.max_iters())
    .with_min_duration_off(cfg.min_duration_off())
    .with_smoothing_epsilon(cfg.smoothing_epsilon());
    let offline_out = diarize_offline(&input)?;
    let hard_clusters = offline_out.hard_clusters();
    let num_clusters = offline_out.num_clusters();
    debug_assert_eq!(hard_clusters.len(), total_chunks);

    // ── 4. Per-range reconstruct → spans → original timeline ───────
    //
    // `reconstruct` sizes its output grid as `(num_output_frames,
    // num_clusters_local)` where `num_clusters_local =
    // max(max(hard_clusters_slice) + 1, max(count_slice), 1)`. We
    // recompute it the same way so `discrete_to_spans`'s shape
    // assertion holds. Span cluster ids are the global hard-cluster
    // ids regardless of `num_clusters_local`, so cross-range identity
    // is preserved automatically.
    let _ = num_clusters; // global count not used here; per-range computed below.
    let mut all_spans: Vec<DiarizedSpan> = Vec::new();
    let sr = SAMPLE_RATE_HZ as f64;
    let mut chunk_offset = 0usize;
    for r in &self.ranges {
      let hc_slice = &hard_clusters[chunk_offset..chunk_offset + r.num_chunks];
      chunk_offset += r.num_chunks;

      let recon_input = ReconstructInput::new(
        &r.segmentations,
        r.num_chunks,
        FRAMES_PER_WINDOW,
        SLOTS_PER_CHUNK,
        hc_slice,
        &r.count,
        r.count.len(),
        r.chunks_sw_local,
        r.frames_sw_local,
      )
      .with_smoothing_epsilon(cfg.smoothing_epsilon());
      let discrete = reconstruct_grid(&recon_input)?;

      let max_cluster_local = hc_slice
        .iter()
        .flat_map(|row| row.iter())
        .copied()
        .max()
        .unwrap_or(-1);
      let max_count_local = r.count.iter().copied().max().unwrap_or(0) as usize;
      let num_clusters_local = if max_cluster_local < 0 {
        // No assigned clusters → reconstruct returns a 1D
        // `num_output_frames`-length zero vector (see
        // `reconstruct::algo::reconstruct` early-out at
        // `max_cluster < 0`). `discrete_to_spans` would then assert
        // on `grid.len() == num_output_frames * num_clusters`, so
        // skip the call entirely.
        debug_assert_eq!(discrete.len(), r.count.len());
        continue;
      } else {
        ((max_cluster_local + 1) as usize).max(max_count_local.max(1))
      };

      let local_spans: Vec<RttmSpan> = discrete_to_spans(
        &discrete,
        r.count.len(),
        num_clusters_local,
        r.frames_sw_local,
        cfg.min_duration_off(),
      );

      for span in local_spans {
        let start_off_samples = (span.start() * sr).max(0.0) as u64;
        let dur_samples = (span.duration() * sr).max(0.0) as u64;
        all_spans.push(DiarizedSpan {
          start_sample: r.abs_start_sample.saturating_add(start_off_samples),
          end_sample: r
            .abs_start_sample
            .saturating_add(start_off_samples)
            .saturating_add(dur_samples),
          speaker_id: span.cluster() as u32,
        });
      }
    }

    // Sort by start time so callers can stream the output in order.
    all_spans.sort_by_key(|s| s.start_sample);
    // One-time `Vec`→`Arc<[T]>` copy at the boundary. `all_spans` is
    // built by `Vec::push` because span count is unknown a-priori
    // (it depends on per-range `discrete_to_spans` output); converting
    // to `Arc<[DiarizedSpan]>` lets downstream consumers fan out
    // cheaply via `Arc::clone`.
    Ok(Arc::from(all_spans))
  }

  /// Drop accumulated tensors. Useful for reusing the same diarizer
  /// across multiple sessions. Does not reset speaker-id assignment
  /// since IDs are decided at `finalize`-time, not held as state.
  pub fn reset(&mut self) {
    self.ranges.clear();
  }
}
