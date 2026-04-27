//! `Diarizer` state holder. Spec §4.4 / §5.7-§5.12.

extern crate alloc;

use alloc::{collections::VecDeque, vec::Vec};

use crate::{
  cluster::{Clusterer, SpeakerCentroid},
  diarizer::{
    builder::{DiarizerBuilder, DiarizerOptions},
    error::Error,
    span::{CollectedEmbedding, DiarizedSpan},
  },
  embed::EmbedModel,
  segment::{Action, SegmentModel, Segmenter, WINDOW_SAMPLES},
};

/// Top-level streaming diarizer.
///
/// Owns a [`Segmenter`], a [`Clusterer`], a rolling audio buffer, and
/// (under `#[cfg(feature = "ort")]`) the per-frame reconstruction state
/// machine.
///
/// **Threading:** `Send + Sync` auto-derived (the compile-time assertion
/// in `mod.rs` enforces this). Single-stream — use one `Diarizer` per
/// concurrent diarization session. Pass `&mut SegmentModel` and
/// `&mut EmbedModel` per `process_samples`/`finish_stream` call (rev-2
/// borrow-by-`&mut` pattern; spec §11.0). Both models are `!Sync`.
pub struct Diarizer {
  pub(crate) opts: DiarizerOptions,
  pub(crate) segmenter: Segmenter,
  pub(crate) clusterer: Clusterer,
  /// Rolling audio buffer indexed by absolute samples. Element 0
  /// corresponds to absolute sample [`audio_base`](Self::audio_base).
  /// Trim policy: keep the last `dia::segment::WINDOW_SAMPLES` samples
  /// (§5.7 / §11.5).
  pub(crate) audio_buffer: VecDeque<f32>,
  pub(crate) audio_base: u64,
  pub(crate) total_samples_pushed: u64,
  /// Per-activity context, populated when `opts.collect_embeddings = true`.
  pub(crate) collected_embeddings: Vec<CollectedEmbedding>,
  /// Per-frame reconstruction state (filled in by Phase 10).
  pub(crate) reconstruct: crate::diarizer::reconstruct::ReconstructState,
}

impl Diarizer {
  /// Construct a new `Diarizer` with the given options.
  pub fn new(opts: DiarizerOptions) -> Self {
    let segmenter = Segmenter::new(opts.segment_options().clone());
    let clusterer = Clusterer::new(opts.cluster_options().clone());
    Self {
      opts,
      segmenter,
      clusterer,
      audio_buffer: VecDeque::new(),
      audio_base: 0,
      total_samples_pushed: 0,
      collected_embeddings: Vec::new(),
      reconstruct: crate::diarizer::reconstruct::ReconstructState::new(),
    }
  }

  /// Convenience constructor returning a [`DiarizerBuilder`].
  pub fn builder() -> DiarizerBuilder {
    DiarizerBuilder::new()
  }

  /// Borrow the configured options.
  pub fn options(&self) -> &DiarizerOptions {
    &self.opts
  }

  /// Reset for a new session. Spec §4.4 / rev-7 §11.12:
  /// - [`Segmenter`] cleared (generation bump → stale window-ids reject).
  /// - [`Clusterer`] cleared (`speaker_id` resets to 0).
  /// - Audio buffer drained; `audio_base = 0`; `total_samples_pushed = 0`.
  /// - Per-frame stitching state dropped; open per-cluster runs are
  ///   DROPPED (NOT auto-emitted — call `finish_stream` first if you
  ///   want them out).
  /// - `collected_embeddings` is **NOT** cleared. Call
  ///   [`clear_collected`](Self::clear_collected) to drop those.
  /// - Configured options are preserved.
  pub fn clear(&mut self) {
    self.segmenter.clear();
    self.clusterer.clear();
    self.audio_buffer.clear();
    self.audio_base = 0;
    self.total_samples_pushed = 0;
    self.reconstruct.clear();
  }

  /// Borrow the per-activity context collected during streaming.
  /// Empty until [`process_samples`](Self::process_samples) is called
  /// AND `opts.collect_embeddings = true`.
  pub fn collected_embeddings(&self) -> &[CollectedEmbedding] {
    &self.collected_embeddings
  }

  /// Drop all collected per-activity context.
  pub fn clear_collected(&mut self) {
    self.collected_embeddings.clear();
  }

  /// Number of segmentation windows awaiting `push_inference`.
  /// Bounded by the segmenter's pending-window queue.
  pub fn pending_inferences(&self) -> usize {
    self.segmenter.pending_inferences()
  }

  /// Number of audio samples currently retained for activity-range slicing.
  pub fn buffered_samples(&self) -> usize {
    self.audio_buffer.len()
  }

  /// Number of frames buffered for per-cluster reconstruction.
  /// Always `0` until Phase 10 wires the state machine.
  pub fn buffered_frames(&self) -> usize {
    self.reconstruct.buffered_frame_count()
  }

  /// Cumulative count of samples ever passed to
  /// [`process_samples`](Self::process_samples) since the last
  /// [`clear`](Self::clear). Monotonic; never decremented except by
  /// `clear()`. Caller-side anchor for VAD-to-original-time mapping
  /// per spec §11.12.
  pub fn total_samples_pushed(&self) -> u64 {
    self.total_samples_pushed
  }

  /// Number of distinct speakers opened by the online clusterer.
  pub fn num_speakers(&self) -> usize {
    self.clusterer.num_speakers()
  }

  /// Snapshot of the online clusterer's speakers.
  pub fn speakers(&self) -> Vec<SpeakerCentroid> {
    self.clusterer.speakers()
  }
}

impl Diarizer {
  /// Push samples into the audio buffer and the segmenter; advance the
  /// total-samples counter. **Trim is deferred** until after the pump
  /// completes (§5.7) so activities emitted mid-pump can still slice
  /// their audio range from the buffer.
  pub(crate) fn push_audio(&mut self, samples: &[f32]) {
    self.audio_buffer.extend(samples.iter().copied());
    self.segmenter.push_samples(samples);
    self.total_samples_pushed += samples.len() as u64;
  }

  /// Trim audio buffer to retain the last `WINDOW_SAMPLES` samples
  /// (§5.7 / §11.5). Idempotent.
  pub(crate) fn trim_audio(&mut self) {
    let win = WINDOW_SAMPLES as u64;
    if self.total_samples_pushed > win {
      let target_base = self.total_samples_pushed - win;
      let drop_n = target_base.saturating_sub(self.audio_base) as usize;
      let drop_n = drop_n.min(self.audio_buffer.len());
      for _ in 0..drop_n {
        self.audio_buffer.pop_front();
      }
      self.audio_base += drop_n as u64;
    }
  }

  /// Slice the audio buffer for an absolute-sample range `[s0, s1)`.
  ///
  /// Returns [`Error::Internal`] with [`InternalError::AudioBufferUnderflow`]
  /// or [`InternalError::AudioBufferOverrun`] if the range falls outside
  /// the buffer's retained window. Defense-in-depth: should be unreachable
  /// per the §5.7 segment-contract trim discipline.
  ///
  /// [`InternalError::AudioBufferUnderflow`]: crate::diarizer::InternalError::AudioBufferUnderflow
  /// [`InternalError::AudioBufferOverrun`]: crate::diarizer::InternalError::AudioBufferOverrun
  #[allow(dead_code)] // Consumer (Phase-9 embed extraction) lands in a later task.
  pub(crate) fn slice_audio(&self, s0: u64, s1: u64) -> Result<Vec<f32>, Error> {
    use crate::diarizer::error::InternalError;
    if s0 < self.audio_base {
      return Err(Error::Internal(InternalError::AudioBufferUnderflow {
        activity_start: s0,
        audio_base: self.audio_base,
      }));
    }
    let end = self.audio_base + self.audio_buffer.len() as u64;
    if s1 > end {
      return Err(Error::Internal(InternalError::AudioBufferOverrun {
        activity_end: s1,
        audio_end: end,
      }));
    }
    let lo = (s0 - self.audio_base) as usize;
    let hi = (s1 - self.audio_base) as usize;
    Ok(self.audio_buffer.range(lo..hi).copied().collect())
  }
}

impl Diarizer {
  /// Process a chunk of samples. Pushes into the audio buffer + segmenter,
  /// pumps inference and per-window activities, and emits any closed
  /// [`DiarizedSpan`]s via `emit`.
  ///
  /// **Phase 8 skeleton:** the segmenter is fully pumped (segment-model
  /// inference runs; per-window activities are observed). Embed, cluster,
  /// and reconstruct are NO-OPs awaiting Phases 9-11; no `DiarizedSpan`s
  /// are produced yet.
  pub fn process_samples<F>(
    &mut self,
    seg_model: &mut SegmentModel,
    embed_model: &mut EmbedModel,
    samples: &[f32],
    mut emit: F,
  ) -> Result<(), Error>
  where
    F: FnMut(DiarizedSpan),
  {
    self.push_audio(samples);
    self.drain(seg_model, embed_model, &mut emit)?;
    self.trim_audio();
    Ok(())
  }

  /// Finalize the stream. Flushes any tail-anchor inference plus
  /// (in Phase 11) any open per-cluster runs.
  pub fn finish_stream<F>(
    &mut self,
    seg_model: &mut SegmentModel,
    embed_model: &mut EmbedModel,
    mut emit: F,
  ) -> Result<(), Error>
  where
    F: FnMut(DiarizedSpan),
  {
    self.segmenter.finish();
    self.drain(seg_model, embed_model, &mut emit)?;
    // Phase 11 will add: self.reconstruct.flush_open_runs(&mut emit);
    Ok(())
  }

  fn drain<F>(
    &mut self,
    seg_model: &mut SegmentModel,
    _embed_model: &mut EmbedModel,
    _emit: &mut F,
  ) -> Result<(), Error>
  where
    F: FnMut(DiarizedSpan),
  {
    while let Some(action) = self.segmenter.poll() {
      match action {
        Action::NeedsInference { id, samples } => {
          let scores = seg_model.infer(&samples)?;
          self.segmenter.push_inference(id, &scores)?;
        }
        Action::SpeakerScores { .. } => {
          // Phase 11: buffer per WindowId for the reconstruct state machine.
        }
        Action::Activity(_) => {
          // Phase 11: cluster + reconstruct.
        }
        Action::VoiceSpan(_) => {
          // Diarizer doesn't surface VoiceSpan currently.
        }
      }
    }
    Ok(())
  }
}
