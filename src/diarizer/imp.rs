//! `Diarizer` state holder. Spec §4.4 / §5.7-§5.12.

extern crate alloc;

use alloc::{collections::VecDeque, vec::Vec};

use crate::{
  cluster::{Clusterer, SpeakerCentroid},
  diarizer::{
    builder::{DiarizerBuilder, DiarizerOptions},
    span::{CollectedEmbedding, DiarizedSpan},
  },
  segment::Segmenter,
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

#[allow(dead_code)]
fn _diarized_span_unused_warning_suppression(_: DiarizedSpan) {}
