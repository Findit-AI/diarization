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

  /// Reset for a new session. Drops ALL session-local state including
  /// collected per-activity context.
  ///
  /// Spec §4.4 / rev-7 §11.12 + Codex review post-rev-9:
  /// - [`Segmenter`] cleared (generation bump → stale window-ids reject).
  /// - [`Clusterer`] cleared (`speaker_id` resets to 0).
  /// - Audio buffer drained; `audio_base = 0`; `total_samples_pushed = 0`.
  /// - Per-frame stitching state dropped; open per-cluster runs are
  ///   DROPPED (NOT auto-emitted — call `finish_stream` first if you
  ///   want them out).
  /// - `collected_embeddings` dropped (rev-9 + Codex review fix:
  ///   privacy/tenant-isolation safety for pooled diarizers).
  /// - Configured options are preserved.
  ///
  /// If you want to extract collected embeddings before resetting (for
  /// offline re-clustering or audit), call
  /// [`take_collected`](Self::take_collected) first — it returns the
  /// embeddings AND clears them in one operation.
  pub fn clear(&mut self) {
    self.segmenter.clear();
    self.clusterer.clear();
    self.audio_buffer.clear();
    self.audio_base = 0;
    self.total_samples_pushed = 0;
    self.collected_embeddings.clear();
    self.reconstruct.clear();
  }

  /// Borrow the per-activity context collected during streaming.
  /// Empty until [`process_samples`](Self::process_samples) is called
  /// AND `opts.collect_embeddings = true`.
  pub fn collected_embeddings(&self) -> &[CollectedEmbedding] {
    &self.collected_embeddings
  }

  /// Drop all collected per-activity context, leaving session state
  /// (segmenter, clusterer, audio buffer, reconstruction) intact.
  ///
  /// Use [`clear`](Self::clear) for a full session reset (which now
  /// also drops collected embeddings). This method is for the rare
  /// case where a caller wants to preserve session state but reset
  /// audit data.
  pub fn clear_collected(&mut self) {
    self.collected_embeddings.clear();
  }

  /// Take the collected per-activity context, leaving the Diarizer's
  /// `collected_embeddings` empty. Useful for the offline-re-clustering
  /// pattern: extract embeddings → re-cluster → continue the session.
  ///
  /// Equivalent to `std::mem::take(&mut diarizer.collected_embeddings)`
  /// but provides a public-API entry point (the field itself is
  /// `pub(crate)`).
  pub fn take_collected(&mut self) -> Vec<CollectedEmbedding> {
    core::mem::take(&mut self.collected_embeddings)
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

  /// Finalize the stream: drain any pending segmentation work, then
  /// flush still-open per-cluster runs as `DiarizedSpan`s. After this
  /// call, `process_samples` must NOT be called again until [`clear`](Self::clear).
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
    // segment is finished → peek_next_window_start = u64::MAX → all
    // remaining frames finalize. flush_open_runs sets the boundary
    // explicitly and drains, plus closes any final open runs.
    self.reconstruct.flush_open_runs(&mut emit);
    self.trim_audio();
    Ok(())
  }

  fn drain<F>(
    &mut self,
    seg_model: &mut SegmentModel,
    embed_model: &mut EmbedModel,
    emit: &mut F,
  ) -> Result<(), Error>
  where
    F: FnMut(DiarizedSpan),
  {
    use std::collections::HashMap;

    use crate::segment::{
      SpeakerActivity, WindowId,
      options::{FRAMES_PER_WINDOW, MAX_SPEAKER_SLOTS},
    };

    // Per-WindowId batching. Action::SpeakerScores arrives BEFORE the
    // window's Activity events (Task 30 contract), but we make the
    // pump robust to any order by buffering until both have been seen.
    type RawProbs = Box<[[f32; FRAMES_PER_WINDOW]; MAX_SPEAKER_SLOTS as usize]>;
    let mut pending_scores: HashMap<WindowId, RawProbs> = HashMap::new();
    let mut pending_window_starts: HashMap<WindowId, u64> = HashMap::new();
    let mut pending_activities: HashMap<WindowId, Vec<SpeakerActivity>> = HashMap::new();
    // Track which window IDs have been observed AT ALL during this drain.
    // After the segmenter is fully drained for one drain() call, we
    // process every window that has scores — even with zero activities
    // (a silent window's scores still need to drive boundary advance).
    let mut observed: Vec<WindowId> = Vec::new();

    while let Some(action) = self.segmenter.poll() {
      match action {
        Action::NeedsInference { id, samples } => {
          let scores = seg_model.infer(&samples)?;
          self.segmenter.push_inference(id, &scores)?;
        }
        Action::SpeakerScores {
          id,
          window_start,
          raw_probs,
        } => {
          if !observed.contains(&id) {
            observed.push(id);
          }
          pending_scores.insert(id, raw_probs);
          pending_window_starts.insert(id, window_start);
        }
        Action::Activity(activity) => {
          let id = activity.window_id();
          if !observed.contains(&id) {
            observed.push(id);
          }
          pending_activities.entry(id).or_default().push(activity);
        }
        Action::VoiceSpan(_) => {
          // Diarizer doesn't currently surface VoiceSpan.
        }
      }
    }

    // Process every observed window (in arrival order — `observed` is
    // a Vec, not a HashSet) once the segmenter is drained. A window is
    // only processed after `push_inference` has fired its Activity +
    // SpeakerScores events, so by the time we get here, every observed
    // window has its scores ready.
    for id in observed {
      let Some(raw_probs) = pending_scores.remove(&id) else {
        // No scores → this WindowId only had activities but no
        // SpeakerScores. Should never happen given Task 30's
        // contract; skip defensively.
        continue;
      };
      let window_start = pending_window_starts.remove(&id).unwrap_or(0);
      let activities = pending_activities.remove(&id).unwrap_or_default();

      // Embed each activity, cluster, record (window, slot) → cluster_id.
      self.process_window_activities(id, window_start, &raw_probs, &activities, embed_model)?;

      // Integrate this window into reconstruction.
      self.reconstruct.integrate_window(
        id,
        window_start,
        &raw_probs,
        self.opts.binarize_threshold(),
      );
    }

    // Advance finalization boundary based on segmenter state.
    let next = self.segmenter.peek_next_window_start();
    self.reconstruct.advance_finalization_boundary(next);

    // Emit any finalized DiarizedSpans.
    let max_spk = self
      .opts
      .cluster_options()
      .max_speakers()
      .unwrap_or(u32::MAX);
    self.reconstruct.emit_finalized_frames(max_spk, &mut *emit);

    // Bound memory.
    self.reconstruct.evict_finalized_window_metadata();

    Ok(())
  }

  /// Process all activities for a single window: derive `keep_mask`,
  /// extract embedding via [`EmbedModel::embed_masked`], cluster the
  /// embedding, record `(window_id, slot) → cluster_id` mapping, and
  /// optionally collect the embedding for post-hoc analysis.
  ///
  /// Implements the §5.8 fall-back chain:
  /// 1. With `exclude_overlap = true`: try the clean (overlap-excluded)
  ///    mask first.
  /// 2. If `embed_masked` returns `Error::InvalidClip`, retry with the
  ///    speaker-only mask.
  /// 3. If THAT also fails with `InvalidClip`, skip the activity (matches
  ///    pyannote `speaker_verification.py:611-612`).
  ///
  /// With `exclude_overlap = false`: skip step 1, use a `[true; len]`
  /// mask directly (equivalent to `EmbedModel::embed`).
  fn process_window_activities(
    &mut self,
    window_id: crate::segment::WindowId,
    window_start: u64,
    raw_probs: &[[f32; crate::segment::options::FRAMES_PER_WINDOW];
       crate::segment::options::MAX_SPEAKER_SLOTS as usize],
    activities: &[crate::segment::SpeakerActivity],
    embed_model: &mut EmbedModel,
  ) -> Result<(), Error> {
    use crate::{
      diarizer::overlap::{
        binarize_per_frame, decide_keep_mask, frame_mask_to_sample_keep_mask, speaker_mask_for_slot,
      },
      embed::Error as EmbedError,
    };

    for activity in activities {
      let s = activity.speaker_slot();
      let s0 = activity.range().start_pts() as u64;
      let s1 = activity.range().end_pts() as u64;

      // Slice the audio for this activity.
      let activity_samples = self.slice_audio(s0, s1)?;

      // Build the initial mask (clean if exclude_overlap = true; all-true otherwise).
      let want_clean = self.opts.exclude_overlap();
      let (mask, used_clean_initial) = if want_clean {
        let dec = decide_keep_mask(
          raw_probs,
          self.opts.binarize_threshold(),
          s,
          window_start,
          s0,
          s1,
        );
        (dec.keep_mask, dec.used_clean)
      } else {
        let all_true = vec![true; activity_samples.len()];
        (all_true, false)
      };

      // Try embed_masked. On InvalidClip from a clean mask, fall back
      // to a speaker-only mask. On InvalidClip even on speaker-only,
      // skip the activity (pyannote semantics).
      let (embedding_opt, used_clean) = match embed_model.embed_masked(&activity_samples, &mask) {
        Ok(r) => (Some(*r.embedding()), used_clean_initial),
        Err(EmbedError::InvalidClip { .. }) if used_clean_initial => {
          // Fall back to speaker-only mask.
          let binarized = binarize_per_frame(raw_probs, self.opts.binarize_threshold());
          let speaker = speaker_mask_for_slot(&binarized, s);
          let speaker_keep = frame_mask_to_sample_keep_mask(&speaker, window_start, s0, s1);
          match embed_model.embed_masked(&activity_samples, &speaker_keep) {
            Ok(r) => (Some(*r.embedding()), false),
            Err(EmbedError::InvalidClip { .. }) => (None, false), // skip
            Err(e) => return Err(e.into()),
          }
        }
        Err(EmbedError::InvalidClip { .. }) => {
          // First attempt was speaker-only or all-true; persistent failure → skip.
          (None, false)
        }
        Err(e) => return Err(e.into()),
      };

      let Some(embedding) = embedding_opt else {
        continue; // skipped (matches pyannote skip semantics)
      };

      // Cluster.
      let assignment = self.clusterer.submit(&embedding)?;
      let cluster_id = assignment.speaker_id();

      // Record one ActivityCluster per activity (Codex review CRITICAL:
      // pyannote hysteresis can emit MULTIPLE disjoint Activity events
      // for the same (window_id, slot) within one window — each gets
      // its own embedding+cluster, and reconstruction must apply each
      // cluster only to its activity's frame range, not to every frame
      // where the slot's raw probability is positive).
      let activity_start_in_win = s0.saturating_sub(window_start);
      let activity_end_in_win = s1.saturating_sub(window_start);
      let frame_lo_in_window = (activity_start_in_win
        * crate::segment::options::FRAMES_PER_WINDOW as u64
        / crate::segment::options::WINDOW_SAMPLES as u64) as u32;
      let frame_hi_in_window =
        (activity_end_in_win * crate::segment::options::FRAMES_PER_WINDOW as u64)
          .div_ceil(crate::segment::options::WINDOW_SAMPLES as u64) as u32;
      let frame_hi_in_window =
        frame_hi_in_window.min(crate::segment::options::FRAMES_PER_WINDOW as u32);
      // Record the new activity AND its index in the per-cluster index so
      // emit_finalized_frames only scans activities for the current cluster
      // (Codex review MEDIUM).
      let activity_index = self.reconstruct.activities.len();
      self
        .reconstruct
        .activities
        .push(crate::diarizer::reconstruct::ActivityCluster {
          window_id,
          slot: s,
          frame_lo_in_window,
          frame_hi_in_window,
          cluster_id,
          used_clean_mask: used_clean,
        });
      self
        .reconstruct
        .activities_by_cluster
        .entry(cluster_id)
        .or_default()
        .push(activity_index);

      // Optional collected_embeddings.
      if self.opts.collect_embeddings() {
        self.collected_embeddings.push(CollectedEmbedding {
          range: activity.range(),
          embedding,
          online_speaker_id: cluster_id,
          speaker_slot: s,
          used_clean_mask: used_clean,
        });
      }
    }
    Ok(())
  }
}
