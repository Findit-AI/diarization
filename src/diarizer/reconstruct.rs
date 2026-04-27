//! Per-frame stitching state machine for [`crate::diarizer::Diarizer`].
//! Spec §5.9-§5.11.
//!
//! Tracks per-frame per-cluster activation overlap-add (rev-3 sum, NOT
//! mean — divisor lives in [`FrameCount::activation_chunk_count`]),
//! per-frame instantaneous-speaker-count (mean-with-warm-up-trim per
//! pyannote `speaker_count(warm_up=(0.1, 0.1))`), and per-cluster
//! open-run state for RLE-to-[`DiarizedSpan`](crate::diarizer::DiarizedSpan)
//! emission.
//!
//! **Phase 10:** this file currently defines TYPES only. The algorithmic
//! methods (`integrate_window`, `emit_finalized_frames`, `flush_open_runs`)
//! land in Tasks 40-42.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

use crate::segment::{
  WindowId,
  options::{FRAMES_PER_WINDOW, WINDOW_SAMPLES},
};

/// Number of frames trimmed from the LEFT of each window before
/// counting toward instantaneous-speaker count.
///
/// Matches pyannote `speaker_count(warm_up=(0.1, 0.1))`:
/// `round(589 * 0.1) = 58.9 → 59`.
#[allow(dead_code)] // consumed by integrate_window in Task 40
pub(crate) const SPEAKER_COUNT_WARM_UP_FRAMES_LEFT: u32 = 59;

/// Number of frames trimmed from the RIGHT of each window before
/// counting toward instantaneous-speaker count.
#[allow(dead_code)] // consumed by integrate_window in Task 40
pub(crate) const SPEAKER_COUNT_WARM_UP_FRAMES_RIGHT: u32 = 59;

/// `frame_idx (u64) → sample (u64)` helper. Bit-for-bit equivalent to
/// [`crate::segment::stitch::frame_to_sample`] but operates in `u64`
/// throughout to avoid the `u32` truncating cast on long sessions
/// (sessions > ~2.7 days at 16 kHz exceed `u32::MAX` samples).
///
/// Spec §15 #54 tracks folding back into `dia::segment` once a u64
/// version lands there.
#[allow(dead_code)] // consumed by emit_finalized_frames / flush_open_runs in Tasks 41-42
pub(crate) const fn frame_to_sample_u64(frame_idx: u64) -> u64 {
  // Mirror of segment's formula:
  //   frame_to_sample(f) = (f * WINDOW_SAMPLES + FRAMES_PER_WINDOW/2) / FRAMES_PER_WINDOW
  let n = frame_idx * WINDOW_SAMPLES as u64;
  let half = (FRAMES_PER_WINDOW as u64) / 2;
  (n + half) / FRAMES_PER_WINDOW as u64
}

/// `sample_idx (u64) → frame_idx (u64)`. Same formula as
/// [`crate::segment::stitch::frame_index_of`] but kept local for
/// symmetry with [`frame_to_sample_u64`].
#[allow(dead_code)] // consumed by integrate_window in Task 40
pub(crate) const fn frame_index_of(sample_idx: u64) -> u64 {
  sample_idx * (FRAMES_PER_WINDOW as u64) / (WINDOW_SAMPLES as u64)
}

/// Per-frame bookkeeping for the reconstruction accumulator (spec §5.9 step C).
#[allow(dead_code)] // fields consumed by integrate_window / emit_finalized_frames in Tasks 40-41
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct FrameCount {
  /// Number of windows whose un-trimmed frames contributed to this
  /// frame's per-cluster activation accumulator. Used for §5.11
  /// `average_activation` normalization (`activation_sum / activation_chunk_count`).
  pub(crate) activation_chunk_count: u32,
  /// Sum across windows of `count(speakers > threshold at this frame)`,
  /// taken only from frames OUTSIDE the warm-up margin (§5.10).
  pub(crate) count_sum: f32,
  /// Number of windows whose warm-up-trimmed frames contributed to this
  /// frame's `count_sum`. Divisor for the rounded
  /// instantaneous-speaker-count.
  pub(crate) count_chunk_count: u32,
}

/// Per-cluster open-run state (spec §5.11). Lives in
/// [`ReconstructState::open_runs`] until emitted as a
/// [`DiarizedSpan`](crate::diarizer::DiarizedSpan) by
/// `emit_finalized_frames` or `flush_open_runs`.
#[allow(dead_code)] // fields consumed by emit_finalized_frames / flush_open_runs in Tasks 41-42
#[derive(Debug, Default)]
pub(crate) struct PerClusterRun {
  /// Absolute frame where the run started.
  pub(crate) start_frame: Option<u64>,
  /// Absolute frame of the most recent active assignment.
  pub(crate) last_active_frame: Option<u64>,
  /// Sum of per-frame normalized activations
  /// (`activation_sum / activation_chunk_count`) across the run's frames.
  /// `f64` accumulator for stability over long runs.
  pub(crate) activation_sum_normalized: f64,
  /// Number of frames that contributed to `activation_sum_normalized`.
  pub(crate) frame_count: u32,
  /// Set of `(WindowId, slot)` pairs that contributed to this run.
  /// Used for the `activity_count` and `clean_mask_fraction` quality
  /// metrics on the emitted `DiarizedSpan`.
  pub(crate) contributing_activities: HashSet<(WindowId, u8)>,
  /// Of the contributing activities, how many used the clean
  /// (overlap-excluded) mask. `clean_mask_fraction =
  /// clean_mask_count / contributing_activities.len()`.
  pub(crate) clean_mask_count: u32,
}

/// Reconstruction state machine. Owned by
/// [`crate::diarizer::Diarizer`]; accessed via `pub(crate)` paths.
#[derive(Debug, Default)]
pub(crate) struct ReconstructState {
  /// Absolute frame at index 0 of `activations` / `counts`.
  /// Increments only by `emit_finalized_frames` (rev-7 §5.11).
  pub(crate) base_frame: u64,
  /// Per-frame per-cluster activation accumulator (sparse: most frames
  /// have ≤ a few clusters active). VecDeque so finalized frames at the
  /// front can be popped in O(1).
  pub(crate) activations: VecDeque<HashMap<u64, f32>>,
  /// Per-frame counts (parallel to `activations`).
  pub(crate) counts: VecDeque<FrameCount>,
  /// `(window_id, slot) → cluster_id`. Populated by Phase 11's pump
  /// (Task 43) when activities are clustered. Read by `integrate_window`.
  /// `BTreeMap` (not `HashMap`) so iteration order is stable.
  pub(crate) slot_to_cluster: BTreeMap<(WindowId, u8), u64>,
  /// `(window_id, slot) → used_clean_mask_flag`. Populated alongside
  /// `slot_to_cluster` for the `clean_mask_fraction` metric.
  pub(crate) activity_clean_flags: HashMap<(WindowId, u8), bool>,
  /// `window_id → window_start (absolute samples)`. Populated by
  /// `integrate_window`; consumed by the pump (Task 43) when slicing
  /// audio for embedding extraction.
  pub(crate) window_starts: HashMap<WindowId, u64>,
  /// Per-cluster open-run state. Cluster id → run.
  pub(crate) open_runs: HashMap<u64, PerClusterRun>,
  /// Set of cluster ids ever emitted as a `DiarizedSpan` since
  /// `new()` / `clear()`. Drives `is_new_speaker` on the
  /// emitted span. Persists across `emit_finalized_frames` calls.
  pub(crate) emitted_speaker_ids: HashSet<u64>,
  /// Absolute frame below which everything is finalized. Monotonic;
  /// only `emit_finalized_frames` advances it.
  pub(crate) finalization_boundary: u64,
}

impl ReconstructState {
  /// Construct an empty reconstruction state.
  pub(crate) fn new() -> Self {
    Self::default()
  }

  /// Reset all bookkeeping. Called from
  /// [`Diarizer::clear`](crate::diarizer::Diarizer::clear).
  ///
  /// Drops any open per-cluster runs WITHOUT emitting them. If the
  /// caller wants the open runs flushed, they must call
  /// `Diarizer::finish_stream` BEFORE `Diarizer::clear`.
  pub(crate) fn clear(&mut self) {
    self.base_frame = 0;
    self.activations.clear();
    self.counts.clear();
    self.slot_to_cluster.clear();
    self.activity_clean_flags.clear();
    self.window_starts.clear();
    self.open_runs.clear();
    self.emitted_speaker_ids.clear();
    self.finalization_boundary = 0;
  }

  /// Number of frames currently buffered in the activation accumulator.
  /// Bounded by the segmenter's lookback window per spec §5.7.
  pub(crate) fn buffered_frame_count(&self) -> usize {
    self.activations.len()
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn frame_to_sample_u64_matches_segment_u32_over_full_range() {
    // Bit-exact equivalence over the entire u32 input range that
    // dia::segment uses (one window = 2356 frames over the typical
    // streaming session lookback).
    for frame_idx in 0u32..=2356 {
      let u32_result = crate::segment::stitch::frame_to_sample(frame_idx) as u64;
      let u64_result = frame_to_sample_u64(frame_idx as u64);
      assert_eq!(
        u32_result, u64_result,
        "frame_idx = {frame_idx} u32 result {u32_result} != u64 result {u64_result}"
      );
    }
  }

  #[test]
  fn frame_index_of_matches_segment_u64() {
    // Same formula, range covers ~4 windows.
    for sample_idx in 0u64..=(WINDOW_SAMPLES as u64 * 4) {
      let local = frame_index_of(sample_idx);
      let seg = crate::segment::stitch::frame_index_of(sample_idx);
      assert_eq!(local, seg, "sample_idx = {sample_idx}");
    }
  }

  #[test]
  fn warm_up_constants_match_pyannote_default() {
    // round(589 * 0.1) = 58.9 → 59.
    assert_eq!(SPEAKER_COUNT_WARM_UP_FRAMES_LEFT, 59);
    assert_eq!(SPEAKER_COUNT_WARM_UP_FRAMES_RIGHT, 59);
  }

  #[test]
  fn new_state_is_empty() {
    let s = ReconstructState::new();
    assert_eq!(s.base_frame, 0);
    assert_eq!(s.buffered_frame_count(), 0);
    assert!(s.open_runs.is_empty());
    assert!(s.emitted_speaker_ids.is_empty());
    assert_eq!(s.finalization_boundary, 0);
  }

  #[test]
  fn clear_resets_all_fields() {
    let mut s = ReconstructState::new();
    s.base_frame = 100;
    s.finalization_boundary = 50;
    s.activations.push_back(HashMap::from([(7u64, 0.9f32)]));
    s.counts.push_back(FrameCount {
      activation_chunk_count: 3,
      count_sum: 1.5,
      count_chunk_count: 2,
    });
    s.emitted_speaker_ids.insert(42);

    s.clear();
    assert_eq!(s.base_frame, 0);
    assert_eq!(s.finalization_boundary, 0);
    assert_eq!(s.buffered_frame_count(), 0);
    assert!(s.emitted_speaker_ids.is_empty());
  }
}
