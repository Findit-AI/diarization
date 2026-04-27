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

use crate::segment::options::MAX_SPEAKER_SLOTS;

impl ReconstructState {
  /// Integrate one processed window's raw probabilities into the
  /// per-frame accumulators. Spec §5.9 / §5.10.
  ///
  /// **Pre-conditions** (caller / Phase 11 pump — Task 43):
  /// - For every active `(window_id, slot)` in this window,
  ///   `slot_to_cluster.insert((window_id, slot), cluster_id)` AND
  ///   `activity_clean_flags.insert((window_id, slot), used_clean)`
  ///   have been called.
  /// - Slots NOT present in `slot_to_cluster` are "inactive" — silently
  ///   skipped (matches pyannote's `inactive_speakers → -2` throwaway).
  ///
  /// **Post-conditions:**
  /// - `window_starts[window_id] = window_start`.
  /// - `activations` and `counts` extended to cover this window's frames.
  /// - For each frame: per-cluster max-collapsed activation summed into
  ///   `activations[buf_idx]`; `activation_chunk_count` incremented.
  /// - For frames outside the warm-up margin: `count_sum` += (n active
  ///   slots > threshold); `count_chunk_count` += 1.
  ///
  /// **Frames before `base_frame` are skipped silently** (defensive;
  /// the segment contract guarantees windows arrive in non-decreasing
  /// `window_start_frame` order, so this should be unreachable on
  /// well-behaved input).
  #[allow(dead_code)] // wired into pump in Task 43
  pub(crate) fn integrate_window(
    &mut self,
    window_id: WindowId,
    window_start: u64,
    raw_probs: &[[f32; FRAMES_PER_WINDOW]; MAX_SPEAKER_SLOTS as usize],
    binarize_threshold: f32,
  ) {
    self.window_starts.insert(window_id, window_start);

    let window_start_frame = frame_index_of(window_start);
    let last_frame_excl = window_start_frame + FRAMES_PER_WINDOW as u64;

    // Grow activations/counts so that index `last_frame_excl - base_frame`
    // is in-range. If `window_start_frame < base_frame` (defensive; should
    // not happen per the segment contract), only the in-range tail of this
    // window's frames will contribute.
    if last_frame_excl > self.base_frame {
      let needed_len = (last_frame_excl - self.base_frame) as usize;
      while self.activations.len() < needed_len {
        self.activations.push_back(HashMap::new());
        self.counts.push_back(FrameCount::default());
      }
    }

    // ── Step A: collapse-by-max within each cluster for THIS window. ──
    //
    // per_cluster_max[c][f] = max over slots-mapped-to-c of raw_probs[slot][f].
    // Skips slots without a (window_id, slot) → cluster mapping (inactive).
    let mut per_cluster_max: HashMap<u64, [f32; FRAMES_PER_WINDOW]> = HashMap::new();
    for slot in 0..MAX_SPEAKER_SLOTS {
      let Some(&cluster_id) = self.slot_to_cluster.get(&(window_id, slot)) else {
        continue;
      };
      let entry = per_cluster_max
        .entry(cluster_id)
        .or_insert([0.0f32; FRAMES_PER_WINDOW]);
      for (e, p) in entry.iter_mut().zip(raw_probs[slot as usize].iter()) {
        *e = e.max(*p);
      }
    }

    // ── Step B: overlap-add SUM into the per-frame accumulators. ──
    //
    // Pyannote `Inference.aggregate(skip_average=True)` — sum across windows,
    // not mean. Activation_chunk_count tracks the divisor for output
    // (rev-3 / rev-8 T2-A: separate from count_chunk_count).
    for f_in_window in 0..FRAMES_PER_WINDOW {
      let abs_frame = window_start_frame + f_in_window as u64;
      if abs_frame < self.base_frame {
        continue;
      }
      let buf_idx = (abs_frame - self.base_frame) as usize;
      for (cluster_id, frame_scores) in per_cluster_max.iter() {
        *self.activations[buf_idx].entry(*cluster_id).or_insert(0.0) += frame_scores[f_in_window];
      }
      self.counts[buf_idx].activation_chunk_count += 1;
    }

    // ── Step C: per-frame instantaneous-speaker-count (warm-up trimmed). ──
    //
    // Only frames in [WARM_UP_LEFT, FRAMES_PER_WINDOW - WARM_UP_RIGHT)
    // contribute (pyannote `speaker_count(warm_up=(0.1, 0.1))`).
    // count_sum / count_chunk_count → mean → round → cap by max_speakers
    // (capping happens in emit_finalized_frames / Task 41).
    let warm_left = SPEAKER_COUNT_WARM_UP_FRAMES_LEFT as usize;
    let warm_right = SPEAKER_COUNT_WARM_UP_FRAMES_RIGHT as usize;
    // Drive the warm-up-trimmed frame loop with an enumerated iterator over
    // slot 0's per-frame slice (any slot would do — we only use it as the
    // length-typed iterator handle so clippy doesn't flag a `needless_range_loop`
    // on a 2-axis matrix access). The 2-axis lookup `raw_probs[s][f_in_window]`
    // is fine inside the outer iterator-form loop.
    for (f_in_window, _) in raw_probs[0]
      .iter()
      .enumerate()
      .take(FRAMES_PER_WINDOW - warm_right)
      .skip(warm_left)
    {
      let abs_frame = window_start_frame + f_in_window as u64;
      if abs_frame < self.base_frame {
        continue;
      }
      let buf_idx = (abs_frame - self.base_frame) as usize;
      let n_active = (0..MAX_SPEAKER_SLOTS as usize)
        .filter(|s| raw_probs[*s][f_in_window] > binarize_threshold)
        .count() as f32;
      self.counts[buf_idx].count_sum += n_active;
      self.counts[buf_idx].count_chunk_count += 1;
    }
  }
}

#[cfg(test)]
mod integrate_tests {
  use super::*;
  use crate::segment::options::SAMPLE_RATE_TB;
  use mediatime::TimeRange;

  fn make_window_id(start: i64, generation: u64) -> WindowId {
    WindowId::new(
      TimeRange::new(start, start + WINDOW_SAMPLES as i64, SAMPLE_RATE_TB),
      generation,
    )
  }

  #[test]
  fn integrate_window_grows_buffers_and_records_window_start() {
    let mut s = ReconstructState::new();
    let id = make_window_id(0, 0);
    let mut probs = [[0.0f32; FRAMES_PER_WINDOW]; MAX_SPEAKER_SLOTS as usize];
    // Slot 0 active for frames 100..200 at high probability.
    for p in probs[0].iter_mut().take(200).skip(100) {
      *p = 0.9;
    }
    s.slot_to_cluster.insert((id, 0), 7);

    s.integrate_window(id, 0, &probs, 0.5);

    // Buffers grew to FRAMES_PER_WINDOW = 589 frames.
    assert_eq!(s.activations.len(), FRAMES_PER_WINDOW);
    assert_eq!(s.counts.len(), FRAMES_PER_WINDOW);
    // window_start recorded.
    assert_eq!(s.window_starts.get(&id), Some(&0u64));

    // Frame 100: cluster 7 has activation 0.9 (single window so far).
    assert!(s.activations[100].contains_key(&7));
    assert!((s.activations[100][&7] - 0.9).abs() < 1e-6);
    // Frame 50: slot 0 is mapped to cluster 7 but its probability is 0.0
    // there, so the entry exists with value 0.0 (the algorithm
    // unconditionally accumulates per-frame for any cluster bound to the
    // window — the zero-valued entry is benign and §5.11 normalization
    // handles it via activation_chunk_count).
    assert!(s.activations[50].contains_key(&7));
    assert!(s.activations[50][&7].abs() < 1e-6);

    // Activation chunk count = 1 for EVERY frame (this single window).
    for fc in &s.counts {
      assert_eq!(fc.activation_chunk_count, 1);
    }

    // count_chunk_count = 1 outside warm-up, 0 inside.
    let warm_left = SPEAKER_COUNT_WARM_UP_FRAMES_LEFT as usize;
    let warm_right = SPEAKER_COUNT_WARM_UP_FRAMES_RIGHT as usize;
    for (f, fc) in s.counts.iter().enumerate() {
      let in_warm = f < warm_left || f >= (FRAMES_PER_WINDOW - warm_right);
      if in_warm {
        assert_eq!(
          fc.count_chunk_count, 0,
          "frame {f} should be warm-up trimmed"
        );
      } else {
        assert_eq!(
          fc.count_chunk_count, 1,
          "frame {f} should contribute to count"
        );
      }
    }
  }

  #[test]
  fn collapse_by_max_when_two_slots_same_cluster() {
    // Two slots mapped to the same cluster → max wins (not sum).
    let mut s = ReconstructState::new();
    let id = make_window_id(0, 0);
    let mut probs = [[0.0f32; FRAMES_PER_WINDOW]; MAX_SPEAKER_SLOTS as usize];
    probs[0][100] = 0.6;
    probs[1][100] = 0.9;
    s.slot_to_cluster.insert((id, 0), 7);
    s.slot_to_cluster.insert((id, 1), 7);

    s.integrate_window(id, 0, &probs, 0.5);

    // Activation should be max(0.6, 0.9) = 0.9 for cluster 7 (NOT 1.5).
    assert!(
      (s.activations[100][&7] - 0.9).abs() < 1e-6,
      "expected max-collapse to 0.9, got {}",
      s.activations[100][&7]
    );
  }

  #[test]
  fn overlap_add_sums_two_windows() {
    // Two windows that overlap by half. Same slot, same cluster, prob 0.5.
    // Overlap region should sum to 1.0 (overlap-add SUM semantics).
    let step = (WINDOW_SAMPLES as u64) / 2; // 80 000
    let id_a = make_window_id(0, 0);
    let id_b = make_window_id(step as i64, 1);
    let mut s = ReconstructState::new();
    s.slot_to_cluster.insert((id_a, 0), 7);
    s.slot_to_cluster.insert((id_b, 0), 7);
    let mut probs = [[0.0f32; FRAMES_PER_WINDOW]; MAX_SPEAKER_SLOTS as usize];
    for p in probs[0].iter_mut() {
      *p = 0.5;
    }
    s.integrate_window(id_a, 0, &probs, 0.5);
    s.integrate_window(id_b, step, &probs, 0.5);

    // Frame at sample `step` is covered by both windows. Activation =
    // 0.5 + 0.5 = 1.0; activation_chunk_count = 2.
    let overlap_frame = frame_index_of(step) as usize;
    assert!(
      (s.activations[overlap_frame][&7] - 1.0).abs() < 1e-5,
      "expected sum 1.0, got {}",
      s.activations[overlap_frame][&7]
    );
    assert_eq!(s.counts[overlap_frame].activation_chunk_count, 2);
  }

  #[test]
  fn inactive_slots_are_skipped() {
    // Slot 0 has high prob but no slot_to_cluster entry → no contribution.
    let mut s = ReconstructState::new();
    let id = make_window_id(0, 0);
    let mut probs = [[0.0f32; FRAMES_PER_WINDOW]; MAX_SPEAKER_SLOTS as usize];
    for p in probs[0].iter_mut().take(200).skip(100) {
      *p = 0.9;
    }
    // NOTE: slot_to_cluster is intentionally empty.

    s.integrate_window(id, 0, &probs, 0.5);

    // Activations completely empty (no clusters bound).
    for frame_state in &s.activations {
      assert!(frame_state.is_empty());
    }
    // But activation_chunk_count is STILL bumped (the window contributed
    // to the per-frame "there was a window here" count, just no cluster
    // mass to add).
    for fc in &s.counts {
      assert_eq!(fc.activation_chunk_count, 1);
    }
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
