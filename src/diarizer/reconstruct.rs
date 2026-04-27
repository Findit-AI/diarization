//! Per-frame stitching state machine for [`crate::diarizer::Diarizer`].
//! Spec §5.9-§5.11.
//!
//! Tracks per-frame per-cluster activation overlap-add (rev-3 sum, NOT
//! mean — divisor lives in [`FrameCount::activation_chunk_count`]),
//! per-frame instantaneous-speaker-count (mean-with-warm-up-trim per
//! pyannote `speaker_count(warm_up=(0.1, 0.1))`), and per-cluster
//! open-run state for RLE-to-[`crate::diarizer::DiarizedSpan`]
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
/// [`crate::diarizer::DiarizedSpan`] by
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
        let score = frame_scores[f_in_window];
        // Sparse storage: don't materialize 0.0 entries. Prevents false-
        // cluster assignment when count_at_frame elsewhere would pick a
        // zero-scored cluster as the only candidate (Codex review post-
        // rev-9 HIGH).
        if score == 0.0 {
          continue;
        }
        *self.activations[buf_idx].entry(*cluster_id).or_insert(0.0) += score;
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
      // Cap n_active by mapped slots: unmapped active slots are activity
      // we couldn't embed — pretending their activity counts toward the
      // speaker count would let an unmapped slot's speech steal the
      // top-K active_set choice for an inactive mapped cluster (Codex
      // review post-rev-9 HIGH).
      let n_active = (0..MAX_SPEAKER_SLOTS as usize)
        .filter(|s| {
          raw_probs[*s][f_in_window] > binarize_threshold
            && self.slot_to_cluster.contains_key(&(window_id, *s as u8))
        })
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
    // there. Sparse storage drops zero-valued entries (Codex review
    // post-rev-9 fix) — the frame state is empty, NOT cluster 7 with
    // value 0.0.
    assert!(
      s.activations[50].is_empty(),
      "frame 50 must have no cluster entries (slot 0 inactive there); got {:?}",
      s.activations[50]
    );

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

  #[test]
  fn unmapped_active_slot_does_not_steal_mapped_cluster_activation() {
    // Codex review HIGH-severity regression: window W has slot 0 mapped
    // (active at early frames) and slot 1 unmapped (active at later
    // frames). Without the fix, count_at_frame at slot-1's active region
    // would be 1, frame_state would only contain cluster 7 with 0.0,
    // and emit_finalized_frames would falsely open cluster 7's span over
    // slot 1's speech.
    use crate::segment::options::SAMPLE_RATE_TB;
    use mediatime::TimeRange;

    let mut s = ReconstructState::new();
    let id = WindowId::new(TimeRange::new(0, WINDOW_SAMPLES as i64, SAMPLE_RATE_TB), 0);
    let mut probs = [[0.0f32; FRAMES_PER_WINDOW]; MAX_SPEAKER_SLOTS as usize];
    // Slot 0 active at frames 100..200 (mapped to cluster 7).
    for p in probs[0].iter_mut().skip(100).take(100) {
      *p = 0.9;
    }
    // Slot 1 active at frames 400..500 (UNMAPPED — embedding was skipped).
    for p in probs[1].iter_mut().skip(400).take(100) {
      *p = 0.9;
    }
    s.slot_to_cluster.insert((id, 0), 7);
    // NOTE: no slot_to_cluster entry for (id, 1).

    s.integrate_window(id, 0, &probs, 0.5);

    // Slot 0's mapped frames: cluster 7 has its max-collapsed score.
    assert!(s.activations[150].contains_key(&7));
    assert!((s.activations[150][&7] - 0.9).abs() < 1e-6);

    // Slot 1's unmapped frames: cluster 7 must NOT be present (slot 0
    // contributed nothing here, so per_cluster_max[7][450] = 0.0; sparse
    // storage drops the entry).
    assert!(
      !s.activations[450].contains_key(&7),
      "false cluster 7 entry at frame 450 (slot 1's unmapped speech): {:?}",
      s.activations[450]
    );

    // Speaker count at frame 450: slot 1 is unmapped, so it must NOT
    // contribute to count_sum. n_active should be 0 → count_chunk_count
    // increments but count_sum stays 0.
    let warm_left = SPEAKER_COUNT_WARM_UP_FRAMES_LEFT as usize;
    let warm_right = SPEAKER_COUNT_WARM_UP_FRAMES_RIGHT as usize;
    assert!(
      (warm_left..(FRAMES_PER_WINDOW - warm_right)).contains(&450),
      "test setup: frame 450 must be in the count region"
    );
    assert_eq!(
      s.counts[450].count_chunk_count, 1,
      "frame 450 is in the count region; chunk_count should be 1"
    );
    assert_eq!(
      s.counts[450].count_sum, 0.0,
      "unmapped slot must NOT contribute to count_sum; got {}",
      s.counts[450].count_sum
    );

    // Smoke: drive emit_finalized_frames and verify NO span is emitted
    // for cluster 7 covering slot 1's range.
    s.advance_finalization_boundary(u64::MAX);
    let mut emitted = Vec::new();
    s.emit_finalized_frames(15, |span| emitted.push(span));
    for span in &emitted {
      if span.speaker_id() != 7 {
        continue;
      }
      let s0 = span.range().start_pts() as u64;
      let s1 = span.range().end_pts() as u64;
      let frame_lo = frame_index_of(s0);
      let frame_hi = frame_index_of(s1);
      // Span must NOT extend into the unmapped slot 1's range (400..500).
      assert!(
        frame_hi <= 400,
        "cluster 7 span [{frame_lo}, {frame_hi}) extends into unmapped slot 1's frames (400..500): {span:?}"
      );
    }
  }
}

use mediatime::TimeRange;

use crate::{diarizer::span::DiarizedSpan, segment::options::SAMPLE_RATE_TB};

impl ReconstructState {
  /// Advance the finalization boundary based on the segment's
  /// [`Segmenter::peek_next_window_start`](crate::segment::Segmenter::peek_next_window_start).
  /// Frames before the boundary are finalized — no future window can
  /// contribute to them (the segment guarantees a non-decreasing
  /// `next_window_start` per spec §5.7 / §11.5 trim discipline).
  ///
  /// Monotonic: never moves backward, even if the caller passes a smaller
  /// value (defensive — should not happen on well-behaved input).
  ///
  /// `next_window_start == u64::MAX` (post-`finish_stream`) finalizes
  /// everything.
  #[allow(dead_code)] // wired into pump in Task 43
  pub(crate) fn advance_finalization_boundary(&mut self, next_window_start: u64) {
    let next_frame = if next_window_start == u64::MAX {
      u64::MAX
    } else {
      frame_index_of(next_window_start)
    };
    if next_frame > self.finalization_boundary {
      self.finalization_boundary = next_frame;
    }
  }

  /// Emit DiarizedSpans for all finalized frames. Drains `activations`
  /// and `counts` up to `finalization_boundary`. Spec §5.10 + §5.11.
  ///
  /// Algorithm per finalized frame:
  /// 1. `count_at_frame` = `round(count_sum / count_chunk_count)` capped
  ///    at `max_speakers`. Zero if `count_chunk_count == 0` (warm-up
  ///    trimmed everywhere — no count signal).
  /// 2. `active_set` = top-`count_at_frame` clusters by activation, with
  ///    smaller-cluster-id tie-break (rev-3 deterministic; pyannote uses
  ///    `np.argsort` which is also stable, but explicit is better).
  /// 3. Close runs for any open cluster not in `active_set`; emit span.
  /// 4. Open or extend runs for clusters in `active_set`.
  ///
  /// `max_speakers` caps the per-frame speaker count. Pass `u32::MAX`
  /// for no cap.
  #[allow(dead_code)] // wired into pump in Task 43
  pub(crate) fn emit_finalized_frames<F: FnMut(DiarizedSpan)>(
    &mut self,
    max_speakers: u32,
    mut emit: F,
  ) {
    while self.base_frame < self.finalization_boundary {
      let Some(frame_state) = self.activations.pop_front() else {
        break;
      };
      let frame_count_state = self
        .counts
        .pop_front()
        .expect("counts and activations grow in lockstep");

      // ── Step 1: count_at_frame ──
      let count_at_frame = if frame_count_state.count_chunk_count == 0 {
        0u32
      } else {
        let mean = frame_count_state.count_sum / frame_count_state.count_chunk_count as f32;
        (mean.round() as u32).min(max_speakers)
      };

      // ── Step 2: active_set (top-K by activation, deterministic tie-break) ──
      let active_set: HashSet<u64> = if count_at_frame > 0 {
        let mut sorted: Vec<(u64, f32)> = frame_state.iter().map(|(&c, &a)| (c, a)).collect();
        sorted.sort_by(|(a_id, a_v), (b_id, b_v)| b_v.total_cmp(a_v).then(a_id.cmp(b_id)));
        sorted
          .into_iter()
          .take(count_at_frame as usize)
          .map(|(c, _)| c)
          .collect()
      } else {
        HashSet::new()
      };

      // ── Step 3: close runs not in active_set ──
      // Collect IDs first (avoid mutable-while-iterating).
      let to_close: Vec<u64> = self
        .open_runs
        .iter()
        .filter(|(c, run)| run.start_frame.is_some() && !active_set.contains(*c))
        .map(|(c, _)| *c)
        .collect();
      for cluster_id in to_close {
        let run = self
          .open_runs
          .get_mut(&cluster_id)
          .expect("cluster is in open_runs by construction");
        let start = run.start_frame.take().expect("run was open");
        let end = run
          .last_active_frame
          .expect("open run must have last_active_frame")
          + 1;
        let s0 = frame_to_sample_u64(start) as i64;
        let s1 = frame_to_sample_u64(end) as i64;
        let range = TimeRange::new(s0, s1, SAMPLE_RATE_TB);

        let is_new_speaker = !self.emitted_speaker_ids.contains(&cluster_id);
        if is_new_speaker {
          self.emitted_speaker_ids.insert(cluster_id);
        }
        let activity_count = run.contributing_activities.len() as u32;
        let avg_activation = (run.activation_sum_normalized / run.frame_count.max(1) as f64) as f32;
        let clean_fraction = if activity_count == 0 {
          0.0
        } else {
          run.clean_mask_count as f32 / activity_count as f32
        };

        emit(DiarizedSpan {
          range,
          speaker_id: cluster_id,
          is_new_speaker,
          average_activation: avg_activation,
          activity_count,
          clean_mask_fraction: clean_fraction,
        });

        // Reset for next run on this cluster.
        run.last_active_frame = None;
        run.activation_sum_normalized = 0.0;
        run.frame_count = 0;
        run.contributing_activities.clear();
        run.clean_mask_count = 0;
      }

      // ── Step 4: open / extend runs for active clusters ──
      let activation_chunk_count = frame_count_state.activation_chunk_count.max(1) as f32;
      for cluster_id in &active_set {
        let activation_at_frame = frame_state[cluster_id];
        let activation_normalized = (activation_at_frame / activation_chunk_count) as f64;
        let run = self.open_runs.entry(*cluster_id).or_default();
        if run.start_frame.is_none() {
          run.start_frame = Some(self.base_frame);
        }
        run.last_active_frame = Some(self.base_frame);
        run.activation_sum_normalized += activation_normalized;
        run.frame_count += 1;

        // Bookkeeping for activity_count and clean_mask_count: scan
        // slot_to_cluster for entries mapping to this cluster whose
        // window covers `self.base_frame`.
        for ((window_id, slot), bound_cluster) in self.slot_to_cluster.iter() {
          if *bound_cluster != *cluster_id {
            continue;
          }
          let Some(&win_start) = self.window_starts.get(window_id) else {
            continue; // window not yet integrated (defensive)
          };
          let frame_lo = frame_index_of(win_start);
          let frame_hi = frame_lo + FRAMES_PER_WINDOW as u64;
          if self.base_frame >= frame_lo
            && self.base_frame < frame_hi
            && run.contributing_activities.insert((*window_id, *slot))
            && let Some(&true) = self.activity_clean_flags.get(&(*window_id, *slot))
          {
            run.clean_mask_count += 1;
          }
        }
      }

      self.base_frame += 1;
    }
  }
}

#[cfg(test)]
mod emit_tests {
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
  fn single_window_single_speaker_emits_one_span() {
    let mut s = ReconstructState::new();
    let id = make_window_id(0, 0);
    let mut probs = [[0.0f32; FRAMES_PER_WINDOW]; MAX_SPEAKER_SLOTS as usize];
    // Slot 0 high probability for frames 100..200.
    for p in probs[0].iter_mut().skip(100).take(100) {
      *p = 0.9;
    }
    s.slot_to_cluster.insert((id, 0), 7);
    s.activity_clean_flags.insert((id, 0), true);
    s.integrate_window(id, 0, &probs, 0.5);

    // Force-finalize all frames.
    s.advance_finalization_boundary(u64::MAX);
    let mut emitted = Vec::new();
    s.emit_finalized_frames(15, |span| emitted.push(span));

    let spans_for_7: Vec<&DiarizedSpan> = emitted.iter().filter(|s| s.speaker_id() == 7).collect();
    assert_eq!(
      spans_for_7.len(),
      1,
      "expected 1 span for cluster 7; got emissions {emitted:?}"
    );
    let span = spans_for_7[0];
    assert_eq!(span.activity_count(), 1, "1 activity contributed");
    assert!(span.is_new_speaker());
    assert!(
      (span.clean_mask_fraction() - 1.0).abs() < 1e-7,
      "1/1 = 1.0 clean fraction"
    );
    assert!(
      span.average_activation() > 0.5,
      "expected avg > 0.5; got {}",
      span.average_activation()
    );
  }

  #[test]
  fn count_zero_in_warm_up_emits_nothing() {
    // Slot 0 active ONLY in warm-up region (frames 0..50). count_chunk_count
    // = 0 there → count_at_frame = 0 → no cluster ever in active_set.
    let mut s = ReconstructState::new();
    let id = make_window_id(0, 0);
    let mut probs = [[0.0f32; FRAMES_PER_WINDOW]; MAX_SPEAKER_SLOTS as usize];
    for p in probs[0].iter_mut().take(50) {
      *p = 0.9;
    }
    s.slot_to_cluster.insert((id, 0), 7);
    s.integrate_window(id, 0, &probs, 0.5);

    s.advance_finalization_boundary(u64::MAX);
    let mut emitted = Vec::new();
    s.emit_finalized_frames(15, |span| emitted.push(span));

    assert!(
      emitted.is_empty(),
      "expected 0 spans (warm-up only); got {emitted:?}"
    );
  }

  #[test]
  fn second_speaker_run_marked_not_new() {
    // Two windows, each opening + closing cluster 7. First emission has
    // is_new_speaker=true; second has is_new_speaker=false.
    let mut s = ReconstructState::new();

    let id_a = make_window_id(0, 0);
    let mut probs_a = [[0.0f32; FRAMES_PER_WINDOW]; MAX_SPEAKER_SLOTS as usize];
    for p in probs_a[0].iter_mut().skip(100).take(100) {
      *p = 0.9;
    }
    s.slot_to_cluster.insert((id_a, 0), 7);
    s.activity_clean_flags.insert((id_a, 0), false);
    s.integrate_window(id_a, 0, &probs_a, 0.5);

    // Finalize window A's frames.
    s.advance_finalization_boundary(WINDOW_SAMPLES as u64);
    let mut emitted = Vec::new();
    s.emit_finalized_frames(15, |span| emitted.push(span));
    assert!(
      emitted
        .iter()
        .any(|sp| sp.speaker_id() == 7 && sp.is_new_speaker()),
      "first emission must mark cluster 7 as new"
    );

    // Window B at offset = WINDOW_SAMPLES (no overlap).
    let id_b = make_window_id(WINDOW_SAMPLES as i64, 1);
    let mut probs_b = [[0.0f32; FRAMES_PER_WINDOW]; MAX_SPEAKER_SLOTS as usize];
    for p in probs_b[0].iter_mut().skip(100).take(100) {
      *p = 0.9;
    }
    s.slot_to_cluster.insert((id_b, 0), 7);
    s.activity_clean_flags.insert((id_b, 0), false);
    s.integrate_window(id_b, WINDOW_SAMPLES as u64, &probs_b, 0.5);

    s.advance_finalization_boundary(u64::MAX);
    let mut emitted2 = Vec::new();
    s.emit_finalized_frames(15, |span| emitted2.push(span));
    let span_b = emitted2
      .iter()
      .find(|sp| sp.speaker_id() == 7)
      .expect("second window should re-emit cluster 7");
    assert!(
      !span_b.is_new_speaker(),
      "second emission of cluster 7 must NOT be marked new; got {span_b:?}"
    );
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

impl ReconstructState {
  /// End-of-stream flush. Sets `finalization_boundary = u64::MAX`,
  /// drains [`emit_finalized_frames`](Self::emit_finalized_frames),
  /// then closes any per-cluster runs that are STILL open (clusters
  /// that were active up to the very last finalized frame and never
  /// got a "fell out of top" signal because there's no next frame).
  ///
  /// Called from [`Diarizer::finish_stream`](crate::diarizer::Diarizer::finish_stream).
  /// All emissions go through `emit`.
  #[allow(dead_code)] // wired into pump in Task 43
  pub(crate) fn flush_open_runs<F: FnMut(DiarizedSpan)>(&mut self, mut emit: F) {
    self.finalization_boundary = u64::MAX;
    self.emit_finalized_frames(u32::MAX, &mut emit);

    // After draining frames, close any still-open runs. These are runs
    // whose `last_active_frame` is the last frame ever observed; the
    // per-frame loop above didn't close them because there's no next
    // frame to provide a "this cluster fell out of top" signal.
    let to_emit: Vec<u64> = self
      .open_runs
      .iter()
      .filter(|(_, run)| run.start_frame.is_some())
      .map(|(c, _)| *c)
      .collect();

    for cluster_id in to_emit {
      let run = self
        .open_runs
        .get_mut(&cluster_id)
        .expect("cluster is in open_runs by construction");
      let start = run.start_frame.take().expect("run was open");
      let end = run
        .last_active_frame
        .expect("open run must have last_active_frame")
        + 1;
      let s0 = frame_to_sample_u64(start) as i64;
      let s1 = frame_to_sample_u64(end) as i64;
      let range = TimeRange::new(s0, s1, SAMPLE_RATE_TB);

      let is_new_speaker = !self.emitted_speaker_ids.contains(&cluster_id);
      if is_new_speaker {
        self.emitted_speaker_ids.insert(cluster_id);
      }
      let activity_count = run.contributing_activities.len() as u32;
      let avg_activation = (run.activation_sum_normalized / run.frame_count.max(1) as f64) as f32;
      let clean_fraction = if activity_count == 0 {
        0.0
      } else {
        run.clean_mask_count as f32 / activity_count as f32
      };

      emit(DiarizedSpan {
        range,
        speaker_id: cluster_id,
        is_new_speaker,
        average_activation: avg_activation,
        activity_count,
        clean_mask_fraction: clean_fraction,
      });

      // Reset for any future runs on this cluster (after a `clear()`
      // or further window integration).
      run.last_active_frame = None;
      run.activation_sum_normalized = 0.0;
      run.frame_count = 0;
      run.contributing_activities.clear();
      run.clean_mask_count = 0;
    }
  }

  /// Evict per-window-id metadata that's no longer referenced. Spec §11.13.
  ///
  /// A window's entries (`slot_to_cluster`, `activity_clean_flags`,
  /// `window_starts`) can be dropped once:
  /// (a) all of its frames have finalized
  ///     (`base_frame >= window_start_frame + FRAMES_PER_WINDOW`), AND
  /// (b) no currently-open per-cluster run still references any
  ///     `(window_id, slot)` pair from this window in
  ///     `contributing_activities`.
  ///
  /// Bounds memory on long sessions; correctness-relevant for runs
  /// spanning many finalized windows.
  #[allow(dead_code)] // wired into pump in Task 43
  pub(crate) fn evict_finalized_window_metadata(&mut self) {
    let evictable: Vec<WindowId> = self
      .window_starts
      .iter()
      .filter(|(window_id, start)| {
        let last_frame_excl = frame_index_of(**start) + FRAMES_PER_WINDOW as u64;
        if self.base_frame < last_frame_excl {
          return false;
        }
        // (b): no open run still references this window.
        self.open_runs.values().all(|run| {
          run
            .contributing_activities
            .iter()
            .all(|(wid, _)| wid != *window_id)
        })
      })
      .map(|(w, _)| *w)
      .collect();

    for w in evictable {
      self.slot_to_cluster.retain(|(wid, _), _| *wid != w);
      self.activity_clean_flags.retain(|(wid, _), _| *wid != w);
      self.window_starts.remove(&w);
    }
  }
}

#[cfg(test)]
mod flush_eviction_tests {
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
  fn flush_emits_open_run_at_end_of_stream() {
    // Slot 0 active until the last unwarmed frame of the window — the
    // emit_finalized_frames loop will close it because the count drops
    // to 0 in the right warm-up region. So this specific case actually
    // tests the SECOND code path (the still-open run remains because
    // it was active right up to the last frame).
    //
    // To exercise the still-open path: keep slot 0 active for the
    // ENTIRE non-warm-up region (frames 59..530). The loop reaches the
    // last frame (530) with count > 0 → cluster 7 still in active_set
    // → run not closed. flush_open_runs must close it.
    let mut s = ReconstructState::new();
    let id = make_window_id(0, 0);
    let mut probs = [[0.0f32; FRAMES_PER_WINDOW]; MAX_SPEAKER_SLOTS as usize];
    let warm_left = SPEAKER_COUNT_WARM_UP_FRAMES_LEFT as usize;
    let warm_right = SPEAKER_COUNT_WARM_UP_FRAMES_RIGHT as usize;
    for p in probs[0]
      .iter_mut()
      .take(FRAMES_PER_WINDOW - warm_right)
      .skip(warm_left)
    {
      *p = 0.9;
    }
    s.slot_to_cluster.insert((id, 0), 7);
    s.activity_clean_flags.insert((id, 0), true);
    s.integrate_window(id, 0, &probs, 0.5);

    let mut emitted = Vec::new();
    s.flush_open_runs(|span| emitted.push(span));

    assert!(
      emitted.iter().any(|sp| sp.speaker_id() == 7),
      "expected cluster 7 to be flushed; got {emitted:?}"
    );
  }

  #[test]
  fn evict_drops_metadata_for_finalized_unused_windows() {
    let mut s = ReconstructState::new();
    let id_a = make_window_id(0, 0);
    let id_b = make_window_id(WINDOW_SAMPLES as i64, 1);

    s.window_starts.insert(id_a, 0);
    s.window_starts.insert(id_b, WINDOW_SAMPLES as u64);
    s.slot_to_cluster.insert((id_a, 0), 7);
    s.slot_to_cluster.insert((id_b, 0), 8);

    // Pretend we've finalized past id_a's last frame.
    s.base_frame = frame_index_of(0) + FRAMES_PER_WINDOW as u64 + 10;

    s.evict_finalized_window_metadata();

    assert!(
      !s.window_starts.contains_key(&id_a),
      "id_a's window_start should be evicted"
    );
    assert!(
      s.window_starts.contains_key(&id_b),
      "id_b is still active (last frame > base_frame)"
    );
    assert!(
      !s.slot_to_cluster.contains_key(&(id_a, 0)),
      "id_a's slot mapping should be evicted"
    );
    assert!(
      s.slot_to_cluster.contains_key(&(id_b, 0)),
      "id_b's slot mapping must be retained"
    );
  }

  #[test]
  fn evict_keeps_metadata_when_open_run_references_it() {
    // id_a is finalized but cluster 7's open run still references
    // (id_a, 0) → must be kept.
    let mut s = ReconstructState::new();
    let id_a = make_window_id(0, 0);
    s.window_starts.insert(id_a, 0);
    s.slot_to_cluster.insert((id_a, 0), 7);
    s.base_frame = frame_index_of(0) + FRAMES_PER_WINDOW as u64 + 10;

    let run = s.open_runs.entry(7).or_default();
    run.start_frame = Some(0);
    run.last_active_frame = Some(50);
    run.contributing_activities.insert((id_a, 0));

    s.evict_finalized_window_metadata();

    assert!(
      s.window_starts.contains_key(&id_a),
      "open run still references id_a → must keep window_start"
    );
    assert!(
      s.slot_to_cluster.contains_key(&(id_a, 0)),
      "must keep slot mapping while open_run references it"
    );
  }
}
