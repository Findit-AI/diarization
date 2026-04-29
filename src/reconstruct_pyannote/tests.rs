//! Model-free unit tests for `dia::reconstruct_pyannote`.

use crate::hungarian::UNMATCHED;
use crate::reconstruct_pyannote::{
  Error, MAX_CLUSTER_ID, ReconstructInput, SlidingWindow, discrete_to_spans, reconstruct,
};

fn default_swins() -> (SlidingWindow, SlidingWindow) {
  // Reasonable defaults: 1s chunk step over 5s chunks, ~17ms output frames.
  let chunks = SlidingWindow {
    start: 0.0,
    duration: 5.0,
    step: 1.0,
  };
  let frames = SlidingWindow {
    start: 0.0,
    duration: 0.062,
    step: 0.0169,
  };
  (chunks, frames)
}

/// NaN segmentation values are rejected at the boundary. Pyannote's
/// `Inference.aggregate` would replace NaN with 0 + mask, but a NaN
/// segmentation is realistically upstream model corruption. The Rust
/// port surfaces it as a clear typed error rather than silently
/// producing a degraded RTTM (Codex review MEDIUM round 2 of Phase 5).
#[test]
fn rejects_nan_segmentation() {
  let (chunks_sw, frames_sw) = default_swins();
  let num_chunks = 1;
  let num_frames_per_chunk = 4;
  let num_speakers = 2;
  let mut segmentations = vec![0.5_f64; num_chunks * num_frames_per_chunk * num_speakers];
  segmentations[3] = f64::NAN;
  let hard_clusters = vec![vec![0i32, 1i32]];
  let count = vec![1u8; 4];
  let input = ReconstructInput {
    segmentations: &segmentations,
    num_chunks,
    num_frames_per_chunk,
    num_speakers,
    hard_clusters: &hard_clusters,
    count: &count,
    num_output_frames: 4,
    chunks_sw,
    frames_sw,
  };
  assert!(matches!(reconstruct(&input), Err(Error::NonFinite(_))));
}

#[test]
fn rejects_pos_inf_segmentation() {
  let (chunks_sw, frames_sw) = default_swins();
  let mut segmentations = vec![0.5_f64; 8];
  segmentations[0] = f64::INFINITY;
  let hard_clusters = vec![vec![0i32, 1i32]];
  let input = ReconstructInput {
    segmentations: &segmentations,
    num_chunks: 1,
    num_frames_per_chunk: 4,
    num_speakers: 2,
    hard_clusters: &hard_clusters,
    count: &[1u8; 4],
    num_output_frames: 4,
    chunks_sw,
    frames_sw,
  };
  assert!(matches!(reconstruct(&input), Err(Error::NonFinite(_))));
}

/// Trailing active span at end-of-grid must close at
/// `timestamps[num_frames - 1]`, not `timestamps[num_frames]`.
/// Pyannote's `Binarize.__call__` uses `t = timestamps[-1]` for the
/// final region's end. Closing one step past would over-extend
/// end-of-file speakers by `frames_sw.step`. Codex review MEDIUM
/// round 3 of Phase 5.
#[test]
fn rttm_eof_active_span_closes_at_last_frame_center() {
  let frames_sw = SlidingWindow {
    start: 0.0,
    duration: 0.062,
    step: 0.0169,
  };
  // 4-frame grid, single cluster, all active. The active region runs
  // through the last frame, so `discrete_to_spans` must close at the
  // center of frame 3 (last index), not frame 4 (one past).
  let grid = vec![1.0_f32, 1.0, 1.0, 1.0];
  let spans = discrete_to_spans(&grid, 4, 1, frames_sw, 0.0);
  assert_eq!(spans.len(), 1);
  let span = &spans[0];
  let expected_start = 0.0 + 0.0 * 0.0169 + 0.062 / 2.0; // timestamps[0]
  let expected_end = 0.0 + 3.0 * 0.0169 + 0.062 / 2.0; // timestamps[3]
  assert!(
    (span.start - expected_start).abs() < 1e-12,
    "start: got {}, want {expected_start}",
    span.start
  );
  assert!(
    (span.start + span.duration - expected_end).abs() < 1e-12,
    "end: got {}, want {expected_end}",
    span.start + span.duration
  );
  // duration = (num_frames - 1 - 0) * step = 3 * 0.0169.
  assert!(
    (span.duration - 3.0 * 0.0169).abs() < 1e-12,
    "duration: got {}, want {:.6}",
    span.duration,
    3.0 * 0.0169
  );
}

/// A single final-frame-only active region (just frame `num_frames-1`
/// is active) must NOT emit a non-empty RTTM span — pyannote's
/// `Binarize` only emits a span when `t > start` after closure;
/// our fix returns no span when `end == start`.
#[test]
fn rttm_eof_single_final_frame_active_emits_no_span() {
  let frames_sw = SlidingWindow {
    start: 0.0,
    duration: 0.062,
    step: 0.0169,
  };
  // 4-frame grid, only the LAST frame active.
  // active_start = Some(3) at end of loop; close at timestamps[3].
  // start = end → no span.
  let grid = vec![0.0_f32, 0.0, 0.0, 1.0];
  let spans = discrete_to_spans(&grid, 4, 1, frames_sw, 0.0);
  assert!(
    spans.is_empty(),
    "single-frame EOF should emit no span: {spans:?}"
  );
}

/// Negative ids other than `UNMATCHED` are rejected at the boundary.
/// Without this guard, `-1` would silently drop the speaker from any
/// cluster mapping (the speakers_in_k filter never matches negative
/// `k_iter`). Codex review MEDIUM round 4 of Phase 5.
#[test]
fn rejects_negative_cluster_id_other_than_unmatched() {
  let (chunks_sw, frames_sw) = default_swins();
  // hard_clusters with a -1 entry (NOT the UNMATCHED -2 sentinel).
  let hard_clusters = vec![vec![0i32, -1i32]];
  let segmentations = vec![0.5_f64; 8];
  let input = ReconstructInput {
    segmentations: &segmentations,
    num_chunks: 1,
    num_frames_per_chunk: 4,
    num_speakers: 2,
    hard_clusters: &hard_clusters,
    count: &[1u8; 4],
    num_output_frames: 4,
    chunks_sw,
    frames_sw,
  };
  assert!(matches!(reconstruct(&input), Err(Error::Shape(_))));
}

/// `UNMATCHED` (`-2`) is the only allowed negative id; this test
/// pins that contract.
#[test]
fn accepts_unmatched_sentinel() {
  let (chunks_sw, frames_sw) = default_swins();
  let hard_clusters = vec![vec![0i32, UNMATCHED]];
  let segmentations = vec![0.5_f64; 8];
  let input = ReconstructInput {
    segmentations: &segmentations,
    num_chunks: 1,
    num_frames_per_chunk: 4,
    num_speakers: 2,
    hard_clusters: &hard_clusters,
    count: &[1u8; 4],
    num_output_frames: 4,
    chunks_sw,
    frames_sw,
  };
  assert!(reconstruct(&input).is_ok());
}

/// Cluster ids beyond `MAX_CLUSTER_ID` are rejected before allocation.
/// Without this guard, a caller passing `k = i32::MAX` would force
/// `num_clusters ≈ 2.1e9`, multiplying with `num_chunks *
/// num_frames_per_chunk` into a multi-petabyte allocation request.
#[test]
fn rejects_cluster_id_above_max() {
  let (chunks_sw, frames_sw) = default_swins();
  let hard_clusters = vec![vec![0i32, MAX_CLUSTER_ID + 1]];
  let segmentations = vec![0.5_f64; 8];
  let input = ReconstructInput {
    segmentations: &segmentations,
    num_chunks: 1,
    num_frames_per_chunk: 4,
    num_speakers: 2,
    hard_clusters: &hard_clusters,
    count: &[1u8; 4],
    num_output_frames: 4,
    chunks_sw,
    frames_sw,
  };
  assert!(matches!(reconstruct(&input), Err(Error::Shape(_))));
}

#[test]
fn rejects_neg_inf_segmentation() {
  let (chunks_sw, frames_sw) = default_swins();
  let mut segmentations = vec![0.5_f64; 8];
  segmentations[5] = f64::NEG_INFINITY;
  let hard_clusters = vec![vec![0i32, 1i32]];
  let input = ReconstructInput {
    segmentations: &segmentations,
    num_chunks: 1,
    num_frames_per_chunk: 4,
    num_speakers: 2,
    hard_clusters: &hard_clusters,
    count: &[1u8; 4],
    num_output_frames: 4,
    chunks_sw,
    frames_sw,
  };
  assert!(matches!(reconstruct(&input), Err(Error::NonFinite(_))));
}
