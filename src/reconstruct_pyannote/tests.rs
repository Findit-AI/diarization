//! Model-free unit tests for `dia::reconstruct_pyannote`.

use crate::reconstruct_pyannote::{Error, ReconstructInput, SlidingWindow, reconstruct};

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
