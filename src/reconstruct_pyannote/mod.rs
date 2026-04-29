//! Pyannote reconstruction stage: hard_clusters + segmentations + count
//! → per-output-frame discrete diarization (binary `(frames, clusters)`
//! grid).
//!
//! Ports two pyannote functions:
//! - `pyannote.audio.pipelines.speaker_diarization.reconstruct` builds
//!   `clustered_segmentations` by maxing per-cluster speaker activity
//!   per frame.
//! - `pyannote.audio.pipelines.utils.diarization.to_diarization` runs
//!   `Inference.aggregate(skip_average=True)` overlap-add on the
//!   clustered segmentations, then top-`count[t]` binarizes per frame.
//!
//! ## Standalone — no `Diarizer` integration yet
//!
//! Phase 5b ships this as a pure-math module. Diarizer rewiring +
//! RTTM emission is Phase 5c.

#![allow(dead_code, unused_imports)]

mod algo;
mod error;

#[cfg(test)]
mod parity_tests;

pub use algo::{ReconstructInput, SlidingWindow, reconstruct};
pub use error::Error;
