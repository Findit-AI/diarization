//! Constrained Hungarian assignment — per-chunk speaker → cluster matching.
//!
//! Ports `pyannote.audio.pipelines.clustering.SpeakerEmbedding.constrained_argmax`
//! (`clustering.py:127-140` in pyannote.audio 4.0.4). Given a per-chunk
//! `(num_speakers, num_clusters)` cost matrix (typically
//! `2 - cosine_distance(embedding, centroid)`), returns the maximum-weight
//! bipartite matching as `Vec<i32>` of length `num_speakers`. Unmatched
//! speakers (possible when `num_speakers > num_clusters`) carry the sentinel
//! [`UNMATCHED`] (`-2`).
//!
//! ## Standalone — no `Diarizer` integration yet
//!
//! Phase 3 ships this as a pure-math module. Integration (`Diarizer`
//! consuming VBx + Hungarian → centroid AHC → per-frame diarization) lands
//! in Phase 5. Until then `diarization::cluster::hungarian` is crate-private (see
//! `src/lib.rs`).

#![allow(dead_code, unused_imports)]

mod algo;
mod error;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod parity_tests;

pub use algo::{
  ChunkAssignment, ChunkLayout, DefaultLayout, Segmentation3, UNMATCHED, constrained_argmax,
};
pub use error::Error;
