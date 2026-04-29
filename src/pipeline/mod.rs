//! Full pyannote-equivalent batch clustering pipeline.
//!
//! Ports the per-chunk diarization assignment in
//! `pyannote.audio.pipelines.clustering.SpeakerEmbedding.__call__`
//! (`clustering.py:570-625` in pyannote.audio 4.0.4):
//!
//! 1. Filter active embeddings (Phase 5a Task 4 — currently caller-supplied).
//! 2. AHC initialization on the active subset (`dia::ahc`).
//! 3. PLDA project (`dia::plda::PldaTransform::project` — currently caller-supplied).
//! 4. VBx EM iterations (`dia::vbx::vbx_iterate`).
//! 5. Drop sp-squashed clusters and compute weighted centroids (`dia::centroid`).
//! 6. Per-chunk per-speaker centroid distances (cdist with cosine metric).
//! 7. `constrained_argmax` over masked soft clusters (`dia::hungarian`).
//!
//! Output: per-chunk hard-cluster assignments `Vec<Vec<i32>>`, where
//! each inner vector has length `num_speakers` and `UNMATCHED = -2`
//! marks speakers with no surviving cluster (only possible when
//! `num_speakers > num_alive_clusters`).
//!
//! Stage 8 (per-frame discrete diarization) is Phase 5b. Diarizer
//! integration is Phase 5c. Until then `dia::pipeline` is crate-private.

#![allow(dead_code, unused_imports)]

mod algo;
mod error;

#[cfg(test)]
mod parity_tests;

pub use algo::{AssignEmbeddingsInput, assign_embeddings};
pub use error::Error;
