//! PLDA (Probabilistic Linear Discriminant Analysis) transform.
//!
//! Ports `pyannote.audio.utils.vbx.vbx_setup` plus its inner `xvec_tf`
//! / `plda_tf` lambdas (`utils/vbx.py:181-218` in pyannote.audio
//! 4.0.4) to Rust. Loads two `.npz` weight files (shipped with
//! `pyannote/speaker-diarization-community-1`, redistributed under
//! `models/plda/`) and exposes a deterministic two-stage projection:
//!
//! ```text
//! 256-d WeSpeaker embedding (f32)
//!         │
//!         ▼  xvec_transform
//! 128-d PLDA stage 1 (f64, sqrt(128)-scaled L2-norm; ‖·‖ ≈ 11.31)
//!         │
//!         ▼  plda_transform
//! 128-d PLDA stage 2 (f64, whitened — input to VBx in Phase 2)
//! ```
//!
//! ## Pinning
//!
//! The implementation tracks pyannote.audio 4.0.4 byte-for-byte via the
//! parity tests in `tests/parity_plda.rs`, which validate against the
//! Phase-0 captured artifacts under
//! `tests/parity/fixtures/01_dialogue/plda_embeddings.npz`. Bumping
//! pyannote requires re-running the Phase-0 capture and re-validating
//! these tests.
//!
//! ## Standalone — no `Diarizer` integration yet
//!
//! Phase 1 ships PLDA as a pure data-transformation module. The actual
//! wiring into the streaming/offline clusterer happens after the
//! remaining phases (VBx, constrained Hungarian, centroid AHC) land.

mod error;
mod loader;
mod transform;

#[cfg(test)]
mod tests;

pub use error::Error;
pub use transform::{PldaTransform, PostXvecEmbedding, RawEmbedding};

/// PLDA stage-1 / stage-2 dimension. Pyannote's
/// `pyannote/speaker-diarization-community-1` always uses 128.
pub const PLDA_DIMENSION: usize = 128;

/// WeSpeaker embedding dimension (input to `xvec_transform`).
pub const EMBEDDING_DIMENSION: usize = 256;
