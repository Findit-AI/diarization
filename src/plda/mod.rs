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
//! parity tests in `src/plda/parity_tests.rs` (a `#[cfg(test)]` module),
//! which validate against the Phase-0 captured artifacts under
//! `tests/parity/fixtures/01_dialogue/plda_embeddings.npz`. Bumping
//! pyannote requires re-running the Phase-0 capture and re-validating
//! these tests. Run with `cargo test plda::parity_tests`.
//!
//! ## Standalone — no `Diarizer` integration yet
//!
//! Phase 1 ships PLDA as a pure data-transformation module. The actual
//! wiring into the streaming/offline clusterer happens after the
//! remaining phases (VBx, constrained Hungarian, centroid AHC) land.
//!
//! Until that integration lands the module is `pub(crate)` (see
//! `src/lib.rs:62-72`), and every item below is dead from the rest of
//! the crate's perspective — Codex review HIGH (round 5). The
//! `#![allow(dead_code, unused_imports)]` below silences that noise
//! without weakening the production-callability gate; it comes off
//! the moment Phase 2/5 wires `PldaTransform` into the clusterer.

#![allow(dead_code, unused_imports)]

mod error;
mod loader;
mod transform;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod parity_tests;

pub use error::Error;
pub use transform::{PldaTransform, PostXvecEmbedding, RawEmbedding};

/// PLDA stage-1 / stage-2 dimension. Pyannote's
/// `pyannote/speaker-diarization-community-1` always uses 128.
pub const PLDA_DIMENSION: usize = 128;

/// WeSpeaker embedding dimension (input to `xvec_transform`).
pub const EMBEDDING_DIMENSION: usize = 256;
