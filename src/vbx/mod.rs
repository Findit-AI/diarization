//! Variational Bayes HMM speaker clustering (VBx).
//!
//! Ports `pyannote.audio.utils.vbx.VBx` (`utils/vbx.py:27-137` in
//! pyannote.audio 4.0.4) to Rust. Consumes the post-PLDA features
//! produced by `dia::plda::PldaTransform::project()` plus the
//! eigenvalue diagonal `dia::plda::PldaTransform::phi()`, runs
//! variational EM iterations, and returns final speaker
//! responsibilities + priors + ELBO trajectory.
//!
//! ## Standalone — no `Diarizer` integration yet
//!
//! Phase 2 ships VBx as a pure-math module. The integration
//! (`Diarizer` consuming VBx output → cluster centroids → per-frame
//! diarization) lands in Phase 5. Until then `dia::vbx` is
//! crate-private (see `src/lib.rs:62-72`).

#![allow(dead_code, unused_imports)]

mod algo;
mod error;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod parity_tests;

pub use algo::{MAX_ITERS_CAP, PiInit, StopReason, VbxOutput, vbx_iterate};
pub use error::Error;
