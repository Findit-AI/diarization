//! Offline (non-streaming) diarization.
//!
//! Wraps the full pyannote `cluster_vbx` flow: PLDA projection on
//! active embeddings → AHC initial clustering → VBx EM → centroid
//! computation → cosine cdist + constrained Hungarian assignment →
//! frame-level reconstruction → RTTM emission. Bit-exact pyannote
//! parity on the 5 short captured fixtures.
//!
//! ## Where this fits
//!
//! - The streaming [`crate::diarizer::Diarizer`] runs an online
//!   cosine + EMA clusterer. It is fast and works on live audio
//!   without seeing the future, but its DER on the captured
//!   community-1 fixtures is ~20-40% (the online clusterer
//!   over-splits compared to pyannote's batch VBx).
//! - This module runs the full pyannote `community-1` clustering
//!   flow as a *batch* operation on already-computed segmentation +
//!   raw-embedding tensors. DER ≈ 0% on the 5 short captured
//!   fixtures (length-dependent divergence at T=1004; tracked
//!   separately).
//!
//! ## What this module accepts
//!
//! [`OfflineInput`] takes pre-computed (segmentation, raw embedding)
//! tensors. The caller is responsible for running segmentation +
//! embedding ONNX inference. Two production sources:
//!
//! 1. The captured pyannote fixtures (`tests/parity/fixtures/*/`)
//!    — used by the parity tests in this module.
//! 2. Custom ONNX inference using [`crate::segment::SegmentModel`] +
//!    [`crate::embed::EmbedModel`].
//!
//! ## Why not feature-gate this behind `ort`
//!
//! The offline pipeline math is pure compute over [`f64`]/[`f32`]
//! tensors — no ONNX inference inside this function. It compiles and
//! runs without the `ort` feature. Useful for downstream consumers
//! that have their own inference path (e.g. CoreML, custom CUDA).

mod algo;

#[cfg(feature = "ort")]
mod owned;

#[cfg(test)]
mod parity_tests;

#[cfg(all(test, feature = "ort"))]
mod owned_smoke_tests;

pub use algo::{Error, OfflineInput, OfflineOutput, diarize_offline};

#[cfg(feature = "ort")]
#[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
pub use owned::{OwnedDiarizationPipeline, OwnedPipelineOptions, SLOTS_PER_CHUNK};

/// Reused by [`crate::streaming::offline_diarizer`] for the same
/// onset-range validation it performs on its
/// [`OwnedPipelineOptions`]-derived config.
#[cfg(feature = "ort")]
pub(crate) use owned::check_onset;
