//! Sans-I/O speaker diarization with pyannote-equivalent accuracy.
//!
//! `diarization` is the Rust port of [`pyannote.audio`](https://github.com/pyannote/pyannote-audio)'s
//! speaker-diarization pipeline. Two entrypoints, both running the same
//! pyannote `cluster_vbx` clustering pipeline (PLDA → AHC → VBx →
//! centroid → cosine → Hungarian → reconstruct):
//!
//! - [`offline::OwnedDiarizationPipeline`] — owned-audio batch path.
//!   Caller passes the entire 16 kHz mono PCM at once.
//! - [`streaming::StreamingOfflineDiarizer`] — voice-range-driven
//!   streaming path. Caller drives a VAD externally and pushes one
//!   voice range at a time; heavy stages 1+2 run eagerly per range,
//!   global clustering is deferred to `finalize`. Same DER as the
//!   offline path, plus per-range latency for the heavy work.
//!
//! ## Modules
//!
//! - [`segment`]: speaker-segmentation state machine
//!   (pyannote/segmentation-3.0 ONNX).
//! - [`embed`]: speaker-fingerprint generation (WeSpeaker ResNet34
//!   ONNX + kaldi fbank). `EmbedModel::embed_chunk_with_frame_mask`
//!   is the masked variant pyannote uses.
//! - [`plda`]: WeSpeaker PLDA whitening + length-norm.
//! - [`cluster`]: pyannote `cluster_vbx` primitives (AHC, VBx,
//!   centroid, Hungarian) plus a generic offline `cluster_offline`.
//! - [`pipeline`]: glues PLDA → cluster_vbx into a single
//!   `assign_embeddings` call.
//! - [`reconstruct`]: per-frame post-clustering smoothing.
//! - [`offline`]: owned-audio orchestrator (`OwnedDiarizationPipeline`).
//! - [`streaming`]: voice-range-driven orchestrator
//!   (`StreamingOfflineDiarizer`).
//!
//! ## Quick start (streaming-offline)
//!
//! ```no_run
//! # #[cfg(all(feature = "ort", feature = "bundled-segmentation"))]
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! use diarization::embed::EmbedModel;
//! use diarization::plda::PldaTransform;
//! use diarization::segment::SegmentModel;
//! use diarization::streaming::{StreamingOfflineConfig, StreamingOfflineDiarizer};
//!
//! // Segmentation + PLDA ship bundled in the crate; only the WeSpeaker
//! // embedding model (27 MB) is BYO.
//! let mut seg = SegmentModel::bundled()?;
//! let mut emb = EmbedModel::from_file("models/wespeaker_resnet34_lm.onnx")?;
//! let plda = PldaTransform::new()?;
//! let mut d = StreamingOfflineDiarizer::new(StreamingOfflineConfig::default());
//!
//! // Caller drives VAD externally; pushes one voice range at a time.
//! let samples: Vec<f32> = vec![/* 16 kHz mono PCM */];
//! d.push_voice_range(&mut seg, &mut emb, 0, &samples)?;
//! for span in d.finalize(&plda)? {
//!   println!(
//!     "[{:.2}s..{:.2}s] speaker {}",
//!     span.start_sample() as f64 / 16_000.0,
//!     span.end_sample() as f64 / 16_000.0,
//!     span.speaker_id()
//!   );
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## Design references
//!
//! See `docs/superpowers/specs/2026-04-26-dia-embed-cluster-diarizer-design.md`
//! for the load-bearing spec.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, allow(unused_attributes))]

pub mod cluster;
pub mod embed;
pub mod segment;

// Numerical primitives shared across the algorithm modules. Three-tier
// backend layout (scalar/arch/dispatch) modeled on the colconv crate.
// Crate-private — algorithm modules call into `ops::*`; downstream
// callers don't see this layer. `_bench` flips it to `pub` so external
// benches in `benches/ops.rs` can A/B scalar vs SIMD on the primitives
// directly.
#[cfg_attr(feature = "_bench", doc(hidden))]
#[cfg(feature = "_bench")]
pub mod ops;
#[cfg(not(feature = "_bench"))]
pub(crate) mod ops;

pub mod plda;

pub mod pipeline;

pub mod reconstruct;

pub mod aggregate;

pub mod offline;

#[cfg(feature = "ort")]
#[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
pub mod streaming;
