//! Sans-I/O streaming speaker diarization for variable-length VAD-filtered audio.
//!
//! `dia` is the Rust port of [`pyannote.audio`](https://github.com/pyannote/pyannote-audio)'s
//! speaker-diarization pipeline, restructured around incremental push-based
//! state machines so it can run live on streaming audio. Inputs are arbitrary-
//! length pushes (e.g., per VAD speech region); outputs are emitted per closed
//! speaker turn as windows finalize.
//!
//! ## Modules
//!
//! - [`segment`]: speaker-segmentation state machine
//!   (pyannote/segmentation-3.0 ONNX). Emits `Action::Activity` (per-(window,
//!   slot) speaker presence) and `Action::SpeakerScores` (per-window per-frame
//!   raw probabilities for downstream reconstruction).
//! - [`embed`]: speaker-fingerprint generation (WeSpeaker ResNet34 ONNX +
//!   kaldi fbank). High-level `EmbedModel::embed` (sliding-window mean) and
//!   the masked variant `embed_masked` (rev-8 gather-and-embed).
//! - [`cluster`]: cross-window speaker linking. Online streaming `Clusterer`
//!   plus offline batch `cluster_offline` with spectral (default) and
//!   agglomerative methods.
//! - [`diarizer`]: top-level `Diarizer` orchestrator. Combines the above three
//!   plus a per-frame reconstruction state machine matching pyannote's
//!   `SpeakerDiarization.apply` pipeline.
//!
//! ## Quick start
//!
//! ```no_run
//! # #[cfg(feature = "ort")]
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! use dia::diarizer::{Diarizer, DiarizerOptions};
//! use dia::embed::EmbedModel;
//! use dia::segment::SegmentModel;
//!
//! let mut seg = SegmentModel::from_file("models/segmentation-3.0.onnx")?;
//! let mut emb = EmbedModel::from_file("models/wespeaker_resnet34_lm.onnx")?;
//! let mut d = Diarizer::new(DiarizerOptions::default());
//!
//! let samples: Vec<f32> = vec![/* 16 kHz mono PCM */];
//! d.process_samples(&mut seg, &mut emb, &samples, |span| {
//!   println!(
//!     "[{:.2}s..{:.2}s] speaker {}",
//!     span.range().start_pts() as f64 / 16_000.0,
//!     span.range().end_pts() as f64 / 16_000.0,
//!     span.speaker_id()
//!   );
//! })?;
//! d.finish_stream(&mut seg, &mut emb, |_| {})?;
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
pub mod diarizer;
pub mod embed;
// `plda` is intentionally crate-private in v0.1.0. Phase 1 ships
// the math (xvec_tf + plda_tf, parity-validated against pyannote)
// but `RawEmbedding`'s only constructor is `#[cfg(test)]` because
// any public constructor would have to admit arbitrary `[f32; 256]`
// — and admitting an L2-normalized vector there silently corrupts
// downstream VBx (Codex review HIGH, rounds 2–5). The Phase-5
// integration will own a single typed entry from `EmbedModel`'s
// raw-output path; that's when `plda` flips back to `pub`.
pub(crate) mod plda;
// `vbx` is intentionally crate-private in v0.1.0 for the same
// reason as `plda`: the math ships here but the public-API
// integration with `EmbedModel` + `Diarizer` lands in Phase 5.
pub mod segment;
pub(crate) mod vbx;
