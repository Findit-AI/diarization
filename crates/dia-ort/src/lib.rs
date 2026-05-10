//! ONNX Runtime (ort) inference backend for the dia diarization
//! pipeline.
//!
//! This crate is a thin re-export shim over [`diarization`] (a.k.a.
//! `dia-core`) with the `ort` feature pre-activated. The intent of
//! the split is to give downstream users a single `cargo add
//! dia-ort` entry-point — no feature-flag spelunking — and to make
//! room for the ORT inference code to migrate out of `dia-core`
//! later without changing the public surface that integrations
//! depend on.
//!
//! # Today's layout
//!
//! Everything you see exported from this crate is `pub use`'d from
//! `diarization`. The cfg-gated `ort` modules still physically
//! live under `crates/dia-core/src/embed/` and `segment/`. Phase B
//! step 2 physically migrates them without breaking the surface.
//!
//! # Usage
//!
//! ```no_run
//! use dia_ort::{SegmentModel, EmbedModel};
//!
//! // segmentation: bundled pyannote/segmentation-3.0 (default-on
//! // via the `bundled-segmentation` feature)
//! let _seg = SegmentModel::bundled().unwrap();
//!
//! // embedding: WeSpeaker ResNet34-LM (load from disk)
//! // let emb = EmbedModel::from_file("wespeaker_resnet34_lm.onnx").unwrap();
//! # let _ = EmbedModel::from_file as fn(_) -> _;
//! ```

#![doc(html_root_url = "https://docs.rs/dia-ort/0.1.0")]

// Re-export everything `diarization` exposes. Downstream code can
// migrate from `use diarization::*;` to `use dia_ort::*;` without
// any other change today.
pub use diarization::*;
