//! TorchScript (tch / libtorch) inference backend for the dia
//! diarization pipeline.
//!
//! This crate is a thin re-export shim over [`diarization`] (a.k.a.
//! `dia-core`). With the `tch` feature on, the embedding model can
//! be loaded from a TorchScript `.pt` file via
//! [`EmbedModel::from_torchscript_file`] — that path is bit-exact
//! with pyannote's PyTorch inference on heavy-overlap fixtures
//! (where ONNX→ORT can diverge by O(1) per element). Segmentation
//! always runs on ORT (no TorchScript export of
//! pyannote/segmentation-3.0 is shipped with this repo).
//!
//! # Why a separate crate
//!
//! Same shape as [`dia_ort`](https://docs.rs/dia-ort): pull-mode
//! split so downstream users get a single `cargo add dia-tch`
//! entry-point + room for the actual `tch` code to migrate out of
//! `dia-core` later. Today the `tch_backend` lives under
//! `crates/dia-core/src/embed/model.rs` cfg-gated by
//! `feature = "tch"`; Phase B step 2 moves it physically.
//!
//! # libtorch setup
//!
//! `tch` requires a libtorch install at runtime. Either:
//! - set `LIBTORCH` to an extracted libtorch tree, or
//! - opt into `tch`'s `download-libtorch` feature in your own
//!   Cargo.toml.
//!
//! See the [`tch` docs](https://github.com/LaurentMazare/tch-rs)
//! for the supported version matrix.
//!
//! # Usage
//!
//! ```no_run
//! # #[cfg(feature = "tch")]
//! # {
//! use dia_tch::EmbedModel;
//! // bit-exact-pyannote embedding via TorchScript
//! let _emb = EmbedModel::from_torchscript_file("wespeaker.pt").unwrap();
//! # }
//! ```

#![doc(html_root_url = "https://docs.rs/dia-tch/0.1.0")]

pub use diarization::*;
