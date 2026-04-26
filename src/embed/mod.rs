//! Speaker fingerprint generation: WeSpeaker ResNet34 ONNX wrapper +
//! kaldi-compatible fbank + sliding-window mean for variable-length clips.
//!
//! See the crate-level docs and `docs/superpowers/specs/` for the design.
//! Layered API (spec §2.3):
//! - High-level: [`EmbedModel::embed`], [`embed_weighted`], [`embed_masked`]
//! - Low-level: [`compute_fbank`], [`EmbedModel::embed_features`],
//!   [`EmbedModel::embed_features_batch`]

mod error;
mod options;
mod types;

// pub use error::Error;       // re-enabled in Task 8
pub use options::{
  EMBED_WINDOW_SAMPLES, EMBEDDING_DIM, FBANK_FRAMES, FBANK_NUM_MELS, HOP_SAMPLES, MIN_CLIP_SAMPLES,
  NORM_EPSILON,
};
pub use types::{Embedding, EmbeddingMeta, EmbeddingResult, cosine_similarity};
