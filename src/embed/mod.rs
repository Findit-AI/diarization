//! Speaker fingerprint generation: WeSpeaker ResNet34 ONNX wrapper +
//! kaldi-compatible fbank + sliding-window mean for variable-length clips.
//!
//! See the crate-level docs and `docs/superpowers/specs/` for the design.
//! Layered API (spec §2.3):
//! - High-level: `EmbedModel::embed`, `embed_weighted`, `embed_masked` (added in phase 5)
//! - Low-level: `compute_fbank`, `EmbedModel::embed_features`,
//!   `EmbedModel::embed_features_batch` (added in phase 5)

mod error;
mod fbank;
mod options;
mod types;

pub use error::Error;
pub use fbank::compute_fbank;
pub use options::{
  EMBED_WINDOW_SAMPLES, EMBEDDING_DIM, FBANK_FRAMES, FBANK_NUM_MELS, HOP_SAMPLES, MIN_CLIP_SAMPLES,
  NORM_EPSILON,
};
pub use types::{Embedding, EmbeddingMeta, EmbeddingResult, cosine_similarity};

// Compile-time trait assertions. Catches a future field-type change that
// would silently regress Send/Sync auto-derive on the public types.
const _: fn() = || {
  fn assert_send_sync<T: Send + Sync>() {}
  assert_send_sync::<Embedding>();
  assert_send_sync::<EmbeddingMeta>();
  assert_send_sync::<EmbeddingResult>();
  assert_send_sync::<Error>();
};
