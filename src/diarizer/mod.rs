//! Top-level streaming speaker-diarization orchestrator.
//!
//! Combines [`crate::segment`] + [`crate::embed`] + [`crate::cluster`] +
//! a per-frame reconstruction state machine. Spec §4.4 / §5.7-§5.12.
//!
//! Output: one [`DiarizedSpan`] per closed speaker turn (rev-6 shape;
//! NOT per-`(window, slot)` — that finer granularity lives on
//! [`CollectedEmbedding`]).

mod builder;
mod error;
mod span;

#[cfg(feature = "ort")]
mod imp;
#[cfg(feature = "ort")]
mod overlap;
#[cfg(feature = "ort")]
mod reconstruct;

pub use builder::{DiarizerBuilder, DiarizerOptions};
pub use error::{Error, InternalError};
pub use span::{CollectedEmbedding, DiarizedSpan};

#[cfg(feature = "ort")]
#[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
pub use imp::Diarizer;

// Compile-time trait assertions (spec §9). Catches a future field-type
// change that would silently regress Send/Sync auto-derive on the
// public types.
#[cfg(feature = "ort")]
const _: fn() = || {
  fn assert_send_sync<T: Send + Sync>() {}
  assert_send_sync::<Diarizer>();
  assert_send_sync::<DiarizerOptions>();
  assert_send_sync::<CollectedEmbedding>();
  assert_send_sync::<DiarizedSpan>();
  assert_send_sync::<Error>();
};

#[cfg(not(feature = "ort"))]
const _: fn() = || {
  fn assert_send_sync<T: Send + Sync>() {}
  assert_send_sync::<DiarizerOptions>();
  assert_send_sync::<CollectedEmbedding>();
  assert_send_sync::<DiarizedSpan>();
  assert_send_sync::<Error>();
};
