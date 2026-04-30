//! Speaker clustering — online streaming ([`Clusterer`]), offline batch
//! ([`cluster_offline`]), and the pyannote `cluster_vbx`-pipeline
//! primitives ([`ahc`], [`vbx`], [`centroid`], [`hungarian`]).
//!
//! # Online path
//! [`Clusterer`] accepts one embedding at a time and maintains a set of
//! per-speaker centroids.  Call [`Clusterer::submit`] for each embedding
//! produced by `diarization::embed`; it returns a [`ClusterAssignment`] that carries
//! the globally-unique speaker id and whether a new speaker was opened.
//!
//! # Offline path
//! [`cluster_offline`] takes a slice of embeddings and returns a
//! `Vec<u64>` of speaker labels (one per embedding). Dispatches to
//! [`agglomerative`](OfflineMethod::Agglomerative) (Single / Complete /
//! Average linkage) or [`spectral`](OfflineMethod::Spectral) (default;
//! eigengap-K detection + K-means++ + Lloyd refinement, byte-deterministic
//! via [`ChaCha8Rng`](rand_chacha::ChaCha8Rng)).
//!
//! # Pyannote `cluster_vbx` primitives
//! The [`ahc`], [`vbx`], [`centroid`], and [`hungarian`] submodules are
//! the algorithm-level building blocks of the pyannote
//! `clustering.VBxClustering` pipeline. They're orchestrated by
//! [`crate::pipeline::assign_embeddings`] and
//! [`crate::offline::diarize_offline`]. Direct use is uncommon — the
//! pipeline / offline entrypoints are the supported API surface.

pub mod ahc;
pub mod centroid;
pub mod hungarian;
pub mod vbx;

mod error;
mod options;
mod types;

pub use crate::embed::Embedding;
pub use error::Error;
pub use offline::cluster_offline;
pub use online::Clusterer;
pub use options::{
  ClusterOptions, DEFAULT_EMA_ALPHA, DEFAULT_SIMILARITY_THRESHOLD, Linkage, MAX_AUTO_SPEAKERS,
  MAX_OFFLINE_INPUT, OfflineClusterOptions, OfflineMethod, OverflowStrategy, UpdateStrategy,
};
pub use types::{ClusterAssignment, SpeakerCentroid};

mod agglomerative;
mod offline;
mod online;
mod spectral;

#[cfg(test)]
mod test_util;
#[cfg(test)]
mod tests;

// Compile-time trait assertions. Catches a future field-type change that
// would silently regress Send/Sync auto-derive on the public types.
const _: fn() = || {
  fn assert_send_sync<T: Send + Sync>() {}
  assert_send_sync::<Clusterer>();
  assert_send_sync::<ClusterOptions>();
  assert_send_sync::<OfflineClusterOptions>();
  assert_send_sync::<ClusterAssignment>();
  assert_send_sync::<SpeakerCentroid>();
  assert_send_sync::<Error>();
};
