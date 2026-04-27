//! Online streaming speaker clustering (`Clusterer`) plus offline batch
//! clustering (`cluster_offline`).
//!
//! # Online path
//! [`Clusterer`] accepts one embedding at a time and maintains a set of
//! per-speaker centroids.  Call [`Clusterer::submit`] for each embedding
//! produced by `dia::embed`; it returns a [`ClusterAssignment`] that carries
//! the globally-unique speaker id and whether a new speaker was opened.
//!
//! # Offline path
//! [`cluster_offline`] takes a slice of embeddings and returns a
//! `Vec<u64>` of speaker labels (one per embedding).  Filled in by
//! Task 14 (agglomerative) and Phase 4 (spectral).

mod error;
mod options;
mod types;

pub use error::Error;
pub use offline::cluster_offline;
pub use online::Clusterer;
pub use options::{
  ClusterOptions, DEFAULT_EMA_ALPHA, DEFAULT_SIMILARITY_THRESHOLD, Linkage, MAX_AUTO_SPEAKERS,
  OfflineClusterOptions, OfflineMethod, OverflowStrategy, UpdateStrategy,
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
