//! Spectral clustering. Filled in by Tasks 17-21.

use crate::{
  cluster::{Error, options::OfflineClusterOptions},
  embed::Embedding,
};

/// Stub. Real implementation arrives in Tasks 17-21. Currently
/// unreachable because `cluster_offline` short-circuits N<=2 and
/// Task 14 tests never invoke an N>=3 dispatch.
pub(crate) fn cluster(
  _embeddings: &[Embedding],
  _opts: &OfflineClusterOptions,
) -> Result<Vec<u64>, Error> {
  unreachable!("spectral::cluster stub — Tasks 17-21 will implement this")
}
