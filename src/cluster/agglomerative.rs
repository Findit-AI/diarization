//! Hierarchical agglomerative clustering. Filled in by Task 15.

use crate::{
  cluster::{
    Error,
    options::{Linkage, OfflineClusterOptions},
  },
  embed::Embedding,
};

/// Stub. Real implementation arrives in Task 15. Currently unreachable
/// because `cluster_offline` short-circuits N<=2 and Task 14 tests
/// never invoke an N>=3 dispatch.
pub(crate) fn cluster(
  _embeddings: &[Embedding],
  _linkage: Linkage,
  _opts: &OfflineClusterOptions,
) -> Result<Vec<u64>, Error> {
  unreachable!("agglomerative::cluster stub — Task 15 will implement this")
}
