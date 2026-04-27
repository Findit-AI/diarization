//! Hierarchical agglomerative clustering. Filled in by Task 15.

use crate::{
  cluster::{
    Error,
    options::{Linkage, OfflineClusterOptions},
  },
  embed::Embedding,
};

/// Stub. Real impl arrives in Task 15.
pub(crate) fn cluster(
  _embeddings: &[Embedding],
  _linkage: Linkage,
  _opts: &OfflineClusterOptions,
) -> Result<Vec<u64>, Error> {
  Err(Error::EmptyInput) // placeholder — never reached by Task 14 tests (all use N<=2 fast paths or validation errors).
}
