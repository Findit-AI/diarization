//! Spectral clustering. Filled in by Tasks 17-21.

use crate::{
  cluster::{Error, options::OfflineClusterOptions},
  embed::Embedding,
};

/// Stub. Real impl arrives in Tasks 17-21.
pub(crate) fn cluster(
  _embeddings: &[Embedding],
  _opts: &OfflineClusterOptions,
) -> Result<Vec<u64>, Error> {
  Err(Error::EmptyInput) // placeholder — never reached by Task 14 tests.
}
