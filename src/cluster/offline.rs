//! Filled in by Task 14.
// TODO(task-14): replace stub with validate_offline_input + cluster_offline
use crate::{
  cluster::{Error, OfflineClusterOptions},
  embed::Embedding,
};
pub fn cluster_offline(_e: &[Embedding], _o: &OfflineClusterOptions) -> Result<Vec<u64>, Error> {
  Err(Error::EmptyInput)
}
