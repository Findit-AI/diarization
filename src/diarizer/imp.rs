//! `Diarizer` implementation. Filled in by Task 36.

use crate::diarizer::{builder::DiarizerOptions, error::Error};

/// Streaming speaker diarizer.
pub struct Diarizer {
  #[allow(dead_code)]
  opts: DiarizerOptions,
}

impl Diarizer {
  /// Stub — real constructor lands in Task 36.
  #[allow(dead_code)]
  pub(crate) fn _placeholder(opts: DiarizerOptions) -> Result<Self, Error> {
    let _ = opts;
    unimplemented!("Diarizer::new arrives in Task 36")
  }
}
