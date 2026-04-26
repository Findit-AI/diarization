//! Error type for `dia::cluster`.

/// Errors returned by [`Clusterer::submit`](crate::cluster::Clusterer::submit)
/// and [`cluster_offline`](crate::cluster::cluster_offline).
#[derive(Debug, thiserror::Error)]
pub enum Error {
  /// The embedding slice passed to `cluster_offline` was empty.
  #[error("empty input: at least one embedding is required")]
  EmptyInput,

  /// `max_speakers` was set to zero.
  #[error("max_speakers must be at least 1")]
  ZeroMaxSpeakers,

  /// `min_speakers` was set to zero.
  #[error("min_speakers must be at least 1")]
  ZeroMinSpeakers,

  /// `min_speakers` exceeds `max_speakers`.
  #[error("min_speakers ({min}) must not exceed max_speakers ({max})")]
  MinExceedsMax {
    /// The `min_speakers` value supplied.
    min: u32,
    /// The `max_speakers` value supplied.
    max: u32,
  },

  /// The number of open speaker slots has reached `max_speakers` and
  /// [`OverflowStrategy::Reject`](crate::cluster::OverflowStrategy::Reject)
  /// is active.
  #[error("too many speakers: limit of {limit} reached")]
  TooManySpeakers {
    /// The `max_speakers` cap that was hit.
    limit: u32,
  },

  /// EMA α was outside `(0.0, 1.0]`.
  #[error("invalid EMA alpha {alpha}: must be in (0, 1]")]
  InvalidEmaAlpha {
    /// The invalid α value.
    alpha: f32,
  },

  /// Similarity threshold was outside `[0.0, 1.0]`.
  #[error("invalid similarity threshold {threshold}: must be in [0, 1]")]
  InvalidThreshold {
    /// The invalid threshold value.
    threshold: f32,
  },

  /// The number of embeddings in the input exceeds the platform's
  /// `usize::MAX` or an internal allocation limit.
  #[error("input too large: {count} embeddings exceed the processing limit")]
  InputTooLarge {
    /// Number of embeddings that were supplied.
    count: usize,
  },
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn empty_input_display() {
    let e = Error::EmptyInput;
    assert_eq!(
      e.to_string(),
      "empty input: at least one embedding is required"
    );
  }

  #[test]
  fn too_many_speakers_display() {
    let e = Error::TooManySpeakers { limit: 15 };
    assert_eq!(e.to_string(), "too many speakers: limit of 15 reached");
  }
}
