//! Offline batch clustering entry point + shared helpers.
//! Spec §5.5 / §5.6.

use crate::{
  cluster::{
    Error, agglomerative,
    options::{OfflineClusterOptions, OfflineMethod},
    spectral,
  },
  embed::{Embedding, NORM_EPSILON},
};

/// Validate inputs to [`cluster_offline`]. Returns the input length on
/// success. Shared between spectral (§5.5 step 0) and agglomerative
/// (§5.6 step 0) — same checks, same error variants, same order.
pub(crate) fn validate_offline_input(
  embeddings: &[Embedding],
  target_speakers: Option<u32>,
) -> Result<usize, Error> {
  if embeddings.is_empty() {
    return Err(Error::EmptyInput);
  }
  for e in embeddings {
    // f64 accumulator: 256 squared-f32 terms can lose ~8 bits of mantissa
    // in f32 (sum of values ~1.0). Promote for stability, demote at the
    // end. Mirrors online.rs::update_speaker. Not perf-critical — runs
    // once per embedding at validation time.
    let mut sq = 0.0f64;
    for &x in e.as_array() {
      if !x.is_finite() {
        return Err(Error::NonFiniteInput);
      }
      sq += (x as f64) * (x as f64);
    }
    if (sq.sqrt() as f32) < NORM_EPSILON {
      return Err(Error::DegenerateEmbedding);
    }
  }
  let n = embeddings.len();
  if let Some(k) = target_speakers {
    if k < 1 {
      return Err(Error::TargetTooSmall);
    }
    if (k as usize) > n {
      return Err(Error::TargetExceedsInput { target: k, n });
    }
  }
  Ok(n)
}

/// Cluster a batch of embeddings; returns one global speaker id per
/// input, parallel to the input slice.
///
/// Validates input first (empty list, non-finite values, zero-norm
/// embeddings, invalid `target_speakers`), then short-circuits the
/// `N==1` and `N==2` cases (spec §5.5 step 0.1, §5.6 step 0.1), then
/// dispatches to the configured [`OfflineMethod`].
pub fn cluster_offline(
  embeddings: &[Embedding],
  opts: &OfflineClusterOptions,
) -> Result<Vec<u64>, Error> {
  let n = validate_offline_input(embeddings, opts.target_speakers())?;

  // Fast paths (spec §5.5 step 0.1 / §5.6 step 0.1).
  if n == 1 {
    return Ok(vec![0]);
  }
  if n == 2 {
    let sim = embeddings[0].similarity(&embeddings[1]).max(0.0);
    return Ok(match opts.target_speakers() {
      Some(2) => vec![0, 1],
      Some(1) => vec![0, 0],
      _ => {
        if sim >= opts.similarity_threshold() {
          vec![0, 0]
        } else {
          vec![0, 1]
        }
      }
    });
  }

  // Dispatch.
  match opts.method() {
    OfflineMethod::Agglomerative { linkage } => agglomerative::cluster(embeddings, linkage, opts),
    OfflineMethod::Spectral => spectral::cluster(embeddings, opts),
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::embed::EMBEDDING_DIM;

  fn unit(i: usize) -> Embedding {
    let mut v = [0.0f32; EMBEDDING_DIM];
    v[i] = 1.0;
    Embedding::normalize_from(v).unwrap()
  }

  #[test]
  fn empty_input_errors() {
    let r = cluster_offline(&[], &OfflineClusterOptions::default());
    assert!(matches!(r, Err(Error::EmptyInput)));
  }

  #[test]
  fn target_speakers_zero_errors() {
    let r = cluster_offline(
      &[unit(0)],
      &OfflineClusterOptions::default().with_target_speakers(0),
    );
    assert!(matches!(r, Err(Error::TargetTooSmall)));
  }

  #[test]
  fn target_speakers_exceeds_input_errors() {
    let r = cluster_offline(
      &[unit(0), unit(1)],
      &OfflineClusterOptions::default().with_target_speakers(5),
    );
    assert!(matches!(
      r,
      Err(Error::TargetExceedsInput { target: 5, n: 2 })
    ));
  }

  #[test]
  fn fast_path_n_eq_1() {
    let r = cluster_offline(&[unit(0)], &OfflineClusterOptions::default()).unwrap();
    assert_eq!(r, vec![0]);
  }

  #[test]
  fn fast_path_n_eq_2_similar() {
    // Both identical → cosine = 1.0 >= 0.5 threshold → one cluster.
    let mut v = [0.0f32; EMBEDDING_DIM];
    v[0] = 1.0;
    let e = Embedding::normalize_from(v).unwrap();
    let r = cluster_offline(&[e, e], &OfflineClusterOptions::default()).unwrap();
    assert_eq!(r, vec![0, 0]);
  }

  #[test]
  fn fast_path_n_eq_2_dissimilar() {
    // Orthogonal → cosine = 0 < 0.5 → two clusters.
    let r = cluster_offline(&[unit(0), unit(1)], &OfflineClusterOptions::default()).unwrap();
    assert_eq!(r, vec![0, 1]);
  }

  #[test]
  fn fast_path_n_eq_2_target_forces() {
    let r1 = cluster_offline(
      &[unit(0), unit(0)],
      &OfflineClusterOptions::default().with_target_speakers(2),
    )
    .unwrap();
    assert_eq!(
      r1,
      vec![0, 1],
      "target=2 forces 2 clusters even when identical"
    );
    let r2 = cluster_offline(
      &[unit(0), unit(1)],
      &OfflineClusterOptions::default().with_target_speakers(1),
    )
    .unwrap();
    assert_eq!(
      r2,
      vec![0, 0],
      "target=1 forces 1 cluster even when orthogonal"
    );
  }

  #[test]
  fn nan_input_errors() {
    let mut v = [0.0f32; EMBEDDING_DIM];
    v[0] = f32::NAN;
    // Bypass the public Embedding constructor which would reject NaN.
    let e = Embedding(v);
    let r = cluster_offline(&[e, unit(0)], &OfflineClusterOptions::default());
    assert!(matches!(r, Err(Error::NonFiniteInput)));
  }

  #[test]
  fn zero_norm_input_errors() {
    let e = Embedding([0.0f32; EMBEDDING_DIM]);
    let r = cluster_offline(&[e, unit(0)], &OfflineClusterOptions::default());
    assert!(matches!(r, Err(Error::DegenerateEmbedding)));
  }

  #[test]
  fn validate_returns_n_on_valid_no_target() {
    let n = validate_offline_input(&[unit(0), unit(1), unit(2)], None).unwrap();
    assert_eq!(n, 3);
  }

  #[test]
  fn validate_returns_n_on_valid_with_target() {
    let n = validate_offline_input(&[unit(0), unit(1), unit(2)], Some(2)).unwrap();
    assert_eq!(n, 3);
  }

  #[test]
  fn validate_target_equals_n_ok() {
    // target == n is allowed (every embedding can be its own cluster).
    let n = validate_offline_input(&[unit(0), unit(1)], Some(2)).unwrap();
    assert_eq!(n, 2);
  }
}
