//! Constrained Hungarian assignment (per-chunk maximum-weight matching).
//!
//! Ports `pyannote.audio.pipelines.clustering.SpeakerEmbedding.constrained_argmax`
//! (`clustering.py:127-140` in pyannote.audio 4.0.4). Pyannote runs
//! `scipy.optimize.linear_sum_assignment(cost, maximize=True)` on each chunk
//! of a `(num_chunks, num_speakers, num_clusters)` tensor. This Rust port
//! operates per chunk; the caller iterates chunks.

use crate::hungarian::error::Error;
use nalgebra::DMatrix;
use ordered_float::NotNan;
use pathfinding::prelude::{Matrix, kuhn_munkres};

/// Sentinel value for an unmatched speaker. Matches pyannote's
/// `-2 * np.ones((num_chunks, num_speakers), dtype=np.int8)` initializer
/// in `constrained_argmax`.
pub const UNMATCHED: i32 = -2;

/// Per-chunk constrained Hungarian assignment.
///
/// Given a `(num_speakers, num_clusters)` cost matrix, returns the
/// maximum-weight bipartite matching as `Vec<i32>` of length `num_speakers`.
/// Each entry is the cluster index assigned to that speaker, or
/// [`UNMATCHED`] (`-2`) if the speaker had no cluster left (only possible
/// when `num_speakers > num_clusters`).
///
/// # Errors
///
/// - [`Error::Shape`] if either dimension is zero.
/// - [`Error::NonFinite`] if any entry is NaN/`±inf`. Pyannote pre-replaces
///   NaN with `np.nanmin(soft_clusters)` (the global min across all chunks);
///   this Rust port instead fail-fasts at the boundary, since production
///   embeddings produce finite cosine distances and a non-finite cost
///   indicates upstream corruption that should not silently proceed.
///
/// # Algorithm
///
/// `pathfinding::kuhn_munkres` requires `rows <= columns`. When
/// `num_speakers > num_clusters` the cost matrix is transposed to
/// `(num_clusters, num_speakers)` before running kuhn_munkres, and the
/// resulting `cluster → speaker` assignment is inverted.
pub fn constrained_argmax(soft_clusters: &DMatrix<f64>) -> Result<Vec<i32>, Error> {
  let (num_speakers, num_clusters) = soft_clusters.shape();
  if num_speakers == 0 {
    return Err(Error::Shape("num_speakers must be at least 1"));
  }
  if num_clusters == 0 {
    return Err(Error::Shape("num_clusters must be at least 1"));
  }
  for s in 0..num_speakers {
    for k in 0..num_clusters {
      if !soft_clusters[(s, k)].is_finite() {
        return Err(Error::NonFinite("soft_clusters"));
      }
    }
  }

  let mut assignment = vec![UNMATCHED; num_speakers];

  if num_speakers <= num_clusters {
    // Direct path: rows = speakers, cols = clusters.
    let mut data = Vec::with_capacity(num_speakers * num_clusters);
    for s in 0..num_speakers {
      for k in 0..num_clusters {
        data.push(NotNan::new(soft_clusters[(s, k)]).expect("finite (checked above)"));
      }
    }
    let weights =
      Matrix::from_vec(num_speakers, num_clusters, data).expect("matrix dims match data length");
    let (_total, speaker_to_cluster) = kuhn_munkres(&weights);
    for (s, &k) in speaker_to_cluster.iter().enumerate() {
      assignment[s] = i32::try_from(k).expect("cluster idx fits in i32");
    }
  } else {
    // Transpose path: rows = clusters, cols = speakers.
    let mut data = Vec::with_capacity(num_clusters * num_speakers);
    for k in 0..num_clusters {
      for s in 0..num_speakers {
        data.push(NotNan::new(soft_clusters[(s, k)]).expect("finite (checked above)"));
      }
    }
    let weights =
      Matrix::from_vec(num_clusters, num_speakers, data).expect("matrix dims match data length");
    let (_total, cluster_to_speaker) = kuhn_munkres(&weights);
    for (k, &s) in cluster_to_speaker.iter().enumerate() {
      assignment[s] = i32::try_from(k).expect("cluster idx fits in i32");
    }
  }

  Ok(assignment)
}
