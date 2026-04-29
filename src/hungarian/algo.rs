//! Constrained Hungarian assignment (per-chunk maximum-weight matching).
//!
//! Ports `pyannote.audio.pipelines.clustering.SpeakerEmbedding.constrained_argmax`
//! (`clustering.py:127-140` in pyannote.audio 4.0.4). Pyannote takes the
//! full `(num_chunks, num_speakers, num_clusters)` cost tensor, replaces
//! NaN entries with the *global* `np.nanmin(soft_clusters)`, and runs
//! `scipy.optimize.linear_sum_assignment(cost, maximize=True)` per chunk.

use crate::hungarian::error::Error;
use nalgebra::DMatrix;
use ordered_float::NotNan;
use pathfinding::prelude::{Matrix, kuhn_munkres};

/// Sentinel value for an unmatched speaker. Matches pyannote's
/// `-2 * np.ones((num_chunks, num_speakers), dtype=np.int8)` initializer.
pub const UNMATCHED: i32 = -2;

/// Batched constrained Hungarian assignment over a stack of per-chunk
/// `(num_speakers, num_clusters)` cost matrices.
///
/// Returns one `Vec<i32>` of length `num_speakers` per chunk. Each entry is
/// the cluster index assigned to that speaker, or [`UNMATCHED`] (`-2`) if
/// the speaker had no cluster left (only possible when
/// `num_speakers > num_clusters`).
///
/// # Pyannote parity: `np.nan_to_num` semantics (NaN only)
///
/// Pyannote's `constrained_argmax` runs `np.nan_to_num(soft_clusters,
/// nan=np.nanmin(soft_clusters))` before per-chunk matching. The realistic
/// NaN source is an empty AHC cluster whose centroid is `NaN/NaN` after
/// averaging zero embeddings; the Rust port replicates that:
///
/// - **NaN** → global `nanmin` across all finite entries
///   (`np.nanmin`-equivalent on the production path where `±inf` cannot
///   appear).
///
/// `±inf` is **rejected** rather than substituted with `f64::MAX/MIN`
/// (numpy's `nan_to_num` defaults). Two reasons:
///
/// 1. Production cosine distances over finite embeddings are always
///    finite, so `±inf` indicates upstream corruption rather than a
///    well-defined edge case the algorithm should silently handle.
/// 2. `pathfinding::kuhn_munkres` does `lx[root] + ly[y]` and other
///    accumulating arithmetic on the costs; feeding `f64::MAX` risks
///    overflow into `±inf`/`NaN` in the slack labelling, and the crate
///    docs explicitly warn that *"indefinite values such as positive or
///    negative infinity or NaN can cause this function to loop endlessly"*.
///    Rejecting at the boundary keeps the solver inside its safe
///    operating envelope.
///
/// # Errors
///
/// - [`Error::Shape`] if `chunks` is empty, any chunk has zero rows or
///   zero columns, or chunks differ in shape.
/// - [`Error::NonFinite`] if any chunk contains `+inf` or `-inf`, or if
///   *every* entry across all chunks is NaN (no finite value to use as
///   the `nanmin` replacement). Pyannote degenerates in the all-NaN case
///   too (`np.nanmin` returns NaN, and the resulting assignment is
///   undefined).
///
/// # Algorithm
///
/// `pathfinding::kuhn_munkres` requires `rows <= columns`. When
/// `num_speakers > num_clusters` the cost matrix is transposed to
/// `(num_clusters, num_speakers)` before running kuhn_munkres, and the
/// resulting `cluster → speaker` assignment is inverted.
pub fn constrained_argmax(chunks: &[DMatrix<f64>]) -> Result<Vec<Vec<i32>>, Error> {
  if chunks.is_empty() {
    return Err(Error::Shape("chunks must contain at least one chunk"));
  }
  let (num_speakers, num_clusters) = chunks[0].shape();
  if num_speakers == 0 {
    return Err(Error::Shape("num_speakers must be at least 1"));
  }
  if num_clusters == 0 {
    return Err(Error::Shape("num_clusters must be at least 1"));
  }
  for chunk in chunks {
    if chunk.shape() != (num_speakers, num_clusters) {
      return Err(Error::Shape("all chunks must share the same shape"));
    }
  }

  // Reject ±inf upfront. Numpy's `np.nan_to_num` substitutes them with
  // `f64::MAX/MIN`, but feeding those values into `kuhn_munkres`'s
  // accumulating slack arithmetic risks overflow into `±inf`/`NaN` and
  // can wedge the solver per the crate's own docs. Production cosine
  // distances are always finite, so `±inf` here indicates upstream
  // corruption — surface a clear typed error rather than silently
  // proceed with values that may panic the solver. Codex review HIGH
  // round 2 of Phase 3.
  for chunk in chunks {
    for &v in chunk.iter() {
      if v.is_infinite() {
        return Err(Error::NonFinite("soft_clusters contains +inf or -inf"));
      }
    }
  }

  // Compute the global nanmin across all chunks for the NaN replacement.
  // After the `±inf` rejection above, `is_finite()` partitions entries
  // into {finite, NaN}, matching numpy's `nanmin` semantics on the
  // production path.
  let mut nanmin = f64::INFINITY;
  let mut any_finite = false;
  for chunk in chunks {
    for &v in chunk.iter() {
      if v.is_finite() {
        any_finite = true;
        if v < nanmin {
          nanmin = v;
        }
      }
    }
  }
  if !any_finite {
    return Err(Error::NonFinite(
      "soft_clusters has no finite entries; cannot compute nanmin replacement",
    ));
  }

  let mut out = Vec::with_capacity(chunks.len());
  for chunk in chunks {
    out.push(assign_one(chunk, num_speakers, num_clusters, nanmin)?);
  }
  Ok(out)
}

/// NaN-only `np.nan_to_num` cleanup: replace `NaN` with `nanmin`. The
/// `±inf` cases are rejected upstream by `constrained_argmax`, so this
/// function is only ever called on `{finite, NaN}` inputs and always
/// returns a finite value.
#[inline]
fn clean(v: f64, nanmin: f64) -> f64 {
  if v.is_nan() { nanmin } else { v }
}

fn assign_one(
  chunk: &DMatrix<f64>,
  num_speakers: usize,
  num_clusters: usize,
  nanmin: f64,
) -> Result<Vec<i32>, Error> {
  let mut assignment = vec![UNMATCHED; num_speakers];

  if num_speakers <= num_clusters {
    // Direct path: rows = speakers, cols = clusters.
    let mut data = Vec::with_capacity(num_speakers * num_clusters);
    for s in 0..num_speakers {
      for k in 0..num_clusters {
        data.push(NotNan::new(clean(chunk[(s, k)], nanmin)).expect("clean() yields finite f64"));
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
        data.push(NotNan::new(clean(chunk[(s, k)], nanmin)).expect("clean() yields finite f64"));
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
