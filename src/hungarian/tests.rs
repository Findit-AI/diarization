//! Model-free unit tests for `dia::hungarian`.
//!
//! Heavy parity against pyannote's captured `hard_clusters` lives in
//! `src/hungarian/parity_tests.rs`. This module covers smaller invariants
//! that should hold for any input.

use crate::hungarian::{Error, UNMATCHED, constrained_argmax};
use nalgebra::DMatrix;

/// Run a single chunk through the batched API. Most unit tests work on
/// one chunk at a time; this wrapper avoids repeating the slice + index
/// boilerplate.
fn one(cost: DMatrix<f64>) -> Result<Vec<i32>, Error> {
  constrained_argmax(&[cost]).map(|mut v| v.remove(0))
}

#[test]
fn rejects_empty_chunks() {
  let result = constrained_argmax(&[]);
  assert!(matches!(result, Err(Error::Shape(_))), "got {result:?}");
}

#[test]
fn rejects_empty_speakers() {
  let cost = DMatrix::<f64>::zeros(0, 3);
  let result = one(cost);
  assert!(matches!(result, Err(Error::Shape(_))), "got {result:?}");
}

#[test]
fn rejects_empty_clusters() {
  let cost = DMatrix::<f64>::zeros(3, 0);
  let result = one(cost);
  assert!(matches!(result, Err(Error::Shape(_))), "got {result:?}");
}

#[test]
fn rejects_chunks_with_different_shapes() {
  let a = DMatrix::<f64>::from_element(2, 2, 0.5);
  let b = DMatrix::<f64>::from_element(3, 2, 0.5);
  let result = constrained_argmax(&[a, b]);
  assert!(matches!(result, Err(Error::Shape(_))), "got {result:?}");
}

/// Square 2x2 — direct kuhn_munkres path. Diagonal dominates.
#[test]
fn square_2x2_picks_diagonal_when_diagonal_dominates() {
  let cost = DMatrix::<f64>::from_row_slice(2, 2, &[0.9, 0.1, 0.2, 0.8]);
  let assign = one(cost).expect("constrained_argmax");
  assert_eq!(assign, vec![0, 1]);
}

/// Square 2x2 — anti-diagonal dominates. Catches a greedy "row max" bug.
#[test]
fn square_2x2_picks_anti_diagonal_when_off_diagonal_dominates() {
  let cost = DMatrix::<f64>::from_row_slice(2, 2, &[0.2, 0.9, 0.8, 0.1]);
  let assign = one(cost).expect("constrained_argmax");
  assert_eq!(assign, vec![1, 0]);
}

/// Tall (S < K): 2 speakers, 3 clusters. Both speakers must be matched
/// to distinct clusters; the unused cluster index is just dropped.
#[test]
fn tall_2x3_assigns_both_speakers_to_distinct_clusters() {
  let cost = DMatrix::<f64>::from_row_slice(2, 3, &[0.1, 0.5, 1.0, 0.9, 0.4, 0.3]);
  let assign = one(cost).expect("constrained_argmax");
  assert_eq!(assign, vec![2, 0]);
  assert!(!assign.contains(&UNMATCHED));
}

/// Wide (S > K): 3 speakers, 2 clusters — captured-fixture shape.
/// Exercises the transpose path. Two speakers matched, one UNMATCHED.
#[test]
fn wide_3x2_leaves_one_speaker_unmatched() {
  let cost = DMatrix::<f64>::from_row_slice(
    3,
    2,
    &[0.95, 0.05, 0.05, 0.95, 0.10, 0.10],
  );
  let assign = one(cost).expect("constrained_argmax");
  assert_eq!(assign, vec![0, 1, UNMATCHED]);
}

/// Wide (S > K) where the optimal assignment leaves a *non-weakest*
/// speaker unmatched. Speaker 0 has cell 0.95 in cluster 0, but assigning
/// {2→0 (0.99), 1→1 (0.95)} sums to 1.94 > {0→0 (0.95), 1→1 (0.95)} = 1.90.
/// Catches a "leave the lowest-row speaker unmatched" greedy bug.
#[test]
fn wide_3x2_optimal_unmatches_non_weakest_speaker() {
  let cost = DMatrix::<f64>::from_row_slice(
    3,
    2,
    &[0.95, 0.10, 0.05, 0.95, 0.99, 0.10],
  );
  let assign = one(cost).expect("constrained_argmax");
  assert_eq!(assign, vec![UNMATCHED, 1, 0]);
}

/// Distinct-cluster invariant: every matched assignment uses a different
/// cluster index. Holds for square, tall, and wide shapes.
#[test]
fn matched_speakers_are_assigned_distinct_clusters() {
  let cost = DMatrix::<f64>::from_fn(4, 4, |i, j| ((i * 7 + j * 13) % 17) as f64 * 0.1);
  let assign = one(cost).expect("constrained_argmax");
  let mut used = std::collections::HashSet::new();
  for &k in &assign {
    if k != UNMATCHED {
      assert!(used.insert(k), "cluster {k} assigned twice in {assign:?}");
    }
  }
  assert!(!assign.contains(&UNMATCHED));
}

#[test]
fn single_speaker_single_cluster() {
  let cost = DMatrix::<f64>::from_element(1, 1, 0.42);
  let assign = one(cost).expect("constrained_argmax");
  assert_eq!(assign, vec![0]);
}

#[test]
fn single_speaker_multiple_clusters_picks_max() {
  let cost = DMatrix::<f64>::from_row_slice(1, 4, &[0.1, 0.5, 0.9, 0.3]);
  let assign = one(cost).expect("constrained_argmax");
  assert_eq!(assign, vec![2]);
}

#[test]
fn single_cluster_multiple_speakers_matches_max_speaker() {
  let cost = DMatrix::<f64>::from_row_slice(3, 1, &[0.1, 0.9, 0.5]);
  let assign = one(cost).expect("constrained_argmax");
  assert_eq!(assign, vec![UNMATCHED, 0, UNMATCHED]);
}

#[test]
fn deterministic_on_repeated_calls() {
  let cost = DMatrix::<f64>::from_fn(5, 4, |i, j| ((i + 2 * j) as f64 * 0.13).cos());
  let a = one(cost.clone()).expect("a");
  let b = one(cost).expect("b");
  assert_eq!(a, b);
}

// ── nan_to_num semantics (Codex review MEDIUM round 1 of Phase 3) ─
//
// Pyannote runs `np.nan_to_num(soft_clusters, nan=np.nanmin(soft_clusters))`
// before per-chunk matching. The Rust port replicates this:
// NaN → global nanmin across all chunks, +inf → f64::MAX, -inf → f64::MIN.

/// NaN entries in a single chunk are replaced with the chunk's own min.
/// The replacement must produce a valid optimal matching, not error out.
#[test]
fn nan_in_single_chunk_replaced_with_min() {
  // 2x2 with NaN in (1, 0). Other entries: 0.9, 0.5, NaN, 0.8.
  // nanmin = 0.5. After replacement: 0.9, 0.5, 0.5, 0.8.
  // Optimal: speaker 0 → cluster 0 (0.9), speaker 1 → cluster 1 (0.8).
  let mut cost = DMatrix::<f64>::from_row_slice(2, 2, &[0.9, 0.5, 0.0, 0.8]);
  cost[(1, 0)] = f64::NAN;
  let assign = one(cost).expect("constrained_argmax with NaN must replace, not error");
  assert_eq!(assign, vec![0, 1]);
}

/// NaN replacement uses the *global* min across all chunks, not the per-
/// chunk min — this is the pyannote contract Codex called out.
///
/// Setup: chunk 0 = [[0.9, 0.5], [0.7, NaN]]. Chunk 1 contains -5.0.
/// - Local nanmin (0.5) replacement of chunk 0's NaN:
///   {s0→c0 (0.9), s1→c1 (0.5)} = 1.4 vs {s0→c1 (0.5), s1→c0 (0.7)} = 1.2
///   → optimal pairs s0→c0, s1→c1 (assignment vec![0, 1]).
/// - Global nanmin (-5.0) replacement of chunk 0's NaN:
///   {s0→c0 (0.9), s1→c1 (-5.0)} = -4.1 vs {s0→c1 (0.5), s1→c0 (0.7)} = 1.2
///   → optimal pairs s0→c1, s1→c0 (assignment vec![1, 0]).
///
/// Different assignments confirm global vs local replacement behavior.
#[test]
fn nan_replacement_uses_global_nanmin_across_chunks() {
  let mut chunk_a = DMatrix::<f64>::from_row_slice(2, 2, &[0.9, 0.5, 0.7, 0.0]);
  chunk_a[(1, 1)] = f64::NAN;
  let chunk_b = DMatrix::<f64>::from_row_slice(2, 2, &[0.0, 0.0, 0.0, -5.0]);

  let assigns = constrained_argmax(&[chunk_a, chunk_b]).expect("constrained_argmax");
  assert_eq!(assigns.len(), 2);
  // Global-min replacement (-5.0) drives the chunk-0 optimal to anti-
  // diagonal: speaker 0 → cluster 1, speaker 1 → cluster 0.
  assert_eq!(assigns[0], vec![1, 0]);
}

/// `+inf` → `f64::MAX`, `-inf` → `f64::MIN` (numpy's `posinf`/`neginf`
/// defaults). With these substitutions a `+inf` cell is the strongest
/// possible match and `-inf` is the weakest — the assignment must
/// reflect that.
#[test]
fn pos_inf_replaced_with_f64_max_drives_assignment() {
  // 2x2: speaker 0 → cluster 1 has +inf, all others modest. Optimal
  // pairs speaker 0 with cluster 1 (the +inf → f64::MAX dominates).
  let mut cost = DMatrix::<f64>::from_row_slice(2, 2, &[0.9, 0.0, 0.5, 0.6]);
  cost[(0, 1)] = f64::INFINITY;
  let assign = one(cost).expect("constrained_argmax");
  assert_eq!(assign, vec![1, 0]);
}

#[test]
fn neg_inf_replaced_with_f64_min_avoids_assignment() {
  // 2x2: speaker 0 → cluster 0 has -inf, so optimal must avoid it.
  let mut cost = DMatrix::<f64>::from_row_slice(2, 2, &[0.0, 0.5, 0.7, 0.0]);
  cost[(0, 0)] = f64::NEG_INFINITY;
  let assign = one(cost).expect("constrained_argmax");
  assert_eq!(assign, vec![1, 0]);
}

/// All entries non-finite → there's no value to use as the nanmin
/// replacement. Pyannote degenerates here too (`np.nanmin` of an
/// all-NaN array returns NaN, and `nan_to_num(x, nan=NaN)` is a no-op).
/// The Rust port surfaces this as `Error::NonFinite` rather than
/// silently producing a NaN-poisoned assignment.
#[test]
fn rejects_when_all_entries_non_finite() {
  let cost = DMatrix::<f64>::from_element(2, 2, f64::NAN);
  let result = one(cost);
  assert!(
    matches!(result, Err(Error::NonFinite(_))),
    "got {result:?}"
  );
}
