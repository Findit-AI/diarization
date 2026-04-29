//! Model-free unit tests for `dia::hungarian`.
//!
//! Heavy parity against pyannote's captured `hard_clusters` lives in
//! `src/hungarian/parity_tests.rs`. This module covers smaller invariants
//! that should hold for any input.

use crate::hungarian::{Error, UNMATCHED, constrained_argmax};
use nalgebra::DMatrix;

#[test]
fn rejects_empty_speakers() {
  let cost = DMatrix::<f64>::zeros(0, 3);
  let result = constrained_argmax(&cost);
  assert!(matches!(result, Err(Error::Shape(_))), "got {result:?}");
}

#[test]
fn rejects_empty_clusters() {
  let cost = DMatrix::<f64>::zeros(3, 0);
  let result = constrained_argmax(&cost);
  assert!(matches!(result, Err(Error::Shape(_))), "got {result:?}");
}

#[test]
fn rejects_nan_entry() {
  let mut cost = DMatrix::<f64>::from_element(2, 2, 0.5);
  cost[(0, 1)] = f64::NAN;
  let result = constrained_argmax(&cost);
  assert!(
    matches!(result, Err(Error::NonFinite(_))),
    "got {result:?}"
  );
}

#[test]
fn rejects_pos_inf_entry() {
  let mut cost = DMatrix::<f64>::from_element(2, 2, 0.5);
  cost[(1, 0)] = f64::INFINITY;
  let result = constrained_argmax(&cost);
  assert!(
    matches!(result, Err(Error::NonFinite(_))),
    "got {result:?}"
  );
}

#[test]
fn rejects_neg_inf_entry() {
  let mut cost = DMatrix::<f64>::from_element(2, 2, 0.5);
  cost[(0, 0)] = f64::NEG_INFINITY;
  let result = constrained_argmax(&cost);
  assert!(
    matches!(result, Err(Error::NonFinite(_))),
    "got {result:?}"
  );
}

/// Square 2x2 — direct kuhn_munkres path. Diagonal dominates.
#[test]
fn square_2x2_picks_diagonal_when_diagonal_dominates() {
  let cost = DMatrix::<f64>::from_row_slice(2, 2, &[0.9, 0.1, 0.2, 0.8]);
  let assign = constrained_argmax(&cost).expect("constrained_argmax");
  assert_eq!(assign, vec![0, 1]);
}

/// Square 2x2 — anti-diagonal dominates. Catches a greedy "row max" bug.
#[test]
fn square_2x2_picks_anti_diagonal_when_off_diagonal_dominates() {
  let cost = DMatrix::<f64>::from_row_slice(2, 2, &[0.2, 0.9, 0.8, 0.1]);
  let assign = constrained_argmax(&cost).expect("constrained_argmax");
  assert_eq!(assign, vec![1, 0]);
}

/// Tall (S < K): 2 speakers, 3 clusters. Both speakers must be matched
/// to distinct clusters; the unused cluster index is just dropped.
#[test]
fn tall_2x3_assigns_both_speakers_to_distinct_clusters() {
  let cost = DMatrix::<f64>::from_row_slice(2, 3, &[0.1, 0.5, 1.0, 0.9, 0.4, 0.3]);
  let assign = constrained_argmax(&cost).expect("constrained_argmax");
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
  let assign = constrained_argmax(&cost).expect("constrained_argmax");
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
  let assign = constrained_argmax(&cost).expect("constrained_argmax");
  assert_eq!(assign, vec![UNMATCHED, 1, 0]);
}

/// Distinct-cluster invariant: every matched assignment uses a different
/// cluster index. Holds for square, tall, and wide shapes.
#[test]
fn matched_speakers_are_assigned_distinct_clusters() {
  let cost = DMatrix::<f64>::from_fn(4, 4, |i, j| ((i * 7 + j * 13) % 17) as f64 * 0.1);
  let assign = constrained_argmax(&cost).expect("constrained_argmax");
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
  let assign = constrained_argmax(&cost).expect("constrained_argmax");
  assert_eq!(assign, vec![0]);
}

#[test]
fn single_speaker_multiple_clusters_picks_max() {
  let cost = DMatrix::<f64>::from_row_slice(1, 4, &[0.1, 0.5, 0.9, 0.3]);
  let assign = constrained_argmax(&cost).expect("constrained_argmax");
  assert_eq!(assign, vec![2]);
}

#[test]
fn single_cluster_multiple_speakers_matches_max_speaker() {
  let cost = DMatrix::<f64>::from_row_slice(3, 1, &[0.1, 0.9, 0.5]);
  let assign = constrained_argmax(&cost).expect("constrained_argmax");
  assert_eq!(assign, vec![UNMATCHED, 0, UNMATCHED]);
}

#[test]
fn deterministic_on_repeated_calls() {
  let cost = DMatrix::<f64>::from_fn(5, 4, |i, j| ((i + 2 * j) as f64 * 0.13).cos());
  let a = constrained_argmax(&cost).expect("a");
  let b = constrained_argmax(&cost).expect("b");
  assert_eq!(a, b);
}
