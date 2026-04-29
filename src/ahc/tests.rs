//! Model-free unit tests for `dia::ahc`.
//!
//! Heavy parity against pyannote's captured `ahc_init_labels.npy` lives
//! in `src/ahc/parity_tests.rs`. This module covers smaller invariants
//! that should hold for any input.

use crate::ahc::{Error, ahc_init};
use nalgebra::DMatrix;

#[test]
fn rejects_empty_embeddings() {
  let m = DMatrix::<f64>::zeros(0, 4);
  assert!(matches!(ahc_init(&m, 0.5), Err(Error::Shape(_))));
}

#[test]
fn rejects_zero_dimension() {
  let m = DMatrix::<f64>::zeros(3, 0);
  assert!(matches!(ahc_init(&m, 0.5), Err(Error::Shape(_))));
}

#[test]
fn rejects_non_positive_threshold() {
  let m = DMatrix::<f64>::from_element(3, 4, 1.0);
  assert!(matches!(ahc_init(&m, 0.0), Err(Error::Shape(_))));
  assert!(matches!(ahc_init(&m, -0.1), Err(Error::Shape(_))));
}

#[test]
fn rejects_non_finite_threshold() {
  let m = DMatrix::<f64>::from_element(3, 4, 1.0);
  assert!(matches!(ahc_init(&m, f64::NAN), Err(Error::Shape(_))));
  assert!(matches!(ahc_init(&m, f64::INFINITY), Err(Error::Shape(_))));
}

#[test]
fn rejects_nan_in_embedding() {
  let mut m = DMatrix::<f64>::from_element(3, 4, 1.0);
  m[(1, 2)] = f64::NAN;
  assert!(matches!(ahc_init(&m, 0.5), Err(Error::NonFinite(_))));
}

#[test]
fn rejects_inf_in_embedding() {
  let mut m = DMatrix::<f64>::from_element(3, 4, 1.0);
  m[(0, 0)] = f64::INFINITY;
  assert!(matches!(ahc_init(&m, 0.5), Err(Error::NonFinite(_))));
}

#[test]
fn rejects_zero_norm_row() {
  let mut m = DMatrix::<f64>::from_element(3, 4, 1.0);
  for c in 0..4 {
    m[(1, c)] = 0.0;
  }
  assert!(matches!(ahc_init(&m, 0.5), Err(Error::Shape(_))));
}

/// Single row → single cluster (matches pyannote's `< 2` short-circuit).
#[test]
fn single_row_returns_single_cluster() {
  let m = DMatrix::<f64>::from_row_slice(1, 3, &[1.0, 0.0, 0.0]);
  let labels = ahc_init(&m, 0.5).expect("ahc_init");
  assert_eq!(labels, vec![0]);
}

/// Two near-identical rows + a far row → two clusters when threshold
/// admits the close pair but not the far one. The test mirrors scipy's
/// behavior that we hand-verified during development.
///
/// Rows (after L2 normalization):
/// - Row 0 ≈ (1, 0, 0)
/// - Row 1 ≈ (0.99, 0.01, 0)  → close to Row 0
/// - Row 2 ≈ (0, 1, 0)         → orthogonal
///
/// Distances after L2 norm: d(0,1) ≈ 0.014, d(0,2) ≈ 1.414, d(1,2) ≈ 1.404.
/// At threshold = 0.5: only the (0,1) pair merges → labels `[0, 0, 1]`.
#[test]
fn merges_close_pair_separates_far_row() {
  let m = DMatrix::<f64>::from_row_slice(3, 3, &[1.0, 0.0, 0.0, 100.0, 1.0, 0.0, 0.0, 1.0, 0.0]);
  let labels = ahc_init(&m, 0.5).expect("ahc_init");
  assert_eq!(labels, vec![0, 0, 1]);
}

/// All identical rows (after normalization) → single cluster regardless
/// of threshold. Distances are zero, so any positive threshold merges all.
#[test]
fn all_identical_normed_rows_collapse_to_one_cluster() {
  let m = DMatrix::<f64>::from_row_slice(
    4,
    2,
    &[
      1.0, 0.0, 2.0, 0.0, // same direction → same after L2 norm
      3.0, 0.0, 0.5, 0.0,
    ],
  );
  let labels = ahc_init(&m, 0.001).expect("ahc_init");
  assert_eq!(labels, vec![0, 0, 0, 0]);
}

/// Threshold below all merge distances → every row is its own cluster.
#[test]
fn tiny_threshold_keeps_every_row_isolated() {
  // Three orthogonal directions; pairwise distance after L2 norm ≈ √2 ≈ 1.414.
  let m = DMatrix::<f64>::from_row_slice(3, 3, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
  let labels = ahc_init(&m, 0.1).expect("ahc_init");
  // Encounter-order labels — each leaf is its own cluster, labelled in
  // its first-encountered order.
  assert_eq!(labels, vec![0, 1, 2]);
}

/// Labels must be encounter-order contiguous `0..k` (this is the
/// `np.unique(return_inverse=True)` post-processing pyannote does).
#[test]
fn labels_are_encounter_order_contiguous() {
  // Six rows: two pairs that should merge, plus two singletons that
  // shouldn't. Specific arrangement: pair A (rows 0, 3), pair B (rows
  // 1, 4), singleton (row 2), singleton (row 5).
  let m = DMatrix::<f64>::from_row_slice(
    6,
    3,
    &[
      1.0, 0.0, 0.0, // row 0: pair A
      0.0, 1.0, 0.0, // row 1: pair B
      0.0, 0.0, 1.0, // row 2: singleton
      1.001, 0.0, 0.0, // row 3: pair A (close to row 0 after norm)
      0.0, 1.001, 0.0, // row 4: pair B (close to row 1 after norm)
      1.0, 1.0, 1.0, // row 5: singleton
    ],
  );
  let labels = ahc_init(&m, 0.1).expect("ahc_init");
  // Encounter order of labels: row 0 → 0, row 1 → 1, row 2 → 2,
  // row 3 → 0 (same cluster as row 0), row 4 → 1, row 5 → 3.
  assert_eq!(labels, vec![0, 1, 2, 0, 1, 3]);

  // Sanity: labels are contiguous 0..k where k = number of distinct.
  let max = *labels.iter().max().unwrap();
  let mut seen = vec![false; max + 1];
  for &l in &labels {
    seen[l] = true;
  }
  assert!(seen.iter().all(|&s| s), "labels {labels:?} not contiguous");
}

/// Determinism: same input → identical output.
#[test]
fn deterministic_on_repeated_calls() {
  let m = DMatrix::<f64>::from_fn(8, 4, |i, j| ((i * 7 + j * 13) as f64 * 0.1).sin() + 1.0);
  let a = ahc_init(&m, 0.5).expect("a");
  let b = ahc_init(&m, 0.5).expect("b");
  assert_eq!(a, b);
}
