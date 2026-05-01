//! Model-free unit tests for `diarization::cluster::ahc`.
//!
//! Heavy parity against pyannote's captured `ahc_init_labels.npy` lives
//! in `src/ahc/parity_tests.rs`. This module covers smaller invariants
//! that should hold for any input.

use crate::cluster::ahc::{Error, ahc_init};
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

// ── Centroid-linkage inversion (Codex review HIGH round 1 of Phase 4) ─
//
// Centroid linkage (the method pyannote uses) does not produce
// monotonic dendrograms in general — a parent merge can have a
// *lower* dissimilarity than one of its children. Scipy's
// `fcluster(criterion="distance")` handles this by computing the
// max merge dissimilarity in each subtree before cutting, so a
// flat cluster's pairwise cophenetic distances are all `≤ t`.
//
// The regression test below uses a 4-point unit-vector configuration
// where:
// - d(0, 1) = 0.65          (above threshold 0.6 → step 0)
// - d(2, {0, 1}) = 0.574    (BELOW threshold via Lance-Williams)
// - d(3, *) ≈ 1.89          (far above)
//
// The dendrogram has an inversion at step 1 (lower than step 0).
// A naive bottom-up "union when step.dist ≤ t" walk would merge
// {0, 1, 2} into one cluster (matching root step 1's low dist), but
// scipy splits all three because the {0, 1} subtree's max internal
// merge (0.65) is still above threshold. The Rust port must agree
// with scipy.

/// Pyannote's centroid-linkage flow can produce a non-monotonic
/// dendrogram. The fcluster cut must use the *max* dissimilarity in
/// each subtree, not just the root's `step.dissimilarity`. This test
/// constructs a deterministic 4-point input that triggers the
/// inversion at threshold 0.6 — same partition as scipy.
#[test]
fn centroid_linkage_inversion_matches_scipy() {
  // 4 unit vectors in 3D. d(0,1)=0.65 above threshold, but step 1
  // (merging point 2 with {0,1}) inverts to dist=0.574, BELOW threshold.
  let alpha = 2.0_f64 * (0.65_f64 / 2.0).asin();
  let p0 = (1.0_f64, 0.0_f64, 0.0_f64);
  let p1 = (alpha.cos(), alpha.sin(), 0.0_f64);
  // p2 chosen so |p2-p0| = |p2-p1| = 0.66, |p2| = 1.
  let cdota = 1.0 - 0.66_f64.powi(2) / 2.0;
  let cy = (cdota - p1.0 * cdota) / p1.1;
  let cz = (1.0_f64 - cdota * cdota - cy * cy).sqrt();
  let p2 = (cdota, cy, cz);
  let p3 = (-1.0_f64, 0.0_f64, 0.0_f64);

  let m = DMatrix::<f64>::from_row_slice(
    4,
    3,
    &[
      p0.0, p0.1, p0.2, p1.0, p1.1, p1.2, p2.0, p2.1, p2.2, p3.0, p3.1, p3.2,
    ],
  );

  let labels = ahc_init(&m, 0.6).expect("ahc_init");

  // Scipy on this dendrogram:
  //   step 0 (merge 0, 1): d=0.65 > 0.6
  //   step 1 (merge 2, {0,1}): d=0.574 ≤ 0.6 BUT subtree's max = 0.65 > 0.6
  //   step 2 (merge 3, ...): d=1.89 > 0.6
  // → no merges accepted; each leaf is its own cluster.
  // Encounter-order labels: [0, 1, 2, 3].
  assert_eq!(
    labels,
    vec![0, 1, 2, 3],
    "inversion case must match scipy: subtree max > threshold means split"
  );
}

/// Determinism: same input → identical output.
#[test]
fn deterministic_on_repeated_calls() {
  let m = DMatrix::<f64>::from_fn(8, 4, |i, j| ((i * 7 + j * 13) as f64 * 0.1).sin() + 1.0);
  let a = ahc_init(&m, 0.5).expect("a");
  let b = ahc_init(&m, 0.5).expect("b");
  assert_eq!(a, b);
}

/// Codex adversarial review MEDIUM (this commit). The SIMD pdist
/// path differs from scalar by O(1e-12) relative on well-conditioned
/// inputs (FMA + parallel-lane reduction). AHC cuts at hard
/// `<= threshold` — in principle a distance landing within ~1e-12 of
/// `threshold` could flip a merge. These tests empirically check the
/// risk for production-shaped inputs.
mod simd_partition_stability {
  use crate::cluster::ahc::algo::ahc_init_with_simd;
  use nalgebra::DMatrix;
  use rand::{SeedableRng, prelude::*};
  use rand_chacha::ChaCha20Rng;

  /// 50 random seeds × random embeddings (N=20, D=128). Both backends
  /// must produce *identical* partitions at the production threshold.
  /// If this ever fails, the failing seed lets us reproduce and decide
  /// whether to revert AHC pdist to scalar.
  #[test]
  fn random_inputs_agree_at_pyannote_community_threshold() {
    const SEEDS: u64 = 50;
    const N: usize = 20;
    const D: usize = 128;
    const PYANNOTE_COMMUNITY_THRESHOLD: f64 = 0.6;
    for seed in 0..SEEDS {
      let mut rng = ChaCha20Rng::seed_from_u64(seed);
      let m = DMatrix::<f64>::from_fn(N, D, |_, _| rng.random::<f64>() * 2.0 - 1.0);
      let scalar = ahc_init_with_simd(&m, PYANNOTE_COMMUNITY_THRESHOLD, false).expect("scalar AHC");
      let simd = ahc_init_with_simd(&m, PYANNOTE_COMMUNITY_THRESHOLD, true).expect("SIMD AHC");
      assert_eq!(
        scalar, simd,
        "AHC partition diverged on seed {seed}: scalar={scalar:?}, simd={simd:?}"
      );
    }
  }

  /// Threshold-adjacent boundary, dense input. Random (D-dim, every
  /// dim non-zero) base vectors, then perturbed copies whose pairwise
  /// distance is engineered to land near the threshold by scaling the
  /// perturbation. Dense inputs exercise the SIMD parallel-lane
  /// reduction (the simple-orthogonal construction with only 2
  /// non-zero dims/row would compute bit-identically across backends).
  ///
  /// Sweeps offsets across {±1e-10, ±1e-12, ±1e-13} from the
  /// pyannote-community threshold of 0.6. If any perturbed pair flips
  /// the AHC merge between scalar and SIMD, the test fails with the
  /// offending offset.
  #[test]
  fn boundary_constructed_inputs_agree() {
    const THRESHOLD: f64 = 0.6;
    const D: usize = 128;
    let offsets: [f64; 6] = [-1.0e-10, -1.0e-12, -1.0e-13, 1.0e-13, 1.0e-12, 1.0e-10];
    for (k, &offset) in offsets.iter().enumerate() {
      let target_dist = THRESHOLD + offset;
      let mut rng = ChaCha20Rng::seed_from_u64(0xb0 ^ k as u64);
      // Random dense unit base vector.
      let base: Vec<f64> = {
        let raw: Vec<f64> = (0..D).map(|_| rng.random::<f64>() * 2.0 - 1.0).collect();
        let norm: f64 = raw.iter().map(|v| v * v).sum::<f64>().sqrt();
        raw.iter().map(|v| v / norm).collect()
      };
      // Random orthogonal perturbation direction (Gram-Schmidt against base).
      let perturb: Vec<f64> = {
        let raw: Vec<f64> = (0..D).map(|_| rng.random::<f64>() * 2.0 - 1.0).collect();
        let proj: f64 = raw.iter().zip(&base).map(|(r, b)| r * b).sum();
        let mut orth: Vec<f64> = raw.iter().zip(&base).map(|(r, b)| r - proj * b).collect();
        let norm: f64 = orth.iter().map(|v| v * v).sum::<f64>().sqrt();
        for v in orth.iter_mut() {
          *v /= norm;
        }
        orth
      };
      // For unit base + unit orthogonal perturbation scaled by alpha,
      // the distance ||base - (cos·base + sin·perturb)|| where
      // cos = 1 - target²/2 lands at target. Build 6 points: 3 pairs.
      let cos_theta = 1.0 - target_dist * target_dist / 2.0;
      let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
      let n = 6usize;
      let mut data = vec![0.0_f64; n * D];
      for pair in 0..3 {
        let i = 2 * pair;
        let j = i + 1;
        // Rotate the perturbation per pair to avoid all 3 pairs being
        // colinear (would degenerate AHC's centroid linkage).
        let phase = pair as f64 * 0.1;
        let cph = phase.cos();
        let sph = phase.sin();
        for d in 0..D {
          let bd = base[d] * cph + perturb[d] * sph;
          let pd = perturb[d] * cph - base[d] * sph;
          data[i * D + d] = bd;
          data[j * D + d] = cos_theta * bd + sin_theta * pd;
        }
      }
      let m = DMatrix::<f64>::from_row_slice(n, D, &data);
      let scalar = ahc_init_with_simd(&m, THRESHOLD, false).expect("scalar AHC");
      let simd = ahc_init_with_simd(&m, THRESHOLD, true).expect("SIMD AHC");
      assert_eq!(
        scalar, simd,
        "AHC partition diverged at target_dist={target_dist} (offset={offset:e}): \
         scalar={scalar:?}, simd={simd:?}"
      );
    }
  }
}
