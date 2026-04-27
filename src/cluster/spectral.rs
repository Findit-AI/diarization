//! Spectral clustering. Spec §5.5.
//!
//! Pipeline: cosine affinity (ReLU-clamped) → degree precondition →
//! normalized Laplacian L_sym = I - D^{-1/2} A D^{-1/2} →
//! eigendecomposition (Task 19) → eigengap-K (Task 19) → row-normalized
//! eigenvector matrix (Task 21) → K-means++ seeding (Task 20) → Lloyd
//! refinement (Task 21) → labels.
//!
//! This file currently implements steps 1-3 (affinity + degree + Laplacian).
//! `cluster` is a `todo!()` placeholder; Tasks 19-21 wire in the rest.

use crate::{
  cluster::{
    Error,
    options::{MAX_AUTO_SPEAKERS, OfflineClusterOptions},
  },
  embed::{Embedding, NORM_EPSILON},
};
use nalgebra::{DMatrix, DVector};

/// Cluster `embeddings` via spectral clustering. Caller guarantees
/// `embeddings.len() >= 3` (the N<=2 fast path lives in `cluster_offline`).
///
/// Currently a partial implementation: only steps 1-3 are wired up.
/// The eigendecomposition + K-detection + K-means stages arrive in
/// Tasks 19-21.
pub(crate) fn cluster(
  embeddings: &[Embedding],
  opts: &OfflineClusterOptions,
) -> Result<Vec<u64>, Error> {
  debug_assert!(embeddings.len() >= 3, "fast path covers N <= 2");
  // Pipeline overview (full impl arrives in Tasks 19-21):
  //   let a = build_affinity(embeddings);
  //   let degrees = compute_degrees(&a)?;
  //   let l = normalized_laplacian(&a, &degrees);
  //   ... eigendecomp, K-detection, row-normalize, K-means++, Lloyd ...
  let _ = embeddings;
  let _ = opts;
  todo!("Tasks 19-21 fill in the remaining spectral pipeline")
}

/// Build the N x N affinity matrix `A[i][j] = max(0, e_i · e_j)`; `A[i][i] = 0`.
///
/// Affinity is f64 for numerical stability through the eigendecomposition.
/// ReLU clamp matches spec §5.5 step 1 (rev-3).
///
/// Relies on the [`Embedding`] L2-normalized invariant: dot product equals
/// cosine similarity. `Embedding::similarity` enforces this.
#[allow(dead_code)] // used in tests; wired into cluster() in Tasks 19-21
pub(crate) fn build_affinity(embeddings: &[Embedding]) -> DMatrix<f64> {
  let n = embeddings.len();
  let mut a = DMatrix::<f64>::zeros(n, n);
  for (i, ei) in embeddings.iter().enumerate() {
    for (offset, ej) in embeddings.iter().skip(i + 1).enumerate() {
      let j = i + 1 + offset;
      let sim = ei.similarity(ej).max(0.0) as f64;
      a[(i, j)] = sim;
      a[(j, i)] = sim;
    }
    // a[(i, i)] = 0 by zeros() init.
  }
  a
}

/// Degree vector `D_ii = sum_j A_ij`. Returns
/// [`Error::AllDissimilar`] if any `D_ii < NORM_EPSILON`
/// (rev-3 isolated-node precondition; covers both the all-zero
/// affinity case and individually-isolated nodes).
///
/// Real embed-model outputs are L2-normalized and cannot be
/// degenerate, so hitting this error is almost certainly a
/// caller-fabricated input. See spec §4.3.
#[allow(dead_code)] // used in tests; wired into cluster() in Tasks 19-21
pub(crate) fn compute_degrees(a: &DMatrix<f64>) -> Result<Vec<f64>, Error> {
  let eps = NORM_EPSILON as f64;
  let degrees: Vec<f64> = a.row_iter().map(|row| row.sum()).collect();
  if degrees.iter().any(|&d| d < eps) {
    return Err(Error::AllDissimilar);
  }
  Ok(degrees)
}

/// Normalized symmetric Laplacian `L_sym = I - D^{-1/2} A D^{-1/2}`.
/// Caller guarantees `D_ii >= NORM_EPSILON` for all i (enforced by
/// [`compute_degrees`]).
#[allow(dead_code)] // used in tests; wired into cluster() in Tasks 19-21
pub(crate) fn normalized_laplacian(a: &DMatrix<f64>, d: &[f64]) -> DMatrix<f64> {
  let n = a.nrows();
  // D^{-1/2} as a diagonal matrix.
  let inv_sqrt = DVector::from_iterator(n, d.iter().map(|&di| 1.0 / di.sqrt()));
  let inv_sqrt_diag = DMatrix::from_diagonal(&inv_sqrt);
  // L_sym = I - D^{-1/2} A D^{-1/2}.
  DMatrix::<f64>::identity(n, n) - &inv_sqrt_diag * a * &inv_sqrt_diag
}

/// Eigendecompose the symmetric Laplacian `L_sym` and return the eigenvalues
/// and matching eigenvectors sorted by ascending eigenvalue.
///
/// Returns `(eigenvalues, eigenvectors)` where:
/// - `eigenvalues[k]` is the k-th smallest eigenvalue of `L_sym` (ascending).
/// - `eigenvectors[(row, k)]` is the k-th eigenvector (column-major; aligned
///   with `eigenvalues[k]`).
///
/// Uses `nalgebra::SymmetricEigen`, which expects a real symmetric input —
/// `L_sym` qualifies by construction in [`normalized_laplacian`]. nalgebra
/// returns eigenvalues in implementation-defined order; this function sorts
/// them ascending and reorders the eigenvector columns to match.
///
/// Returns [`Error::EigendecompositionFailed`] if any eigenvalue is non-finite
/// (NaN or infinity), which signals a pathological / singular input matrix.
#[allow(dead_code)] // used in tests; wired into cluster() in Task 21
pub(crate) fn eigendecompose(l: DMatrix<f64>) -> Result<(Vec<f64>, DMatrix<f64>), Error> {
  let n = l.nrows();
  // L_sym is real symmetric; SymmetricEigen is the numerically stable choice.
  let sym = nalgebra::SymmetricEigen::new(l);

  // Detect numerical failure first.
  if sym.eigenvalues.iter().any(|v| !v.is_finite()) {
    return Err(Error::EigendecompositionFailed);
  }

  // Pair each eigenvalue with its original column index, sort ascending.
  let mut indexed: Vec<(f64, usize)> = sym
    .eigenvalues
    .iter()
    .copied()
    .enumerate()
    .map(|(i, v)| (v, i))
    .collect();
  indexed.sort_by(|a, b| a.0.total_cmp(&b.0));

  // Materialize sorted vectors into a fresh DMatrix.
  let mut sorted_vecs = DMatrix::<f64>::zeros(n, n);
  let mut sorted_vals = Vec::with_capacity(n);
  for (new_col, &(val, old_col)) in indexed.iter().enumerate() {
    sorted_vals.push(val);
    sorted_vecs.set_column(new_col, &sym.eigenvectors.column(old_col));
  }

  Ok((sorted_vals, sorted_vecs))
}

/// Choose K (number of clusters) via the eigengap heuristic, with a target
/// override.
///
/// - If `target_speakers = Some(k)`, returns `k` directly.
/// - Otherwise computes the largest gap `λ[k+1] − λ[k]` for k in
///   `[0, k_max)` where `k_max = min(N − 1, MAX_AUTO_SPEAKERS = 15)` (spec
///   §5.5 step 5; spec §4.3 line 697-698 caps the auto-detected count).
/// - Returns `K = argmax_k (λ[k+1] − λ[k]) + 1`, floored at 1.
///
/// `eigenvalues` must be sorted ascending (as produced by [`eigendecompose`]).
/// Indexing assumes `eigenvalues.len() == n`.
#[allow(dead_code)] // used in tests; wired into cluster() in Task 21
pub(crate) fn pick_k(eigenvalues: &[f64], n: usize, target_speakers: Option<u32>) -> usize {
  debug_assert_eq!(
    eigenvalues.len(),
    n,
    "pick_k: eigenvalues slice length must equal n"
  );
  if let Some(k) = target_speakers {
    return k as usize;
  }
  // k_max bounds: at most N-1 gaps exist, capped at MAX_AUTO_SPEAKERS.
  let k_max = (n.saturating_sub(1)).min(MAX_AUTO_SPEAKERS as usize);
  if k_max < 1 {
    return 1;
  }

  // Largest gap: argmax over windows of size 2 in the first k_max+1 entries.
  let (best_k, _) = eigenvalues
    .windows(2)
    .take(k_max)
    .enumerate()
    .map(|(k, w)| (k + 1, w[1] - w[0]))
    .max_by(|a, b| a.1.total_cmp(&b.1))
    .unwrap_or((1, 0.0));

  best_k.max(1)
}

use rand::{
  RngExt as _, SeedableRng,
  distr::{Distribution, Uniform},
};
use rand_chacha::ChaCha8Rng;

/// K-means++ seeding (Arthur & Vassilvitskii 2007) over the rows of `mat`
/// (`N` rows × `dim` columns). Returns the K initial centroid rows
/// (each is `dim`-dimensional).
///
/// Pinned to specific `rand` 0.10 call sites for byte-determinism per
/// spec §5.5 step 8 / §11.9. Task 2's keystream fixture enforces this
/// across rand patch versions:
/// - First centroid: `Uniform::new(0, N).unwrap().sample(&mut rng)`.
/// - Cumulative-mass crossing: `rng.random::<f64>()` (StandardUniform,
///   half-open `[0, 1)`), strict `>` against `t = u * S`.
/// - Step 2b (S == 0 → duplicates): linear-scan compacted `Vec<usize>` of
///   not-yet-chosen indices, then `Uniform::new(0, available.len())`.
/// - All min/sum reductions left-to-right in `f64`.
///
/// Caller invariants: `k >= 1`, `n >= k`, all `mat` rows finite. Caller
/// (Task 21) guarantees these via [`validate_offline_input`] + the
/// fast-path filter for N<=2.
#[allow(dead_code)] // used in tests; wired into cluster() in Task 21
pub(crate) fn kmeans_pp_seed(mat: &DMatrix<f64>, k: usize, seed: u64) -> Vec<Vec<f64>> {
  let n = mat.nrows();
  debug_assert!(k >= 1, "K must be >= 1");
  debug_assert!(n >= k, "N must be >= K");

  let mut rng = ChaCha8Rng::seed_from_u64(seed);

  // Step 1: pick first centroid uniformly.
  let i0: usize = Uniform::new(0usize, n).unwrap().sample(&mut rng);
  let mut centroids: Vec<Vec<f64>> = vec![row(mat, i0)];
  let mut chosen: Vec<usize> = vec![i0];

  // Step 2: for k = 1..K, weighted-by-D^2 sampling.
  while centroids.len() < k {
    // Step 2a: D[j] = min over chosen centroids of ||row_j - c||^2 (left-to-right).
    let d: Vec<f64> = (0..n)
      .map(|j| {
        centroids
          .iter()
          .map(|c| {
            c.iter()
              .enumerate()
              .map(|(x, &cx)| {
                let diff = mat[(j, x)] - cx;
                diff * diff
              })
              .sum::<f64>()
          })
          .fold(f64::INFINITY, |a, b| if b < a { b } else { a })
      })
      .collect();

    // Step 2b: if S == 0 (all chosen centroids coincide with all remaining
    // rows — degenerate duplicate input), pick uniformly from not-yet-chosen.
    let s: f64 = d.iter().sum::<f64>();
    if s == 0.0 {
      let chosen_ref = &chosen;
      let available: Vec<usize> = (0..n).filter(|j| !chosen_ref.contains(j)).collect();
      let idx = Uniform::new(0usize, available.len())
        .unwrap()
        .sample(&mut rng);
      let pick = available[idx];
      centroids.push(row(mat, pick));
      chosen.push(pick);
      continue;
    }

    // Step 2c: u ~ U[0, 1); t = u * S; smallest j with cumulative > t.
    // u ~ U[0, 1) (half-open). The strict `>` below requires u*S < S
    // strictly, which the half-open interval guarantees: cum can equal S
    // when accumulating all elements, but never exceed it for any prior
    // index when t < S.
    let u: f64 = rng.random::<f64>();
    let t = u * s;
    let mut cum = 0.0f64;
    // pick fallback = n - 1: if floating-point drift means cum never
    // strictly exceeds t (e.g., when t is very close to S), the last
    // index is the correct answer because cum reaches S exactly at j=n-1.
    // Plan code used pick = 0 (initial value); n - 1 is mathematically
    // the same probability mass for the boundary case but more defensible
    // semantically (you get the entry with the largest D^2 contribution).
    let mut pick = n - 1;
    for (j, &dj) in d.iter().enumerate() {
      cum += dj;
      if cum > t {
        pick = j;
        break;
      }
    }
    centroids.push(row(mat, pick));
    chosen.push(pick);
  }

  centroids
}

/// Extract row `i` of matrix `m` as `Vec<f64>`. Helper for K-means seeding /
/// Lloyd iteration (centroids are 1-D vectors over the row dimension).
#[allow(dead_code)] // used in tests; wired into cluster() in Task 21
fn row(m: &DMatrix<f64>, i: usize) -> Vec<f64> {
  m.row(i).iter().copied().collect()
}

#[cfg(test)]
mod kmeans_seed_tests {
  use super::*;

  #[test]
  fn same_seed_same_picks() {
    let mat = DMatrix::<f64>::from_row_slice(4, 2, &[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
    let a = kmeans_pp_seed(&mat, 2, 42);
    let b = kmeans_pp_seed(&mat, 2, 42);
    assert_eq!(a, b, "same seed must produce identical centroid picks");
  }

  #[test]
  fn kmeans_pp_seed_byte_determinism_fixture() {
    // Reference fixture for byte-determinism. Pins kmeans_pp_seed
    // output for a known input+seed. Future drift in the cumulative-
    // mass walk, summation order, f64 reductions, or rand 0.10
    // upgrades would fail this assertion FIRST — before Task 21
    // amplifies the failure into a label-stability regression.
    //
    // Companion to tests/chacha_keystream_fixture.rs (which pins the
    // underlying ChaCha8Rng keystream); this fixture pins the
    // algorithm-level output one layer up.
    let mat = DMatrix::<f64>::from_row_slice(4, 2, &[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 10.0, 10.0]);
    let centroids = kmeans_pp_seed(&mat, 2, 42);
    assert_eq!(centroids.len(), 2);
    assert_eq!(centroids[0].len(), 2);
    assert_eq!(centroids[1].len(), 2);

    // Exact byte-stable values pinned 2026-04-26 against rand 0.10.1 +
    // rand_chacha 0.10.0. If this test fails after a `cargo update`,
    // ChaCha8Rng → kmeans_pp_seed determinism has drifted; investigate
    // and re-pin only after auditing the rand changelog.
    let expected: [[f64; 2]; 2] = [[0.0, 0.0], [10.0, 10.0]];
    assert_eq!(
      centroids[0], expected[0],
      "centroid 0 byte-determinism drift: expected {:?}, got {:?}",
      expected[0], centroids[0]
    );
    assert_eq!(
      centroids[1], expected[1],
      "centroid 1 byte-determinism drift: expected {:?}, got {:?}",
      expected[1], centroids[1]
    );
  }

  #[test]
  fn different_seeds_can_pick_differently() {
    // 8 rows in two clearly-separated 2D clusters.
    let mat = DMatrix::<f64>::from_row_slice(
      8,
      2,
      &[
        0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.1, 5.0, 5.0, 5.1, 5.0, 5.0, 5.1, 5.1, 5.1,
      ],
    );
    let a = kmeans_pp_seed(&mat, 2, 0);
    let b = kmeans_pp_seed(&mat, 2, 999);
    // Both runs return K=2 centroids, each 2-dim. We don't assert the
    // picks differ — well-separated layouts can produce the same selection
    // from any seed when the D^2 weighting is decisive.
    assert_eq!(a.len(), 2);
    assert_eq!(b.len(), 2);
    assert_eq!(a[0].len(), 2);
    assert_eq!(b[0].len(), 2);
  }

  #[test]
  fn k_equals_n_picks_all_points() {
    // 3 rows, K = 3 → every row is selected exactly once.
    let mat = DMatrix::<f64>::from_row_slice(3, 1, &[0.0, 1.0, 2.0]);
    let centroids = kmeans_pp_seed(&mat, 3, 7);
    assert_eq!(centroids.len(), 3);
    let mut sorted_picks: Vec<f64> = centroids.iter().map(|c| c[0]).collect();
    sorted_picks.sort_by(|a, b| a.total_cmp(b));
    assert_eq!(sorted_picks, vec![0.0, 1.0, 2.0]);
  }
}

#[cfg(test)]
mod eigen_tests {
  use super::*;

  #[test]
  fn eigendecompose_identity_yields_unit_eigenvalues() {
    let id = DMatrix::<f64>::identity(4, 4);
    let (vals, _) = eigendecompose(id).unwrap();
    assert_eq!(vals.len(), 4);
    for v in vals {
      assert!(
        (v - 1.0).abs() < 1e-10,
        "identity should have all eigenvalues = 1.0; got {v}"
      );
    }
  }

  #[test]
  fn eigendecompose_diagonal_sorts_ascending() {
    // Diagonal matrix [3, 1, 2] → eigenvalues = [3, 1, 2] in arbitrary order;
    // we want ascending [1, 2, 3].
    let mut m = DMatrix::<f64>::zeros(3, 3);
    m[(0, 0)] = 3.0;
    m[(1, 1)] = 1.0;
    m[(2, 2)] = 2.0;
    let (vals, _) = eigendecompose(m).unwrap();
    assert_eq!(vals.len(), 3);
    assert!((vals[0] - 1.0).abs() < 1e-10);
    assert!((vals[1] - 2.0).abs() < 1e-10);
    assert!((vals[2] - 3.0).abs() < 1e-10);
  }

  #[test]
  fn pick_k_target_speakers_overrides_eigengap() {
    let eigs = vec![0.0, 0.5, 0.6, 0.95];
    assert_eq!(pick_k(&eigs, 4, Some(3)), 3);
    assert_eq!(pick_k(&eigs, 4, Some(1)), 1);
  }

  #[test]
  fn pick_k_eigengap_picks_largest_jump() {
    // Gaps: 0.01-0.0=0.01, 0.02-0.01=0.01, 0.9-0.02=0.88. Largest at k=2,
    // returning best_k = 2 + 1 = 3.
    let eigs = vec![0.0, 0.01, 0.02, 0.9];
    assert_eq!(pick_k(&eigs, 4, None), 3);
  }

  #[test]
  fn pick_k_caps_at_max_auto_speakers() {
    // 30 ascending eigenvalues with uniform tiny gaps. The cap, not the
    // argmax, drives the result.
    let eigs: Vec<f64> = (0..30).map(|i| i as f64 * 0.01).collect();
    let k = pick_k(&eigs, 30, None);
    assert!(
      k <= MAX_AUTO_SPEAKERS as usize,
      "K must be ≤ MAX_AUTO_SPEAKERS = {}, got {k}",
      MAX_AUTO_SPEAKERS
    );
  }

  #[test]
  fn pick_k_target_equals_n_returns_n() {
    // target = N is allowed (every embedding can be its own cluster);
    // pick_k should pass it through unchanged.
    let eigs = vec![0.0, 0.5, 0.6, 0.95];
    assert_eq!(pick_k(&eigs, 4, Some(4)), 4);
  }

  #[test]
  fn pipeline_two_clear_clusters_separates_eigenvalues() {
    // End-to-end smoke: 6 embeddings forming two well-separated groups
    // (3 near unit(0), 3 near unit(10)) → run through the full Phase-3
    // pipeline up to eigendecomposition. Expect:
    //   - All N eigenvalues finite and >= 0 (PSD Laplacian).
    //   - Smallest eigenvalue close to 0 (single connected component
    //     within each group; nullspace dimension >= 1 → λ_0 ≈ 0).
    //   - Sorted ascending.
    use crate::embed::EMBEDDING_DIM;
    fn perturbed_unit(i: usize, scale: f32) -> crate::embed::Embedding {
      let mut v = [0.0f32; EMBEDDING_DIM];
      v[i] = 1.0;
      v[(i + 1) % EMBEDDING_DIM] = scale;
      crate::embed::Embedding::normalize_from(v).unwrap()
    }

    let mut e = Vec::new();
    for s in [0.0, 0.05, -0.05] {
      e.push(perturbed_unit(0, s));
    }
    for s in [0.0, 0.05, -0.05] {
      e.push(perturbed_unit(10, s));
    }

    let aff = build_affinity(&e);
    let d =
      compute_degrees(&aff).expect("two well-separated clusters → AllDissimilar should not fire");
    let l = normalized_laplacian(&aff, &d);
    let (vals, vecs) = eigendecompose(l).expect("symmetric Laplacian must decompose cleanly");

    // Output shape: N eigenvalues, N×N eigenvector matrix.
    assert_eq!(vals.len(), 6);
    assert_eq!(vecs.nrows(), 6);
    assert_eq!(vecs.ncols(), 6);

    // PSD: all eigenvalues finite and >= -tolerance (the small negative
    // tolerance covers f64 rounding around λ ≈ 0).
    let tolerance = 1e-9;
    for (k, v) in vals.iter().enumerate() {
      assert!(v.is_finite(), "eigenvalue {k} = {v} should be finite");
      assert!(
        *v >= -tolerance,
        "eigenvalue {k} = {v} should be >= 0 (PSD Laplacian)"
      );
    }

    // Sorted ascending: vals[0] <= vals[1] <= ... <= vals[N-1].
    for w in vals.windows(2) {
      assert!(w[0] <= w[1], "eigenvalues must be sorted ascending");
    }

    // Smallest eigenvalue close to 0 (the all-ones vector lies in the
    // nullspace of the normalized Laplacian for a connected graph; with
    // two disconnected clusters there's at least a 2D nullspace).
    assert!(
      vals[0].abs() < 1e-6,
      "λ_0 should be ≈ 0 for the connected/disconnected-component normalized Laplacian; got {}",
      vals[0]
    );
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
  fn affinity_diagonal_is_zero() {
    let e = vec![unit(0), unit(1), unit(2)];
    let a = build_affinity(&e);
    for i in 0..3 {
      assert_eq!(a[(i, i)], 0.0);
    }
  }

  #[test]
  fn affinity_relu_clamps_negatives() {
    // e[1] is the antipode of e[0]: cosine = -1, clamped to 0.
    let mut neg = [0.0f32; EMBEDDING_DIM];
    neg[0] = -1.0;
    let e = vec![unit(0), Embedding::normalize_from(neg).unwrap(), unit(1)];
    let a = build_affinity(&e);
    assert_eq!(a[(0, 1)], 0.0);
    assert_eq!(a[(1, 0)], 0.0);
    // e[0] · e[2] = 0 (orthogonal axes); ReLU keeps as 0.
    assert_eq!(a[(0, 2)], 0.0);
  }

  #[test]
  fn affinity_identical_embeddings_is_one() {
    // Three copies of unit(0): cosine similarity = 1.0 between every
    // pair; ReLU clamp leaves it at 1.0. Confirms the positive path
    // through the .max(0.0) doesn't accidentally clamp positives.
    let e = vec![unit(0), unit(0), unit(0)];
    let a = build_affinity(&e);
    for i in 0..3 {
      for j in 0..3 {
        if i == j {
          assert_eq!(a[(i, j)], 0.0, "diagonal must stay 0");
        } else {
          assert!(
            (a[(i, j)] - 1.0).abs() < 1e-6,
            "identical embeddings: A[{i}][{j}] should be ~1.0; got {}",
            a[(i, j)]
          );
        }
      }
    }
  }

  #[test]
  fn isolated_node_triggers_alldissimilar() {
    // e[0] and e[1] are close (sim ≈ 0.9), e[2] is orthogonal to both
    // → row-2 of A is all zero → D_22 = 0 < eps → AllDissimilar.
    let mut close_to_0 = [0.0f32; EMBEDDING_DIM];
    close_to_0[0] = 0.9;
    close_to_0[1] = 0.1;
    let e = vec![
      unit(0),
      Embedding::normalize_from(close_to_0).unwrap(),
      unit(2),
    ];
    let a = build_affinity(&e);
    let r = compute_degrees(&a);
    assert!(matches!(r, Err(Error::AllDissimilar)));
  }

  #[test]
  fn all_zero_affinity_triggers_alldissimilar() {
    // Three mutually-orthogonal embeddings → A is all-zero everywhere.
    // Every degree is 0 → AllDissimilar.
    let e = vec![unit(0), unit(1), unit(2)];
    let a = build_affinity(&e);
    let r = compute_degrees(&a);
    assert!(matches!(r, Err(Error::AllDissimilar)));
  }

  #[test]
  fn laplacian_diag_is_one_off_diag_negative() {
    // Construct three embeddings with positive pairwise affinity so
    // that the Laplacian is well-defined.
    let mut a_vec = [0.0f32; EMBEDDING_DIM];
    a_vec[0] = 0.9;
    a_vec[1] = 0.4;
    let mut b_vec = [0.0f32; EMBEDDING_DIM];
    b_vec[0] = 0.4;
    b_vec[1] = 0.9;
    let e = vec![
      Embedding::normalize_from(a_vec).unwrap(),
      Embedding::normalize_from(b_vec).unwrap(),
      unit(0),
    ];
    let aff = build_affinity(&e);
    let d = compute_degrees(&aff).unwrap();
    let l = normalized_laplacian(&aff, &d);
    for i in 0..3 {
      assert!(
        (l[(i, i)] - 1.0).abs() < 1e-12,
        "L_sym diagonal must be exactly 1.0; got {}",
        l[(i, i)]
      );
    }
    // For an off-diagonal where affinity is positive (e[0]·e[1] > 0),
    // L_ij = -D^{-1/2} A_ij D^{-1/2} < 0.
    assert!(
      l[(0, 1)] < 0.0,
      "L_sym off-diagonal where A>0 must be negative; got {}",
      l[(0, 1)]
    );
  }
}
