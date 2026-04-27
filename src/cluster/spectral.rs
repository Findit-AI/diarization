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
  cluster::{Error, options::OfflineClusterOptions},
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
  let n = embeddings.len();
  debug_assert!(n >= 3, "fast path covers N <= 2");

  // Step 1: affinity matrix A[i][j] = max(0, e_i · e_j); A_ii = 0.
  let a = build_affinity(embeddings);

  // Step 2: degree vector D_ii = sum_j A_ij. Returns AllDissimilar
  // if any node is isolated (D_ii < NORM_EPSILON) — covers both the
  // all-zero affinity case and the rev-3 widened isolated-node case.
  let degrees = compute_degrees(&a)?;

  // Step 3: normalized Laplacian L_sym = I - D^{-1/2} A D^{-1/2}.
  let _l = normalized_laplacian(&a, &degrees);

  // Steps 4-9 (eigendecomp + K-detection + row-normalize + K-means)
  // arrive in Tasks 19-21.
  let _ = opts;
  todo!("Tasks 19-21 fill in the remaining spectral pipeline")
}

/// Build the N x N affinity matrix `A[i][j] = max(0, e_i · e_j)`; `A[i][i] = 0`.
///
/// Affinity is f64 for numerical stability through the eigendecomposition.
/// ReLU clamp matches spec §5.5 step 1 (rev-3).
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
pub(crate) fn normalized_laplacian(a: &DMatrix<f64>, d: &[f64]) -> DMatrix<f64> {
  let n = a.nrows();
  // D^{-1/2} as a diagonal matrix.
  let inv_sqrt = DVector::from_iterator(n, d.iter().map(|&di| 1.0 / di.sqrt()));
  let inv_sqrt_diag = DMatrix::from_diagonal(&inv_sqrt);
  // L_sym = I - D^{-1/2} A D^{-1/2}.
  DMatrix::<f64>::identity(n, n) - &inv_sqrt_diag * a * &inv_sqrt_diag
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
