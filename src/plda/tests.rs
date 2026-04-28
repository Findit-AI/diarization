//! Module-level tests for `dia::plda`.
//!
//! Heavy parity tests against pyannote's captured outputs live in
//! `tests/parity_plda.rs`. This module covers smaller, model-free
//! invariants — the kind of thing that should hold for any input,
//! and that catches regressions long before the parity tests fail.

use crate::plda::{EMBEDDING_DIMENSION, PLDA_DIMENSION, PldaTransform};

/// `xvec_transform` output norm is `sqrt(PLDA_DIMENSION) ≈ 11.31` —
/// see `pyannote/audio/utils/vbx.py:211-213`. Catches silent
/// regressions where the outer `sqrt(D_out)` factor is dropped.
#[test]
fn xvec_transform_output_norm_is_sqrt_d_out() {
  let plda = PldaTransform::new().expect("load PLDA");
  // Constant input — non-trivial after centering by mean1.
  let input = [0.1f32; EMBEDDING_DIMENSION];
  let out = plda.xvec_transform(&input);
  let norm = out.iter().map(|v| v * v).sum::<f64>().sqrt();
  let expected = (PLDA_DIMENSION as f64).sqrt();
  assert!(
    (norm - expected).abs() < 1e-6,
    "xvec output norm = {norm}, expected sqrt({PLDA_DIMENSION}) = {expected}"
  );
}

/// `phi` (eigenvalues consumed by VBx) must be sorted descending. The
/// Cholesky-reduced eigh in `transform.rs::generalized_eigh_descending`
/// must produce the same ordering as scipy's `eigh(...)[::-1]`.
#[test]
fn phi_is_sorted_descending() {
  let plda = PldaTransform::new().expect("load PLDA");
  let phi = plda.phi();
  assert_eq!(phi.len(), PLDA_DIMENSION);
  for w in phi.windows(2) {
    assert!(
      w[0] >= w[1],
      "phi must be descending; saw {} < {}",
      w[0],
      w[1]
    );
  }
  // `phi` should also be strictly positive — the generalized eigh
  // of two positive-definite matrices has positive eigenvalues.
  assert!(phi.iter().all(|v| *v > 0.0), "phi must be positive");
}

/// `project()` is `plda_transform(xvec_transform(input))`. Cheap
/// algebraic property: shape-preserving + finite outputs.
#[test]
fn project_chain_is_finite() {
  let plda = PldaTransform::new().expect("load PLDA");
  let input = [0.5f32; EMBEDDING_DIMENSION];
  let projected = plda.project(&input);
  assert_eq!(projected.len(), PLDA_DIMENSION);
  assert!(
    projected.iter().all(|v| v.is_finite()),
    "project produced non-finite values: {projected:?}"
  );
}

/// PLDA construction is deterministic — no RNG anywhere in the load
/// path, so two `new()` calls must return bit-identical state.
#[test]
fn new_is_deterministic() {
  let a = PldaTransform::new().expect("load PLDA");
  let b = PldaTransform::new().expect("load PLDA");
  let phi_a = a.phi();
  let phi_b = b.phi();
  for (x, y) in phi_a.iter().zip(phi_b.iter()) {
    assert_eq!(x, y, "phi differs between two PldaTransform::new() calls");
  }
  // Same projection input → same output, byte-identical.
  let input = [0.0f32; EMBEDDING_DIMENSION];
  let pa = a.project(&input);
  let pb = b.project(&input);
  assert_eq!(pa, pb);
}
