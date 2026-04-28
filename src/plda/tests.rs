//! Module-level tests for `dia::plda`.
//!
//! Heavy parity tests against pyannote's captured outputs live in
//! `tests/parity_plda.rs`. This module covers smaller, model-free
//! invariants — the kind of thing that should hold for any input,
//! and that catches regressions long before the parity tests fail.

use crate::plda::{EMBEDDING_DIMENSION, Error, PLDA_DIMENSION, PldaTransform};

/// `xvec_transform` output norm is `sqrt(PLDA_DIMENSION) ≈ 11.31` —
/// see `pyannote/audio/utils/vbx.py:211-213`. Catches silent
/// regressions where the outer `sqrt(D_out)` factor is dropped.
#[test]
fn xvec_transform_output_norm_is_sqrt_d_out() {
  let plda = PldaTransform::new().expect("load PLDA");
  // Constant input — non-trivial after centering by mean1.
  let input = [0.1f32; EMBEDDING_DIMENSION];
  let out = plda.xvec_transform(&input).expect("non-degenerate input");
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
  let projected = plda.project(&input).expect("non-degenerate input");
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
  let pa = a.project(&input).expect("non-degenerate");
  let pb = b.project(&input).expect("non-degenerate");
  assert_eq!(pa, pb);
}

// ── Validation tests (Codex review MEDIUM) ─────────────────────────

/// NaN input must be rejected before any math runs. Without this
/// check, NaN propagates silently into VBx / clustering with no
/// observability for the caller.
#[test]
fn xvec_transform_rejects_nan_input() {
  let plda = PldaTransform::new().expect("load PLDA");
  let mut input = [0.5f32; EMBEDDING_DIMENSION];
  input[42] = f32::NAN;
  let result = plda.xvec_transform(&input);
  assert!(matches!(result, Err(Error::NonFiniteInput)), "got {result:?}");
}

/// `+inf` and `-inf` must also be rejected — they survive the L2-norm
/// check (norm = inf ≠ 0) and produce "plausible-looking" but
/// meaningless output without the explicit finite check.
#[test]
fn xvec_transform_rejects_pos_inf_input() {
  let plda = PldaTransform::new().expect("load PLDA");
  let mut input = [0.5f32; EMBEDDING_DIMENSION];
  input[7] = f32::INFINITY;
  let result = plda.xvec_transform(&input);
  assert!(matches!(result, Err(Error::NonFiniteInput)), "got {result:?}");
}

#[test]
fn xvec_transform_rejects_neg_inf_input() {
  let plda = PldaTransform::new().expect("load PLDA");
  let mut input = [0.5f32; EMBEDDING_DIMENSION];
  input[42] = f32::NEG_INFINITY;
  let result = plda.xvec_transform(&input);
  assert!(matches!(result, Err(Error::NonFiniteInput)), "got {result:?}");
}

// NOTE on `Error::DegenerateInput` from `xvec_transform`: that error
// only fires when the centered f64 vector `(input as f64) - mean1`
// has L2 norm below `NORM_EPSILON = 1e-12`. Because `mean1` is stored
// in f64 with non-f32-representable values, a `[f32; 256]` input
// rounded from `mean1` introduces ~1e-7 noise per element after
// promotion, which gives a centered norm around 1.6e-7 — above
// NORM_EPSILON. Triggering DegenerateInput from real f32 input is
// effectively impossible given pyannote's mean1 values, so the
// helper-level test for that path lives in `transform.rs` instead
// (see `tests::checked_l2_normalize_rejects_near_zero`).

/// `plda_transform` rejects NaN in `post_xvec`. There is no L2-norm
/// inside this stage so DegenerateInput is not applicable; only the
/// finite check applies.
#[test]
fn plda_transform_rejects_nan_input() {
  let plda = PldaTransform::new().expect("load PLDA");
  let mut input = [0.0f64; PLDA_DIMENSION];
  input[3] = f64::NAN;
  let result = plda.plda_transform(&input);
  assert!(matches!(result, Err(Error::NonFiniteInput)), "got {result:?}");
}

#[test]
fn plda_transform_rejects_inf_input() {
  let plda = PldaTransform::new().expect("load PLDA");
  let mut input = [0.0f64; PLDA_DIMENSION];
  input[100] = f64::INFINITY;
  let result = plda.plda_transform(&input);
  assert!(matches!(result, Err(Error::NonFiniteInput)), "got {result:?}");
}

/// `project()` propagates the underlying error from `xvec_transform`
/// rather than swallowing it.
#[test]
fn project_propagates_xvec_error() {
  let plda = PldaTransform::new().expect("load PLDA");
  let mut input = [0.0f32; EMBEDDING_DIMENSION];
  input[5] = f32::NAN;
  let result = plda.project(&input);
  assert!(matches!(result, Err(Error::NonFiniteInput)), "got {result:?}");
}
