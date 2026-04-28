//! Module-level tests for `dia::plda`.
//!
//! Heavy parity tests against pyannote's captured outputs live in
//! `tests/parity_plda.rs`. This module covers smaller, model-free
//! invariants — the kind of thing that should hold for any input,
//! and that catches regressions long before the parity tests fail.

use crate::plda::{
  EMBEDDING_DIMENSION, Error, PLDA_DIMENSION, PldaTransform, PostXvecEmbedding, RawEmbedding,
};

fn raw(arr: [f32; EMBEDDING_DIMENSION]) -> RawEmbedding {
  RawEmbedding::from_raw_array(arr).expect("test input must be finite")
}

/// `xvec_transform` output norm is `sqrt(PLDA_DIMENSION) ≈ 11.31` —
/// see `pyannote/audio/utils/vbx.py:211-213`. Catches silent
/// regressions where the outer `sqrt(D_out)` factor is dropped.
#[test]
fn xvec_transform_output_norm_is_sqrt_d_out() {
  let plda = PldaTransform::new().expect("load PLDA");
  // Constant input — non-trivial after centering by mean1.
  let input = raw([0.1f32; EMBEDDING_DIMENSION]);
  let out = plda.xvec_transform(&input).expect("non-degenerate input");
  let norm = out.as_array().iter().map(|v| v * v).sum::<f64>().sqrt();
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
  let input = raw([0.5f32; EMBEDDING_DIMENSION]);
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
  let input = raw([0.0f32; EMBEDDING_DIMENSION]);
  let pa = a.project(&input).expect("non-degenerate");
  let pb = b.project(&input).expect("non-degenerate");
  assert_eq!(pa, pb);
}

// ── Validation tests (Codex review MEDIUM + HIGH) ──────────────────
//
// Input finite-ness is now enforced at `RawEmbedding::from_raw_array`
// construction — `xvec_transform` cannot receive a non-finite input
// at all. Tests that previously fed NaN/Inf directly to
// `xvec_transform` therefore moved to the constructor.

/// NaN input must be rejected at the `RawEmbedding` boundary so it
/// cannot reach any math. Without this check, NaN propagates silently
/// into VBx / clustering with no observability for the caller.
#[test]
fn raw_embedding_rejects_nan() {
  let mut arr = [0.5f32; EMBEDDING_DIMENSION];
  arr[42] = f32::NAN;
  let result = RawEmbedding::from_raw_array(arr);
  assert!(
    matches!(result, Err(Error::NonFiniteInput)),
    "got {result:?}"
  );
}

#[test]
fn raw_embedding_rejects_pos_inf() {
  let mut arr = [0.5f32; EMBEDDING_DIMENSION];
  arr[7] = f32::INFINITY;
  let result = RawEmbedding::from_raw_array(arr);
  assert!(
    matches!(result, Err(Error::NonFiniteInput)),
    "got {result:?}"
  );
}

#[test]
fn raw_embedding_rejects_neg_inf() {
  let mut arr = [0.5f32; EMBEDDING_DIMENSION];
  arr[42] = f32::NEG_INFINITY;
  let result = RawEmbedding::from_raw_array(arr);
  assert!(
    matches!(result, Err(Error::NonFiniteInput)),
    "got {result:?}"
  );
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

// ── PostXvecEmbedding boundary (Codex review HIGH stage 2) ─────────
//
// `plda_transform` no longer accepts a bare `[f64; 128]` — its input
// is now `&PostXvecEmbedding`, a newtype that enforces the post-`xvec_tf`
// distribution invariant. NaN/Inf rejection moved to the constructor.

#[test]
fn post_xvec_capture_rejects_nan() {
  let mut arr = [0.0f64; PLDA_DIMENSION];
  arr[3] = f64::NAN;
  let result = PostXvecEmbedding::from_pyannote_capture(arr);
  assert!(matches!(result, Err(Error::NonFiniteInput)), "got {result:?}");
}

#[test]
fn post_xvec_capture_rejects_inf() {
  let mut arr = [0.0f64; PLDA_DIMENSION];
  arr[100] = f64::INFINITY;
  let result = PostXvecEmbedding::from_pyannote_capture(arr);
  assert!(matches!(result, Err(Error::NonFiniteInput)), "got {result:?}");
}

/// L2-normalized 128-d vector (norm = 1.0) is the most likely
/// stage-2 misuse. The `from_pyannote_capture` norm check rejects it.
#[test]
fn post_xvec_capture_rejects_l2_normalized_vector() {
  let mut arr = [0.0f64; PLDA_DIMENSION];
  arr[0] = 1.0; // unit vector along axis 0 — norm = 1.0
  let result = PostXvecEmbedding::from_pyannote_capture(arr);
  assert!(
    matches!(result, Err(Error::WrongPostXvecNorm { actual, expected, .. })
        if (actual - 1.0).abs() < 1e-12 && (expected - (PLDA_DIMENSION as f64).sqrt()).abs() < 1e-9),
    "got {result:?}"
  );
}

/// Random / hand-constructed input with arbitrary norm is also
/// rejected. Catches accidental zero-vectors, mis-scaled inputs, etc.
#[test]
fn post_xvec_capture_rejects_zero_vector() {
  let arr = [0.0f64; PLDA_DIMENSION];
  let result = PostXvecEmbedding::from_pyannote_capture(arr);
  assert!(
    matches!(result, Err(Error::WrongPostXvecNorm { actual: 0.0, .. })),
    "got {result:?}"
  );
}

/// Sanity: a synthetic vector with the right norm passes the gate.
#[test]
fn post_xvec_capture_accepts_correctly_scaled_vector() {
  let expected_norm = (PLDA_DIMENSION as f64).sqrt();
  let per_elem = expected_norm / (PLDA_DIMENSION as f64).sqrt();
  // each element = 1.0; sum of squares = 128; norm = sqrt(128) ✓
  assert!((per_elem - 1.0).abs() < 1e-12);
  let arr = [per_elem; PLDA_DIMENSION];
  let post =
    PostXvecEmbedding::from_pyannote_capture(arr).expect("right norm");
  assert_eq!(post.as_array().len(), PLDA_DIMENSION);
}

/// Round-trip: `xvec_transform`'s output goes straight into
/// `plda_transform` via the type system — no extra validation needed.
#[test]
fn xvec_to_plda_round_trip_uses_post_xvec_type() {
  let plda = PldaTransform::new().expect("load PLDA");
  let input = raw([0.5f32; EMBEDDING_DIMENSION]);
  let post = plda.xvec_transform(&input).expect("non-degenerate");
  let _ = plda.plda_transform(&post); // infallible — no Result on stage 2
}

// ── RawEmbedding domain enforcement (Codex review HIGH) ────────────

/// Feeding an L2-normalized vector (the wrong distribution for PLDA)
/// produces a materially-different output than feeding the
/// corresponding raw vector. The test is observable evidence that
/// the API distinction matters — if a future refactor accidentally
/// loses the `RawEmbedding` wrapper, this test stays as proof of
/// what's at stake.
///
/// We construct the same vector in both forms (`raw_arr` vs
/// `raw_arr / ‖raw_arr‖`), wrap each as `RawEmbedding`, and assert
/// that `xvec_transform`'s outputs differ by far more than float
/// roundoff.
#[test]
fn normalized_vs_raw_input_produce_materially_different_output() {
  let plda = PldaTransform::new().expect("load PLDA");

  // Use a noticeably-non-unit input vector.
  let mut raw_arr = [0.0f32; EMBEDDING_DIMENSION];
  for (i, slot) in raw_arr.iter_mut().enumerate() {
    *slot = ((i as f32) - 128.0) * 0.01;
  }
  let raw_norm: f32 = raw_arr.iter().map(|v| v * v).sum::<f32>().sqrt();
  assert!(
    (raw_norm - 1.0).abs() > 0.5,
    "test input must be far from unit norm: norm = {raw_norm}"
  );
  let mut normed_arr = raw_arr;
  for slot in normed_arr.iter_mut() {
    *slot /= raw_norm;
  }

  let raw_in = raw(raw_arr);
  let normed_in = raw(normed_arr);
  let raw_out = plda.xvec_transform(&raw_in).expect("raw out");
  let normed_out = plda.xvec_transform(&normed_in).expect("normed out");

  let l1_diff: f64 = raw_out
    .as_array()
    .iter()
    .zip(normed_out.as_array().iter())
    .map(|(a, b)| (a - b).abs())
    .sum();
  // The PLDA transform is non-linear (centering + L2-norm + sqrt(D)
  // scaling at two different stages); identical inputs always
  // produce identical outputs, but materially different inputs
  // (raw vs L2-normalized) produce materially different outputs.
  // This bound (>1.0 sum-abs-difference over 128 dims) is loose
  // enough to be robust to tiny test-input changes but tight
  // enough to catch a regression where the type system stops
  // distinguishing raw from normalized.
  assert!(
    l1_diff > 1.0,
    "normalized vs raw produced near-identical output (sum-abs diff = \
     {l1_diff:.3e}); the API contract is broken"
  );
}
