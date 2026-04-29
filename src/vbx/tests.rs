//! Model-free unit tests for `dia::vbx`.
//!
//! Heavy parity tests against pyannote's captured outputs live in
//! `src/vbx/parity_tests.rs`. This module covers smaller, model-free
//! invariants — the kind of thing that should hold for any input,
//! and that catches regressions long before the parity tests fail.

use super::algo::logsumexp_rows;
use nalgebra::DMatrix;

/// scipy.special.logsumexp on a 2x3 matrix along axis=-1 returns a
/// length-2 vector. Reference values computed in Python:
///
/// ```python
/// >>> import math
/// >>> vals = [-100.0, -101.0, -102.0]; mx = max(vals)
/// >>> math.log(sum(math.exp(v - mx) for v in vals)) + mx
/// -99.59239403555561
/// ```
///
/// Row0: logsumexp([1, 2, 3]) = log(e^1 + e^2 + e^3) ≈ 3.40760596
/// Row1: logsumexp([-100, -101, -102]) ≈ -99.59239403555561
#[test]
fn logsumexp_rows_matches_scipy_reference() {
  let m = DMatrix::<f64>::from_row_slice(2, 3, &[1.0, 2.0, 3.0, -100.0, -101.0, -102.0]);
  let lse = logsumexp_rows(&m);
  assert!((lse[0] - 3.40760596).abs() < 1e-8, "row0: {}", lse[0]);
  assert!(
    (lse[1] - (-99.592_394_035_555_61)).abs() < 1e-10,
    "row1: {}",
    lse[1]
  );
}

/// All -inf row → -inf result (matches scipy behavior).
#[test]
fn logsumexp_rows_all_neg_inf_returns_neg_inf() {
  let m = DMatrix::<f64>::from_row_slice(
    1,
    3,
    &[f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY],
  );
  let lse = logsumexp_rows(&m);
  assert!(lse[0].is_infinite() && lse[0] < 0.0, "got {}", lse[0]);
}

use crate::vbx::{Error, vbx_iterate};
use nalgebra::DVector;

#[test]
fn vbx_rejects_phi_with_non_positive_entry() {
  let x = DMatrix::<f64>::zeros(5, 4);
  let mut phi = DVector::<f64>::from_element(4, 1.0);
  phi[2] = -0.5;
  let qinit = DMatrix::<f64>::from_element(5, 2, 0.5);
  let result = vbx_iterate(&x, &phi, &qinit, 0.07, 0.8, 20);
  assert!(
    matches!(result, Err(Error::NonPositivePhi(_, 2))),
    "got {result:?}"
  );
}

#[test]
fn vbx_rejects_shape_mismatch_x_vs_qinit() {
  let x = DMatrix::<f64>::zeros(5, 4);
  let phi = DVector::<f64>::from_element(4, 1.0);
  let qinit = DMatrix::<f64>::from_element(6, 2, 0.5); // T=6 ≠ 5
  let result = vbx_iterate(&x, &phi, &qinit, 0.07, 0.8, 20);
  assert!(matches!(result, Err(Error::Shape(_))), "got {result:?}");
}

#[test]
fn vbx_rejects_shape_mismatch_phi_vs_x() {
  let x = DMatrix::<f64>::zeros(5, 4); // D=4
  let phi = DVector::<f64>::from_element(3, 1.0); // D=3 ≠ 4
  let qinit = DMatrix::<f64>::from_element(5, 2, 0.5);
  let result = vbx_iterate(&x, &phi, &qinit, 0.07, 0.8, 20);
  assert!(matches!(result, Err(Error::Shape(_))), "got {result:?}");
}

#[test]
fn vbx_rejects_qinit_with_zero_clusters() {
  let x = DMatrix::<f64>::zeros(5, 4);
  let phi = DVector::<f64>::from_element(4, 1.0);
  let qinit = DMatrix::<f64>::zeros(5, 0);
  let result = vbx_iterate(&x, &phi, &qinit, 0.07, 0.8, 20);
  assert!(matches!(result, Err(Error::Shape(_))), "got {result:?}");
}

/// VBx must produce a monotonically non-decreasing ELBO (modulo a tiny
/// epsilon-band at convergence). A regression that, e.g., reuses the
/// previous iteration's gamma in the alpha update would break this.
#[test]
fn vbx_elbo_is_monotonically_non_decreasing() {
  // 50 frames × 8 dim × 3 speakers, deterministic non-pathological input.
  let t = 50;
  let d = 8;
  let s = 3;
  let mut x = DMatrix::<f64>::zeros(t, d);
  for i in 0..t {
    for j in 0..d {
      x[(i, j)] = ((i * 7 + j * 13) as f64 % 11.0) - 5.0;
    }
  }
  let phi = DVector::<f64>::from_element(d, 2.0);
  let qinit = DMatrix::<f64>::from_element(t, s, 1.0 / s as f64);
  let out = vbx_iterate(&x, &phi, &qinit, 0.07, 0.8, 20).expect("vbx_iterate");
  for w in out.elbo_trajectory.windows(2) {
    // Allow tiny float wobble at convergence (≤ 1e-6) before the
    // epsilon-based stop fires.
    assert!(
      w[1] - w[0] > -1.0e-6,
      "ELBO must not decrease: {} → {}",
      w[0],
      w[1]
    );
  }
}

/// At every iteration, `gamma[t, :]` is a discrete probability over
/// speakers, so each row must sum to 1 (within float roundoff).
#[test]
fn vbx_gamma_rows_sum_to_one() {
  let t = 30;
  let d = 4;
  let s = 4;
  let mut x = DMatrix::<f64>::zeros(t, d);
  for i in 0..t {
    for j in 0..d {
      x[(i, j)] = ((i + j) as f64).sin();
    }
  }
  let phi = DVector::<f64>::from_element(d, 1.5);
  let qinit = DMatrix::<f64>::from_element(t, s, 1.0 / s as f64);
  let out = vbx_iterate(&x, &phi, &qinit, 0.1, 0.5, 10).expect("vbx_iterate");
  for r in 0..t {
    let row_sum: f64 = (0..s).map(|c| out.gamma[(r, c)]).sum();
    assert!(
      (row_sum - 1.0).abs() < 1e-12,
      "gamma row {r} sums to {row_sum}"
    );
  }
}

/// `pi` is a discrete probability over speakers; it must sum to 1.
#[test]
fn vbx_pi_sums_to_one() {
  let t = 20;
  let d = 4;
  let s = 5;
  let x = DMatrix::<f64>::from_fn(t, d, |i, j| ((i * 3 + j) as f64).cos());
  let phi = DVector::<f64>::from_element(d, 1.0);
  let qinit = DMatrix::<f64>::from_element(t, s, 1.0 / s as f64);
  let out = vbx_iterate(&x, &phi, &qinit, 0.07, 0.8, 20).expect("vbx_iterate");
  let pi_sum: f64 = out.pi.iter().sum();
  assert!((pi_sum - 1.0).abs() < 1e-12, "pi sums to {pi_sum}");
}

/// The algorithm has no RNG anywhere, so two calls with the same input
/// must return bit-identical outputs. Catches regressions where, e.g.,
/// `HashMap` ordering or `f64::partial_cmp` tiebreaks leak into the
/// algorithm.
#[test]
fn vbx_is_deterministic() {
  let t = 15;
  let d = 4;
  let s = 3;
  let x = DMatrix::<f64>::from_fn(t, d, |i, j| (i + 2 * j) as f64 * 0.1);
  let phi = DVector::<f64>::from_element(d, 2.0);
  let qinit = DMatrix::<f64>::from_element(t, s, 1.0 / s as f64);
  let a = vbx_iterate(&x, &phi, &qinit, 0.07, 0.8, 10).expect("a");
  let b = vbx_iterate(&x, &phi, &qinit, 0.07, 0.8, 10).expect("b");
  assert_eq!(a.elbo_trajectory, b.elbo_trajectory);
  for r in 0..t {
    for c in 0..s {
      assert_eq!(a.gamma[(r, c)], b.gamma[(r, c)]);
    }
  }
  for c in 0..s {
    assert_eq!(a.pi[c], b.pi[c]);
  }
}

// ── Input-value validation (Codex review MEDIUM round 1 of Phase 2) ─
//
// Round 1 added validation for `qinit` (finite, nonnegative,
// row-sum ≈ 1) and for `Fa`/`Fb` (positive, finite). Without these,
// a malformed initializer or hyperparameter silently biases the
// first speaker-model update and propagates garbage through the rest
// of the run; pyannote does not validate these, so this is a
// deliberate divergence to fail-fast at the boundary instead of
// producing fabricated speaker evidence.

#[test]
fn vbx_rejects_qinit_with_nan_entry() {
  let t = 5;
  let s = 2;
  let x = DMatrix::<f64>::zeros(t, 4);
  let phi = DVector::<f64>::from_element(4, 1.0);
  let mut qinit = DMatrix::<f64>::from_element(t, s, 0.5);
  qinit[(2, 1)] = f64::NAN;
  let result = vbx_iterate(&x, &phi, &qinit, 0.07, 0.8, 20);
  assert!(
    matches!(result, Err(Error::NonFinite("qinit"))),
    "got {result:?}"
  );
}

#[test]
fn vbx_rejects_qinit_with_inf_entry() {
  let t = 5;
  let s = 2;
  let x = DMatrix::<f64>::zeros(t, 4);
  let phi = DVector::<f64>::from_element(4, 1.0);
  let mut qinit = DMatrix::<f64>::from_element(t, s, 0.5);
  qinit[(0, 0)] = f64::INFINITY;
  let result = vbx_iterate(&x, &phi, &qinit, 0.07, 0.8, 20);
  assert!(
    matches!(result, Err(Error::NonFinite("qinit"))),
    "got {result:?}"
  );
}

#[test]
fn vbx_rejects_qinit_with_negative_entry() {
  let t = 5;
  let s = 2;
  let x = DMatrix::<f64>::zeros(t, 4);
  let phi = DVector::<f64>::from_element(4, 1.0);
  // Per-row sum still 1.0 (0.6 + 0.4) so we exercise the negative-
  // value path, not the row-sum path. Set one entry to -0.1 and
  // bump its sibling to 1.1 so the row sums to 1.0.
  let mut qinit = DMatrix::<f64>::from_element(t, s, 0.5);
  qinit[(0, 0)] = -0.1;
  qinit[(0, 1)] = 1.1;
  let result = vbx_iterate(&x, &phi, &qinit, 0.07, 0.8, 20);
  assert!(matches!(result, Err(Error::Shape(_))), "got {result:?}");
}

#[test]
fn vbx_rejects_qinit_with_unnormalized_row() {
  let t = 5;
  let s = 2;
  let x = DMatrix::<f64>::zeros(t, 4);
  let phi = DVector::<f64>::from_element(4, 1.0);
  // Row 3 has entries [0.5, 0.4] — sum = 0.9, fails the 1e-9 tolerance.
  let mut qinit = DMatrix::<f64>::from_element(t, s, 0.5);
  qinit[(3, 1)] = 0.4;
  let result = vbx_iterate(&x, &phi, &qinit, 0.07, 0.8, 20);
  assert!(matches!(result, Err(Error::Shape(_))), "got {result:?}");
}

#[test]
fn vbx_rejects_zero_fa() {
  let t = 5;
  let s = 2;
  let x = DMatrix::<f64>::zeros(t, 4);
  let phi = DVector::<f64>::from_element(4, 1.0);
  let qinit = DMatrix::<f64>::from_element(t, s, 0.5);
  let result = vbx_iterate(&x, &phi, &qinit, 0.0, 0.8, 20);
  assert!(matches!(result, Err(Error::Shape(_))), "got {result:?}");
}

#[test]
fn vbx_rejects_negative_fa() {
  let t = 5;
  let s = 2;
  let x = DMatrix::<f64>::zeros(t, 4);
  let phi = DVector::<f64>::from_element(4, 1.0);
  let qinit = DMatrix::<f64>::from_element(t, s, 0.5);
  let result = vbx_iterate(&x, &phi, &qinit, -0.1, 0.8, 20);
  assert!(matches!(result, Err(Error::Shape(_))), "got {result:?}");
}

#[test]
fn vbx_rejects_nan_fa() {
  let t = 5;
  let s = 2;
  let x = DMatrix::<f64>::zeros(t, 4);
  let phi = DVector::<f64>::from_element(4, 1.0);
  let qinit = DMatrix::<f64>::from_element(t, s, 0.5);
  let result = vbx_iterate(&x, &phi, &qinit, f64::NAN, 0.8, 20);
  assert!(matches!(result, Err(Error::Shape(_))), "got {result:?}");
}

#[test]
fn vbx_rejects_zero_fb() {
  let t = 5;
  let s = 2;
  let x = DMatrix::<f64>::zeros(t, 4);
  let phi = DVector::<f64>::from_element(4, 1.0);
  let qinit = DMatrix::<f64>::from_element(t, s, 0.5);
  let result = vbx_iterate(&x, &phi, &qinit, 0.07, 0.0, 20);
  assert!(matches!(result, Err(Error::Shape(_))), "got {result:?}");
}

#[test]
fn vbx_rejects_inf_fb() {
  let t = 5;
  let s = 2;
  let x = DMatrix::<f64>::zeros(t, 4);
  let phi = DVector::<f64>::from_element(4, 1.0);
  let qinit = DMatrix::<f64>::from_element(t, s, 0.5);
  let result = vbx_iterate(&x, &phi, &qinit, 0.07, f64::INFINITY, 20);
  assert!(matches!(result, Err(Error::Shape(_))), "got {result:?}");
}

/// `max_iters == 0` is a valid edge case (matches pyannote's
/// `range(0)`): the EM loop never runs, gamma is the (validated)
/// qinit, pi is uniform `1/S`, elbo_trajectory is empty. The qinit
/// validation is what makes this safe — without it, the function
/// would return whatever malformed values the caller passed in as
/// "responsibilities".
#[test]
fn vbx_max_iters_zero_returns_validated_qinit() {
  let t = 5;
  let s = 3;
  let x = DMatrix::<f64>::zeros(t, 4);
  let phi = DVector::<f64>::from_element(4, 1.0);
  let qinit = DMatrix::<f64>::from_element(t, s, 1.0 / s as f64);
  let out = vbx_iterate(&x, &phi, &qinit, 0.07, 0.8, 0).expect("vbx_iterate");
  assert!(
    out.elbo_trajectory.is_empty(),
    "no iterations → empty trajectory"
  );
  for r in 0..t {
    for c in 0..s {
      assert_eq!(out.gamma[(r, c)], qinit[(r, c)]);
    }
  }
  for c in 0..s {
    assert!((out.pi[c] - 1.0 / s as f64).abs() < 1e-15);
  }
}

// ── ELBO step classification (Codex review MEDIUM round 2 of Phase 2) ─
//
// VB EM's monotonicity is a fundamental invariant. The previous
// `delta < epsilon` convergence branch fired for both small-positive
// improvements (intended) and negative deltas (a regression — bug
// or numerical instability). The new `classify_elbo_step` helper
// separates the three regimes, and `vbx_iterate` propagates a
// regression as `Error::ElboRegression` rather than silently
// returning the regressed posterior.

use super::algo::{ElboStep, classify_elbo_step};

#[test]
fn classify_elbo_step_continues_on_large_positive_delta() {
  assert_eq!(classify_elbo_step(0.5, 1.0e-4), ElboStep::Continue);
}

#[test]
fn classify_elbo_step_converges_on_small_positive_delta() {
  assert_eq!(classify_elbo_step(1.0e-5, 1.0e-4), ElboStep::Converged);
}

#[test]
fn classify_elbo_step_converges_on_tiny_negative_delta_within_tolerance() {
  // Delta in float-roundoff regime — treat as converged.
  assert_eq!(
    classify_elbo_step(-1.0e-12, 1.0e-4),
    ElboStep::Converged
  );
}

#[test]
fn classify_elbo_step_regresses_on_large_negative_delta() {
  match classify_elbo_step(-1.0e-4, 1.0e-4) {
    ElboStep::Regressed(d) => assert_eq!(d, -1.0e-4),
    other => panic!("expected Regressed, got {other:?}"),
  }
}

#[test]
fn classify_elbo_step_regresses_just_outside_tolerance() {
  // Just past the regression tolerance — should error, not converge.
  match classify_elbo_step(-1.0e-8, 1.0e-4) {
    ElboStep::Regressed(d) => assert_eq!(d, -1.0e-8),
    other => panic!("expected Regressed, got {other:?}"),
  }
}

#[test]
fn classify_elbo_step_zero_delta_is_converged() {
  // Exactly zero — flat ELBO, treat as converged.
  assert_eq!(classify_elbo_step(0.0, 1.0e-4), ElboStep::Converged);
}
