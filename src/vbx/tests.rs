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

/// Deterministic non-uniform qinit for tests that need a valid VBx
/// initializer (i.e., one that breaks symmetry across speaker
/// columns). Each row `tt` is peaked on speaker `tt % s` with mass
/// 0.95; the remaining 0.05 mass is split evenly across the other
/// speakers. Per-row max for each column is 0.95, well above the
/// `1/S` uniform-rejection floor.
///
/// Codex round 9 closed the uniform-qinit symmetry trap. Tests
/// that previously used `DMatrix::from_element(t, s, 1.0 / s)`
/// (the uniform 1/S pattern) now use this helper.
fn deterministic_qinit(t: usize, s: usize) -> DMatrix<f64> {
  assert!(s > 1, "deterministic_qinit requires S > 1");
  let off = 0.05 / (s - 1) as f64;
  DMatrix::<f64>::from_fn(t, s, |tt, sj| if sj == tt % s { 0.95 } else { off })
}

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
  let qinit = deterministic_qinit(t, s);
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
  let qinit = deterministic_qinit(t, s);
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
  let qinit = deterministic_qinit(t, s);
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
  let qinit = deterministic_qinit(t, s);
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

/// `max_iters == 0` is rejected at the boundary. Skipping the EM
/// loop returns gamma=qinit and pi=1/S, which is internally
/// inconsistent for any non-uniform qinit (pi should equal
/// `gamma.column_sum() / T`) but indistinguishable from a completed
/// VBx run by the type system. Codex review MEDIUM round 7.
#[test]
fn vbx_rejects_max_iters_zero() {
  let t = 6;
  let s = 3;
  let x = DMatrix::<f64>::zeros(t, 4);
  let phi = DVector::<f64>::from_element(4, 1.0);
  // Use a valid (non-uniform, peaked-per-row) qinit so the
  // max_iters check is what fires, not the column-validation.
  let qinit = deterministic_qinit(t, s);
  let result = vbx_iterate(&x, &phi, &qinit, 0.07, 0.8, 0);
  assert!(matches!(result, Err(Error::Shape(_))), "got {result:?}");
}

/// Codex's round-7 recommended regression: a strongly non-uniform
/// qinit (each row peaked on a different speaker) with `max_iters = 0`
/// would return `gamma = qinit` and `pi = 1/S` — inconsistent
/// (`pi` should equal `gamma.col_sum() / T`). Now blocked at the
/// boundary by the max_iters check.
#[test]
fn vbx_rejects_max_iters_zero_with_non_uniform_qinit() {
  let t = 10;
  let s = 2;
  let d = 4;
  let x = DMatrix::<f64>::from_fn(t, d, |i, j| ((i + j) as f64) * 0.3);
  let phi = DVector::<f64>::from_element(d, 1.0);
  // Alternating per-row peak: even rows favor speaker 0, odd rows
  // speaker 1. Each column has rows with mass 0.95, well above the
  // 1/S = 0.5 floor — column-validation passes, max_iters check
  // is what fires.
  let qinit = deterministic_qinit(t, s);
  let result = vbx_iterate(&x, &phi, &qinit, 0.07, 0.8, 0);
  assert!(
    matches!(result, Err(Error::Shape(_))),
    "non-uniform qinit + max_iters=0 must reject (would otherwise \
     return gamma=qinit + pi=1/S inconsistent state); got {result:?}"
  );
}

// ── Allocation + dead-column hardening (Codex review MEDIUM round 4 of Phase 2) ─

/// Regression for the OOM concern: an enormous `max_iters` no longer
/// preallocates an `elbo_trajectory` of that capacity. The loop body
/// converges quickly on the small input here, so this exercises the
/// fix without actually running for billions of iterations.
///
/// Pre-fix: `Vec::with_capacity(usize::MAX)` panics at "capacity
/// overflow" before the loop runs.
/// Post-fix: `Vec::new()` allocates lazily; the algorithm converges
/// in 1-3 iterations on this small input and returns successfully.
#[test]
fn vbx_does_not_oom_on_huge_max_iters() {
  let t = 6;
  let s = 2;
  let d = 4;
  let mut x = DMatrix::<f64>::zeros(t, d);
  for i in 0..t {
    for j in 0..d {
      x[(i, j)] = ((i + j) as f64) * 0.5;
    }
  }
  let phi = DVector::<f64>::from_element(d, 1.0);
  // Non-uniform qinit so the column-validation passes — we need to
  // reach the actual EM loop to exercise the allocation behavior.
  let qinit = deterministic_qinit(t, s);
  // usize::MAX would have triggered a capacity-overflow panic in
  // the pre-fix code's `Vec::with_capacity(max_iters)`.
  let out = vbx_iterate(&x, &phi, &qinit, 0.07, 0.8, usize::MAX).expect("vbx_iterate");
  // Convergence should be extremely fast on this trivial input.
  assert!(
    out.elbo_trajectory.len() < 100,
    "expected fast convergence, got {} iterations",
    out.elbo_trajectory.len()
  );
}

/// Regression for the dead-column concern: a `qinit` with a speaker
/// column whose total mass is zero must be rejected at the entry,
/// not silently resurrected by the uniform-pi initialization.
///
/// Pre-fix: column 1 has zero mass but `pi[1] = 1/S = 0.5`, and the
/// first EM update gives gamma[*, 1] non-zero values, fabricating a
/// second speaker that wasn't in the initialization.
#[test]
fn vbx_rejects_qinit_with_dead_speaker_column() {
  let t = 5;
  let s = 2;
  let x = DMatrix::<f64>::zeros(t, 4);
  let phi = DVector::<f64>::from_element(4, 1.0);
  let mut qinit = DMatrix::<f64>::zeros(t, s);
  // All mass on column 0; column 1 is dead. Each row sums to 1.0,
  // so the row-validation passes — only the new column-validation
  // catches it.
  for tt in 0..t {
    qinit[(tt, 0)] = 1.0;
  }
  let result = vbx_iterate(&x, &phi, &qinit, 0.07, 0.8, 20);
  assert!(matches!(result, Err(Error::Shape(_))), "got {result:?}");
}

/// Realistic per-frame assignment: even rows favor speaker 0,
/// odd rows favor speaker 1. Both columns have at least one row
/// with mass ≥ 1/S, so both pass the per-row-max check. (Round 4's
/// "60/40 across all rows" pattern was a non-pyannote-style global
/// prior — round 8 rejects that case as not a valid per-frame
/// assignment, so the original test was reframed into this one.)
#[test]
fn vbx_accepts_qinit_with_alternating_column_assignment() {
  let t = 10;
  let s = 2;
  let d = 4;
  let x = DMatrix::<f64>::from_fn(t, d, |i, j| ((i + j) as f64) * 0.3);
  let phi = DVector::<f64>::from_element(d, 1.0);
  let mut qinit = DMatrix::<f64>::zeros(t, s);
  for tt in 0..t {
    if tt % 2 == 0 {
      qinit[(tt, 0)] = 0.95;
      qinit[(tt, 1)] = 0.05;
    } else {
      qinit[(tt, 0)] = 0.05;
      qinit[(tt, 1)] = 0.95;
    }
  }
  // Each column has at least one row with max=0.95, well above 1/S=0.5.
  let _out =
    vbx_iterate(&x, &phi, &qinit, 0.07, 0.8, 10).expect("alternating real columns must pass");
}

/// Codex round 9 [MEDIUM]: uniform qinit (every cell = 1/S) is a
/// VBx fixed point. With identical columns gamma_sum / invL / alpha
/// / log_p are symmetric across speakers, EM has no way to break
/// the symmetry, and the algorithm returns the same uniform input
/// as a "clustering result". The strict-greater per-row-max check
/// rejects this case at the boundary.
#[test]
fn vbx_rejects_uniform_qinit_for_s_gt_1() {
  let t = 8;
  let s = 4;
  let d = 4;
  let x = DMatrix::<f64>::from_fn(t, d, |i, j| ((i + j) as f64) * 0.2);
  let phi = DVector::<f64>::from_element(d, 1.0);
  let qinit = DMatrix::<f64>::from_element(t, s, 1.0 / s as f64);
  let result = vbx_iterate(&x, &phi, &qinit, 0.07, 0.8, 10);
  assert!(
    matches!(result, Err(Error::Shape(_))),
    "uniform qinit (per-row max = 1/S = {}) is a symmetry-trap \
     fixed point and must be rejected; got {result:?}",
    1.0 / s as f64
  );
}

/// Codex round 11 [HIGH]: a "near-dead spike" — column with a
/// single row at `1/S + ε` and all other rows at zero — passes
/// round-9's strict `> 1/S` check but has total mass barely above
/// the residue floor. With uniform pi initialization, EM could
/// resurrect this near-dead speaker on low-evidence inputs. The
/// tightened `> 1.5/S` threshold rejects it.
#[test]
fn vbx_rejects_qinit_with_near_dead_spike_column() {
  let t = 19;
  let s = 19;
  let d = 4;
  let x = DMatrix::<f64>::from_fn(t, d, |i, j| ((i + j) as f64) * 0.1);
  let phi = DVector::<f64>::from_element(d, 1.0);

  // First S-1 rows: pyannote-style softmax(7) "real" assignments
  // (one-hot peaked at speaker `tt`).
  let on_mass = (7.0_f64).exp() / ((7.0_f64).exp() + (s - 1) as f64);
  let off_mass = 1.0 / ((7.0_f64).exp() + (s - 1) as f64);
  let mut qinit = DMatrix::<f64>::zeros(t, s);
  for tt in 0..(s - 1) {
    for sj in 0..s {
      qinit[(tt, sj)] = if sj == tt { on_mass } else { off_mass };
    }
  }
  // Last row: speaker S-1 has just `1/S + ε` mass (the near-dead
  // spike). Other speakers in this row share the rest evenly.
  let one_over_s = 1.0 / s as f64;
  let near_dead = one_over_s + 1.0e-12;
  let other = (1.0 - near_dead) / (s - 1) as f64;
  for sj in 0..s {
    qinit[(s - 1, sj)] = if sj == s - 1 { near_dead } else { other };
  }

  // Test setup invariant: col S-1's per-row max is just above 1/S
  // but well below the new 1.5/S threshold.
  let col_max_near_dead = (0..t)
    .map(|tt| qinit[(tt, s - 1)])
    .fold(f64::NEG_INFINITY, f64::max);
  let new_floor = 1.5 / s as f64;
  assert!(
    col_max_near_dead > one_over_s,
    "test invariant: near-dead col_max = {col_max_near_dead:.4e} \
     must sit above the previous round-9 threshold (1/S = {one_over_s:.4e})"
  );
  assert!(
    col_max_near_dead < new_floor,
    "test invariant: near-dead col_max = {col_max_near_dead:.4e} \
     must sit below the new round-11 floor (1.5/S = {new_floor:.4e})"
  );

  let result = vbx_iterate(&x, &phi, &qinit, 0.07, 0.8, 20);
  assert!(
    matches!(result, Err(Error::Shape(_))),
    "near-dead spike column must be rejected; got {result:?}"
  );
}

/// Codex round 12 [HIGH]: round-11's `> 1.5/S` per-row-max
/// threshold still admits a "single-spike" column where one row
/// sits just above 1.5/S and everything else is zero — col_sum is
/// negligible (~ 1.5/S) and the column is barely supported. The
/// new combined check (per-row-max AND col_sum > 0.5) rejects this.
///
/// Note: this is a distinct attack from the round-11 case (which
/// used 1/S + ε, still below the new 1.5/S threshold). Round-12's
/// spike sits ABOVE 1.5/S to bypass the per-row-max check; the
/// col-sum floor catches it.
#[test]
fn vbx_rejects_qinit_with_single_spike_above_threshold() {
  let t = 19;
  let s = 19;
  let d = 4;
  let x = DMatrix::<f64>::from_fn(t, d, |i, j| ((i + j) as f64) * 0.1);
  let phi = DVector::<f64>::from_element(d, 1.0);

  // First S-1 rows: pyannote-style softmax(7) one-hot peaks
  // (real assignments for speakers 0..17).
  let on_mass = (7.0_f64).exp() / ((7.0_f64).exp() + (s - 1) as f64);
  let off_mass = 1.0 / ((7.0_f64).exp() + (s - 1) as f64);
  let mut qinit = DMatrix::<f64>::zeros(t, s);
  for tt in 0..(s - 1) {
    for sj in 0..s {
      qinit[(tt, sj)] = if sj == tt { on_mass } else { off_mass };
    }
  }
  // Last row: speaker S-1 gets a "spike" mass slightly above 1.5/S.
  // For S=19 this is 1.5/19 + ε ≈ 0.0789 + 0.001 = 0.0799.
  // The remaining row mass is split among other speakers.
  let one_over_s = 1.0 / s as f64;
  let spike_mass = 1.5 * one_over_s + 1.0e-3;
  let spike_other = (1.0 - spike_mass) / (s - 1) as f64;
  for sj in 0..s {
    qinit[(s - 1, sj)] = if sj == s - 1 { spike_mass } else { spike_other };
  }

  // Test setup invariants:
  let col_max_spike = (0..t)
    .map(|tt| qinit[(tt, s - 1)])
    .fold(f64::NEG_INFINITY, f64::max);
  let new_floor = 1.5 / s as f64;
  assert!(
    col_max_spike > new_floor,
    "test invariant: spike col_max = {col_max_spike:.4e} must \
     EXCEED the round-11 per-row-max floor (1.5/S = {new_floor:.4e}) \
     so this attack is distinct from the round-11 case"
  );
  let col_sum_spike: f64 = (0..t).map(|tt| qinit[(tt, s - 1)]).sum();
  assert!(
    col_sum_spike < 0.5,
    "test invariant: spike col_sum = {col_sum_spike:.4e} must be \
     below the new col-sum floor (0.5) so the round-12 check fires"
  );

  let result = vbx_iterate(&x, &phi, &qinit, 0.07, 0.8, 20);
  assert!(
    matches!(result, Err(Error::Shape(_))),
    "single-spike column above per-row-max threshold but with \
     negligible total support must be rejected; got {result:?}"
  );
}

/// S=1 is a degenerate case (single speaker) — `qinit` is forced to
/// be all 1.0 by the row-sum invariant, and there is no symmetry to
/// break with one column. The check skips the per-row-max test for
/// S=1 to avoid over-rejecting this corner case.
#[test]
fn vbx_accepts_single_speaker_qinit() {
  let t = 5;
  let s = 1;
  let d = 4;
  let x = DMatrix::<f64>::from_fn(t, d, |i, j| ((i + j) as f64) * 0.1);
  let phi = DVector::<f64>::from_element(d, 1.0);
  let qinit = DMatrix::<f64>::from_element(t, s, 1.0);
  let out =
    vbx_iterate(&x, &phi, &qinit, 0.07, 0.8, 10).expect("S=1 single-speaker qinit must pass");
  // With S=1 there is only one cluster; pi[0] should be 1.0.
  assert!((out.pi[0] - 1.0).abs() < 1e-12, "pi[0] = {}", out.pi[0]);
}

// ── Tighter qinit column-mass + zero-D rejection (Codex review MEDIUM round 6 of Phase 2) ─
//
// Round 4's exact-zero check let through "near-dead" columns with
// tiny positive mass. Round 6 raised it to a fixed column-sum floor
// of 0.5. Round 8 (Codex review HIGH) replaced that with a
// T-invariant per-row-max criterion: column max must reach 1/S
// (the uniform baseline). The column-sum floor failed open at
// long T because residue mass scales with T, while per-row max
// stays bounded by `1/(exp(7) + S - 1) < 1/S` regardless of T.

/// Per-row residue 0.02 in column 1 (well below 1/S = 0.5 for S=2).
/// Round 4 admitted this; round 6 still rejected via column-sum
/// (0.1 < 0.5); round 8 rejects via per-row-max (0.02 < 0.5).
#[test]
fn vbx_rejects_qinit_with_tiny_positive_column_mass() {
  let t = 5;
  let s = 2;
  let x = DMatrix::<f64>::zeros(t, 4);
  let phi = DVector::<f64>::from_element(4, 1.0);
  let mut qinit = DMatrix::<f64>::zeros(t, s);
  for tt in 0..t {
    qinit[(tt, 0)] = 0.98;
    qinit[(tt, 1)] = 0.02;
  }
  let result = vbx_iterate(&x, &phi, &qinit, 0.07, 0.8, 20);
  assert!(matches!(result, Err(Error::Shape(_))), "got {result:?}");
}

// ── Codex review HIGH round 8: T-invariant residue rejection ─

/// Regression for the long-T residue case. Simulates pyannote
/// softmax(7) output where speaker 0 is "on" in every row and
/// speakers 1..S are pure smoothing residue. With T=1000, the
/// residue column sum is ~0.897 — passes the previous fixed-0.5
/// column-sum floor — but its per-row max is ~9e-4 ≪ 1/S = 0.0526,
/// so the new T-invariant per-row-max check correctly rejects.
#[test]
fn vbx_rejects_smoothed_residue_column_at_long_t() {
  let t = 1000;
  let s = 19;
  let d = 4;
  let x = DMatrix::<f64>::from_fn(t, d, |i, j| ((i + j) as f64) * 0.1);
  let phi = DVector::<f64>::from_element(d, 1.0);

  // Pyannote softmax(7) with hard label always 0:
  //   on-mass  ≈ exp(7)/(exp(7) + 18) ≈ 0.984
  //   off-mass ≈ exp(0)/(exp(7) + 18) ≈ 8.97e-4
  let on_mass = (7.0_f64).exp() / ((7.0_f64).exp() + (s - 1) as f64);
  let off_mass = 1.0_f64 / ((7.0_f64).exp() + (s - 1) as f64);
  let mut qinit = DMatrix::<f64>::zeros(t, s);
  for tt in 0..t {
    qinit[(tt, 0)] = on_mass;
    for sj in 1..s {
      qinit[(tt, sj)] = off_mass;
    }
  }

  // Test setup invariants — verify this regression actually exercises
  // the failure-mode the new check is designed to catch.
  let col_sum_residue: f64 = (0..t).map(|tt| qinit[(tt, 1)]).sum();
  assert!(
    col_sum_residue > 0.5,
    "test setup invariant: long-T residue col_sum = {col_sum_residue:.3} \
     must exceed the previous fixed 0.5 floor (else this regression \
     wouldn't cover the case the old floor missed)"
  );
  let per_row_max_residue: f64 = (0..t)
    .map(|tt| qinit[(tt, 1)])
    .fold(f64::NEG_INFINITY, f64::max);
  let one_over_s = 1.0 / s as f64;
  assert!(
    per_row_max_residue < one_over_s,
    "test setup invariant: residue per-row max = {per_row_max_residue:.3e} \
     must be below 1/S = {one_over_s:.4} for the new check to fire"
  );

  let result = vbx_iterate(&x, &phi, &qinit, 0.07, 0.8, 20);
  assert!(
    matches!(result, Err(Error::Shape(_))),
    "long-T smoothed-residue column must be rejected by the per-row-max \
     check (T-invariant); got {result:?}"
  );
}

#[test]
fn vbx_rejects_zero_feature_dimension() {
  // D=0: x has 5 rows × 0 columns; phi is empty.
  let x = DMatrix::<f64>::zeros(5, 0);
  let phi = DVector::<f64>::zeros(0);
  let qinit = DMatrix::<f64>::from_element(5, 2, 0.5);
  let result = vbx_iterate(&x, &phi, &qinit, 0.07, 0.8, 20);
  assert!(matches!(result, Err(Error::Shape(_))), "got {result:?}");
}

/// D=0 must be rejected even at `max_iters = 0`, so the boundary
/// validation runs before the algorithm has a chance to return
/// uniform-prior gamma/pi as if it had clustered.
#[test]
fn vbx_rejects_zero_feature_dimension_even_at_max_iters_zero() {
  let x = DMatrix::<f64>::zeros(5, 0);
  let phi = DVector::<f64>::zeros(0);
  let qinit = DMatrix::<f64>::from_element(5, 2, 0.5);
  let result = vbx_iterate(&x, &phi, &qinit, 0.07, 0.8, 0);
  assert!(matches!(result, Err(Error::Shape(_))), "got {result:?}");
}

// ── X / Phi non-finite hardening (Codex review MEDIUM round 5 of Phase 2) ─
//
// The previous boundary accepted `+inf` Phi (the check used
// `is_nan()` only) and didn't validate X at all. Either case
// poisons G/rho silently — caught downstream as a generic
// `NonFinite("ELBO")` if max_iters > 0, or returned as Ok with the
// unvalidated qinit at max_iters = 0. Tightening to `is_finite()`
// + a leading X scan rejects upstream-corrupted PLDA inputs at the
// boundary with a clear typed error.

#[test]
fn vbx_rejects_phi_with_pos_inf() {
  let x = DMatrix::<f64>::zeros(5, 4);
  let mut phi = DVector::<f64>::from_element(4, 1.0);
  phi[1] = f64::INFINITY;
  let qinit = DMatrix::<f64>::from_element(5, 2, 0.5);
  let result = vbx_iterate(&x, &phi, &qinit, 0.07, 0.8, 20);
  assert!(
    matches!(result, Err(Error::NonPositivePhi(p, 1)) if p.is_infinite() && p > 0.0),
    "got {result:?}"
  );
}

#[test]
fn vbx_rejects_phi_with_nan() {
  let x = DMatrix::<f64>::zeros(5, 4);
  let mut phi = DVector::<f64>::from_element(4, 1.0);
  phi[3] = f64::NAN;
  let qinit = DMatrix::<f64>::from_element(5, 2, 0.5);
  let result = vbx_iterate(&x, &phi, &qinit, 0.07, 0.8, 20);
  assert!(
    matches!(result, Err(Error::NonPositivePhi(p, 3)) if p.is_nan()),
    "got {result:?}"
  );
}

#[test]
fn vbx_rejects_x_with_nan() {
  let mut x = DMatrix::<f64>::zeros(5, 4);
  x[(2, 1)] = f64::NAN;
  let phi = DVector::<f64>::from_element(4, 1.0);
  let qinit = DMatrix::<f64>::from_element(5, 2, 0.5);
  let result = vbx_iterate(&x, &phi, &qinit, 0.07, 0.8, 20);
  assert!(
    matches!(result, Err(Error::NonFinite("x"))),
    "got {result:?}"
  );
}

#[test]
fn vbx_rejects_x_with_pos_inf() {
  let mut x = DMatrix::<f64>::zeros(5, 4);
  x[(0, 0)] = f64::INFINITY;
  let phi = DVector::<f64>::from_element(4, 1.0);
  let qinit = DMatrix::<f64>::from_element(5, 2, 0.5);
  let result = vbx_iterate(&x, &phi, &qinit, 0.07, 0.8, 20);
  assert!(
    matches!(result, Err(Error::NonFinite("x"))),
    "got {result:?}"
  );
}

#[test]
fn vbx_rejects_x_with_neg_inf() {
  let mut x = DMatrix::<f64>::zeros(5, 4);
  x[(4, 3)] = f64::NEG_INFINITY;
  let phi = DVector::<f64>::from_element(4, 1.0);
  let qinit = DMatrix::<f64>::from_element(5, 2, 0.5);
  let result = vbx_iterate(&x, &phi, &qinit, 0.07, 0.8, 20);
  assert!(
    matches!(result, Err(Error::NonFinite("x"))),
    "got {result:?}"
  );
}

/// Codex's specific concern: at `max_iters = 0` the loop never
/// runs, so the generic NaN-intermediate guard never fires. Boundary
/// validation must catch invalid inputs even when no iterations run.
#[test]
fn vbx_rejects_invalid_x_even_with_max_iters_zero() {
  let mut x = DMatrix::<f64>::zeros(5, 4);
  x[(2, 2)] = f64::NAN;
  let phi = DVector::<f64>::from_element(4, 1.0);
  let qinit = DMatrix::<f64>::from_element(5, 2, 0.5);
  let result = vbx_iterate(&x, &phi, &qinit, 0.07, 0.8, 0);
  assert!(
    matches!(result, Err(Error::NonFinite("x"))),
    "boundary validation must run even at max_iters=0; got {result:?}"
  );
}

#[test]
fn vbx_rejects_invalid_phi_even_with_max_iters_zero() {
  let x = DMatrix::<f64>::zeros(5, 4);
  let mut phi = DVector::<f64>::from_element(4, 1.0);
  phi[2] = f64::INFINITY;
  let qinit = DMatrix::<f64>::from_element(5, 2, 0.5);
  let result = vbx_iterate(&x, &phi, &qinit, 0.07, 0.8, 0);
  assert!(
    matches!(result, Err(Error::NonPositivePhi(p, 2)) if p.is_infinite()),
    "boundary validation must run even at max_iters=0; got {result:?}"
  );
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

// Most classifier tests use small-magnitude `prev`/`elbo` so the
// scale-aware regression band collapses to ~atol (~1e-9). Two tests
// near the bottom exercise the band at large magnitude (Codex round 3).

#[test]
fn classify_elbo_step_continues_on_large_positive_delta() {
  assert_eq!(
    classify_elbo_step(0.5, -1.5, -1.0, 1.0e-4),
    ElboStep::Continue
  );
}

#[test]
fn classify_elbo_step_converges_on_small_positive_delta() {
  assert_eq!(
    classify_elbo_step(1.0e-5, -1.00001, -1.0, 1.0e-4),
    ElboStep::Converged
  );
}

#[test]
fn classify_elbo_step_converges_on_tiny_negative_delta_within_tolerance() {
  // Delta in float-roundoff regime — well inside the band.
  assert_eq!(
    classify_elbo_step(-1.0e-12, -1.0, -1.0 - 1.0e-12, 1.0e-4),
    ElboStep::Converged
  );
}

#[test]
fn classify_elbo_step_regresses_on_large_negative_delta() {
  match classify_elbo_step(-1.0e-4, -1.0, -1.0001, 1.0e-4) {
    ElboStep::Regressed(d) => assert_eq!(d, -1.0e-4),
    other => panic!("expected Regressed, got {other:?}"),
  }
}

#[test]
fn classify_elbo_step_regresses_just_outside_tolerance() {
  // |elbo|=1.0 → tol = 1e-9 + 1e-9*1 = 2e-9. delta=-1e-8 is 5x outside.
  match classify_elbo_step(-1.0e-8, -1.0, -1.00000001, 1.0e-4) {
    ElboStep::Regressed(d) => assert_eq!(d, -1.0e-8),
    other => panic!("expected Regressed, got {other:?}"),
  }
}

#[test]
fn classify_elbo_step_zero_delta_is_converged() {
  // Exactly zero — flat ELBO, treat as converged.
  assert_eq!(
    classify_elbo_step(0.0, -1.0, -1.0, 1.0e-4),
    ElboStep::Converged
  );
}

// ── Scale-aware regression band (Codex review MEDIUM round 3 of Phase 2) ─
//
// ELBO is an accumulated sum over T * S * D matrix entries plus T
// per-frame terms; float roundoff scales with the magnitude of the
// ELBO itself. The previous absolute `-1e-9` regression tolerance
// (calibrated against the |ELBO|≈2700 captured fixture) errored out
// on numerically awkward but otherwise valid inputs. The
// `atol + rtol * max(|prev|, |elbo|)` band absorbs that.

/// Regression for the Codex round-3 reproduction case. Final delta
/// of `-2.47e-8` at |ELBO| ≈ 2700 — outside an absolute `1e-9` band
/// but well inside the scale-aware band (1e-9 + 1e-9 * 2700 ≈ 2.7e-6).
#[test]
fn classify_elbo_step_absorbs_relative_float_roundoff_at_large_magnitude() {
  let prev = -2700.0_f64;
  let delta = -2.47e-8_f64;
  let elbo = prev + delta;
  assert_eq!(
    classify_elbo_step(delta, prev, elbo, 1.0e-4),
    ElboStep::Converged,
    "scale-aware band must absorb a delta the absolute tolerance \
     would have rejected — Codex's round-3 reproduction case"
  );
}

/// Even at large magnitude, materially-large negative drops still
/// surface as `Regressed`. Tests the upper edge of the scale-aware band.
#[test]
fn classify_elbo_step_still_rejects_material_regression_at_large_magnitude() {
  let prev = -2700.0_f64;
  // Band at this magnitude is ~2.7e-6; a -1e-3 drop is ~370× outside.
  let delta = -1.0e-3_f64;
  let elbo = prev + delta;
  match classify_elbo_step(delta, prev, elbo, 1.0e-4) {
    ElboStep::Regressed(d) => assert_eq!(d, delta),
    other => panic!("expected Regressed at large magnitude, got {other:?}"),
  }
}

// ── Stop reason: converged vs max-iters-reached (Codex review MEDIUM round 10) ─
//
// Codex round 10 pointed out that `vbx_iterate` returned the same
// shape of `Ok` for two semantically distinct cases:
//   - Converged within max_iters (early break on ElboStep::Converged)
//   - max_iters reached without ever converging (loop falls through)
// Both could have `elbo_trajectory.len() == max_iters` (when
// convergence happens on the very last allowed iteration). Callers
// could not reliably distinguish the two, so an unconverged
// posterior would silently flow into downstream centroid/label
// assignment. `VbxOutput::stop_reason` makes the distinction
// observable at the type level.

use crate::vbx::StopReason;

/// `max_iters = 1`: the convergence check requires `ii > 0`, so a
/// 1-iter run can never fire the `Converged` branch. The loop ends
/// naturally and `stop_reason == MaxIterationsReached`.
#[test]
fn vbx_reports_max_iterations_reached_when_cap_is_one() {
  let t = 6;
  let s = 2;
  let d = 4;
  let mut x = DMatrix::<f64>::zeros(t, d);
  for i in 0..t {
    for j in 0..d {
      x[(i, j)] = ((i + j) as f64) * 0.5;
    }
  }
  let phi = DVector::<f64>::from_element(d, 1.0);
  let qinit = deterministic_qinit(t, s);
  let out = vbx_iterate(&x, &phi, &qinit, 0.07, 0.8, 1).expect("vbx_iterate");
  assert_eq!(
    out.stop_reason,
    StopReason::MaxIterationsReached,
    "max_iters=1 cannot fire convergence (check requires ii > 0)"
  );
  assert_eq!(out.elbo_trajectory.len(), 1, "ran exactly 1 iteration");
}

/// On a small input that converges quickly, the same call with a
/// generous `max_iters` should report `Converged`. Together with
/// the previous test this proves callers can distinguish the two
/// stop reasons.
#[test]
fn vbx_reports_converged_on_easy_input() {
  let t = 6;
  let s = 2;
  let d = 4;
  let mut x = DMatrix::<f64>::zeros(t, d);
  for i in 0..t {
    for j in 0..d {
      x[(i, j)] = ((i + j) as f64) * 0.5;
    }
  }
  let phi = DVector::<f64>::from_element(d, 1.0);
  let qinit = deterministic_qinit(t, s);
  let out = vbx_iterate(&x, &phi, &qinit, 0.07, 0.8, 50).expect("vbx_iterate");
  assert_eq!(
    out.stop_reason,
    StopReason::Converged,
    "easy input with generous cap must converge before exhaustion; \
     ran {} iterations",
    out.elbo_trajectory.len()
  );
  // Convergence on a trivial input is fast (well below the cap).
  assert!(
    out.elbo_trajectory.len() < 50,
    "expected early convergence, ran {} iters",
    out.elbo_trajectory.len()
  );
}
