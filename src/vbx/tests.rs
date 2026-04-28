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
  let m = DMatrix::<f64>::from_row_slice(2, 3, &[
    1.0, 2.0, 3.0,
    -100.0, -101.0, -102.0,
  ]);
  let lse = logsumexp_rows(&m);
  assert!((lse[0] - 3.40760596).abs() < 1e-8, "row0: {}", lse[0]);
  assert!((lse[1] - (-99.592_394_035_555_61)).abs() < 1e-10, "row1: {}", lse[1]);
}

/// All -inf row → -inf result (matches scipy behavior).
#[test]
fn logsumexp_rows_all_neg_inf_returns_neg_inf() {
  let m = DMatrix::<f64>::from_row_slice(1, 3, &[
    f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY,
  ]);
  let lse = logsumexp_rows(&m);
  assert!(lse[0].is_infinite() && lse[0] < 0.0, "got {}", lse[0]);
}
