//! VBx variational EM iterations.

use crate::vbx::error::Error;
use nalgebra::{DMatrix, DVector};

/// Output of [`vbx_iterate`].
#[derive(Debug, Clone)]
pub struct VbxOutput {
  /// Final responsibilities, shape `(T, S)`.
  pub gamma: nalgebra::DMatrix<f64>,
  /// Final speaker priors, shape `(S,)`. Sums to 1.0.
  pub pi: nalgebra::DVector<f64>,
  /// ELBO at each iteration (length ≤ `max_iters`).
  pub elbo_trajectory: Vec<f64>,
}

/// Row-wise `logsumexp` (numerically stable). For each row `r`:
///
/// ```text
/// out[r] = log(sum_j exp(m[r, j] - max_j m[r, j])) + max_j m[r, j]
/// ```
///
/// Matches `scipy.special.logsumexp(m, axis=-1)` modulo float roundoff.
/// An all-`-inf` row produces `-inf` (the shift trick is bypassed
/// because subtracting `-inf` from `-inf` yields `NaN`).
pub(super) fn logsumexp_rows(m: &DMatrix<f64>) -> DVector<f64> {
  let (rows, cols) = m.shape();
  let mut out = DVector::<f64>::zeros(rows);
  for r in 0..rows {
    let row = m.row(r);
    // Find max for stability shift.
    let mut max = f64::NEG_INFINITY;
    for c in 0..cols {
      let v = row[c];
      if v > max {
        max = v;
      }
    }
    if max == f64::NEG_INFINITY {
      // All -inf row → result is -inf (matches scipy).
      out[r] = f64::NEG_INFINITY;
      continue;
    }
    let mut sum_exp = 0.0;
    for c in 0..cols {
      sum_exp += (row[c] - max).exp();
    }
    out[r] = sum_exp.ln() + max;
  }
  out
}

/// Placeholder; filled in Task 3.
pub fn vbx_iterate(
  _x: &nalgebra::DMatrix<f64>,
  _phi: &nalgebra::DVector<f64>,
  _qinit: &nalgebra::DMatrix<f64>,
  _fa: f64,
  _fb: f64,
  _max_iters: usize,
) -> Result<VbxOutput, Error> {
  Err(Error::Shape("not yet implemented"))
}
