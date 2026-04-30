//! `logsumexp_row` dispatcher.

use crate::ops::scalar;

/// `ln(Σ exp(row[i]))` via the max-shift trick. Step 2: scalar-only.
#[inline]
pub fn logsumexp_row(row: &[f64], _use_simd: bool) -> f64 {
  scalar::logsumexp_row(row)
}
