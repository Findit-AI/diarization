//! Scalar f64 dot product.

/// Inner product of two equal-length f64 slices: `Σ a[i] * b[i]`.
///
/// # Panics (debug only)
///
/// Debug asserts on `a.len() == b.len()`. Release builds trust the
/// caller — SIMD backends in `arch::*` rely on the same precondition.
#[inline]
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
  debug_assert_eq!(a.len(), b.len(), "dot: length mismatch");
  let mut acc = 0.0;
  for i in 0..a.len() {
    acc += a[i] * b[i];
  }
  acc
}
