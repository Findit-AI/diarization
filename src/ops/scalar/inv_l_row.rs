//! Scalar VBx Eq. 17 row.

/// Compute one row of VBx's `invL`:
/// `out[d] = 1 / (1 + fa_over_fb * gamma_sum_s * phi[d])`.
///
/// Used by `vbx_iterate` per speaker `s` per EM iteration. With the
/// outer-product structure folded out (`fa_over_fb * gamma_sum_s` is
/// a per-speaker scalar, multiplied by per-dim `phi[d]`), this
/// reduces to one scalar-vector mul + one constant add + one reciprocal
/// per element, which is straight SIMD territory.
///
/// # Panics (debug only)
///
/// Debug asserts on `out.len() == phi.len()`.
#[inline]
#[allow(dead_code)] // Step 2: scaffolded; vbx will route through this in Step 3.
pub fn inv_l_row(out: &mut [f64], fa_over_fb: f64, gamma_sum_s: f64, phi: &[f64]) {
  debug_assert_eq!(out.len(), phi.len(), "inv_l_row: length mismatch");
  let scale = fa_over_fb * gamma_sum_s;
  for i in 0..out.len() {
    out[i] = 1.0 / (1.0 + scale * phi[i]);
  }
}
