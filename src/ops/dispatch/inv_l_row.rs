//! VBx Eq. 17 row dispatcher.

use crate::ops::scalar;

/// `out[d] = 1 / (1 + fa_over_fb * gamma_sum_s * phi[d])`.
/// Step 2: scalar-only.
#[inline]
#[allow(dead_code)] // Step 2: scaffolded; vbx will route through this in Step 3.
pub fn inv_l_row(out: &mut [f64], fa_over_fb: f64, gamma_sum_s: f64, phi: &[f64], _use_simd: bool) {
  scalar::inv_l_row(out, fa_over_fb, gamma_sum_s, phi);
}
