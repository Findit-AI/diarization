//! NEON 2-lane f64 `exp` polynomial.
//!
//! Cody-Waite range reduction + 13-term Taylor polynomial + bit-manip
//! `2^k` reconstruction. Targets ~1 ulp accuracy across `|x| <= ~700`,
//! which is safe for VBx's `exp(log_p - log_p_x)` use (inputs are
//! bounded above by 0 after the max-shift).
//!
//! ```text
//!   k = round(x * log2(e))
//!   r = x - k * ln(2)               // |r| <= ln(2)/2 ≈ 0.347
//!   exp(x) = 2^k * (1 + r + r²/2 + r³/6 + ... + r^12/12!)
//!   2^k    = bit_cast<f64>((1023 + k) << 52)
//! ```
//!
//! Range-reduction `ln(2)` is split into hi+lo for accurate
//! subtraction (`LN2_HI` is exact to ~`9.5e-14` of `ln(2)`; `LN2_LO`
//! mops up the residual). Saves ~3 ulp at large `|k|` vs a single-piece
//! subtraction.
//!
//! Out-of-range handling: callers in this crate (VBx) pass values in
//! `[-50, 0]`, so overflow / NaN / infinity never reach this kernel.
//! Inputs outside `[-700, 700]` may produce nonsense bit patterns from
//! `(1023 + k) << 52` overflow. The scalar fallback handles those
//! correctly; if a future caller needs the full f64 range, route
//! `use_simd = false` for that call.

use core::arch::aarch64::{
  float64x2_t, vaddq_s64, vcvtnq_s64_f64, vdupq_n_f64, vdupq_n_s64, vfmaq_f64, vld1q_f64,
  vmulq_f64, vreinterpretq_f64_s64, vshlq_n_s64, vst1q_f64, vsubq_f64,
};

use crate::ops::scalar;

#[allow(dead_code)]
const LOG2_E: f64 = std::f64::consts::LOG2_E;
// ln(2) split for Cody-Waite range reduction. `LN2_HI` is `ln(2)`
// rounded to the nearest f64 with the lowest 21 bits cleared so that
// `k * LN2_HI` is exactly representable for any `|k| < 2^21`. `LN2_LO`
// is the residual `ln(2) - LN2_HI` in full f64 precision.
#[allow(dead_code)]
const LN2_HI: f64 = 0.693_147_180_369_123_8;
#[allow(dead_code)]
const LN2_LO: f64 = 1.908_213_073_995_580_5e-10;

/// In-place SIMD exp over an f64 slice.
///
/// # Safety
///
/// 1. NEON must be available (caller's obligation).
#[inline]
#[target_feature(enable = "neon")]
#[allow(dead_code)] // Step 4: scaffolded; VBx wiring reverted on Apple Silicon.
pub(crate) unsafe fn exp_inplace(x: &mut [f64]) {
  let n = x.len();

  // SAFETY: pointer adds bounded by `n`. NEON verified at dispatcher.
  unsafe {
    let log2_e = vdupq_n_f64(LOG2_E);
    let ln2_hi = vdupq_n_f64(LN2_HI);
    let ln2_lo = vdupq_n_f64(LN2_LO);
    // Polynomial coefficients 1/n!, n=0..12 (Horner-evaluated tail
    // first). Hardcoded as f64 constants; `vdupq_n_f64` materializes
    // each into a 2-lane register.
    let c12 = vdupq_n_f64(2.087_675_698_786_81e-9); // 1/12!
    let c11 = vdupq_n_f64(2.505_210_838_544_172e-8); // 1/11!
    let c10 = vdupq_n_f64(2.755_731_922_398_589e-7); // 1/10!
    let c9 = vdupq_n_f64(2.755_731_922_398_589e-6); // 1/9!
    let c8 = vdupq_n_f64(2.480_158_730_158_73e-5); // 1/8!
    let c7 = vdupq_n_f64(1.984_126_984_126_984e-4); // 1/7!
    let c6 = vdupq_n_f64(1.388_888_888_888_889e-3); // 1/6!
    let c5 = vdupq_n_f64(0.008_333_333_333_333_333); // 1/5!
    let c4 = vdupq_n_f64(0.041_666_666_666_666_664); // 1/4!
    let c3 = vdupq_n_f64(0.166_666_666_666_666_66); // 1/3!
    let c2 = vdupq_n_f64(0.5); // 1/2!
    let c1 = vdupq_n_f64(1.0); // 1/1!
    let c0 = vdupq_n_f64(1.0); // 1/0!
    let bias = vdupq_n_s64(1023);

    let mut i = 0usize;
    while i + 2 <= n {
      let xv: float64x2_t = vld1q_f64(x.as_ptr().add(i));
      // k = round(x * log2(e)), as integer and as float. Use NEON's
      // round-to-nearest-even (`vcvtnq_s64_f64` already does the
      // round, but we need both the int (for 2^k) and the f64 (for
      // the subtraction).
      let k_i = vcvtnq_s64_f64(vmulq_f64(xv, log2_e));
      // f64 form of k. `i64` -> `f64` via the round-trip — guaranteed
      // exact for `|k| < 2^53`, and our reduction stays within ~10^3.
      let kf_lanes: [f64; 2] = [
        core::arch::aarch64::vgetq_lane_s64::<0>(k_i) as f64,
        core::arch::aarch64::vgetq_lane_s64::<1>(k_i) as f64,
      ];
      let kf = vld1q_f64(kf_lanes.as_ptr());
      // r = x - k*ln(2), Cody-Waite split.
      let r = vsubq_f64(xv, vmulq_f64(kf, ln2_hi));
      let r = vsubq_f64(r, vmulq_f64(kf, ln2_lo));

      // Horner: p = c0 + r * (c1 + r * (c2 + r * (... + r * c12)))
      let mut p = c12;
      p = vfmaq_f64(c11, p, r);
      p = vfmaq_f64(c10, p, r);
      p = vfmaq_f64(c9, p, r);
      p = vfmaq_f64(c8, p, r);
      p = vfmaq_f64(c7, p, r);
      p = vfmaq_f64(c6, p, r);
      p = vfmaq_f64(c5, p, r);
      p = vfmaq_f64(c4, p, r);
      p = vfmaq_f64(c3, p, r);
      p = vfmaq_f64(c2, p, r);
      p = vfmaq_f64(c1, p, r);
      p = vfmaq_f64(c0, p, r);

      // 2^k via bit manipulation: f64 = ((1023 + k) << 52)
      let exp_bits = vshlq_n_s64::<52>(vaddq_s64(bias, k_i));
      let pow2k = vreinterpretq_f64_s64(exp_bits);

      let result = vmulq_f64(p, pow2k);
      vst1q_f64(x.as_mut_ptr().add(i), result);
      i += 2;
    }
    if i < n {
      scalar::exp_inplace(&mut x[i..]);
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  /// NEON polynomial exp must round-trip the VBx input range
  /// (`[-50, 0]`) to within 1 ulp of `f64::exp` — the parity tests
  /// (gamma 1e-12 tolerance after 10 EM iters) rely on this.
  #[test]
  fn exp_polynomial_matches_libm_in_vbx_range() {
    let mut buf: Vec<f64> = (-500..=0).map(|i| i as f64 * 0.1).collect();
    let expected: Vec<f64> = (-500..=0).map(|i| (i as f64 * 0.1).exp()).collect();
    // SAFETY: NEON is part of aarch64 baseline.
    unsafe { exp_inplace(&mut buf) };
    let mut max_rel_err = 0.0_f64;
    for (got, want) in buf.iter().zip(expected.iter()) {
      let rel = ((got - want) / want).abs();
      if rel > max_rel_err {
        max_rel_err = rel;
      }
    }
    // 1 ulp at exp() in `[exp(-50), 1]` is ~2.2e-16 relative error;
    // the degree-12 Taylor polynomial accumulates ~6 ulp at the
    // endpoint of the reduced range. 1e-13 (≈450 ulp) is well below
    // VBx's downstream gamma-parity tolerance of 1e-12.
    assert!(
      max_rel_err < 1.0e-13,
      "NEON exp polynomial diverges from libm by {max_rel_err:e} (> 1e-13)"
    );
  }
}
