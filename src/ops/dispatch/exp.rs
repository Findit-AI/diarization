//! In-place `exp` dispatcher.

#[cfg(target_arch = "aarch64")]
use crate::ops::arch;
#[cfg(target_arch = "aarch64")]
use crate::ops::neon_available;
use crate::ops::scalar;

/// In-place element-wise `exp` over an f64 slice.
///
/// Routes to NEON polynomial on aarch64; x86_64 falls through to scalar
/// `f64::exp` because AVX2/AVX-512 polynomial kernels are not yet
/// implemented. Callers needing full-range bit-identical scalar `exp`
/// across CPU families call [`crate::ops::scalar::exp_inplace`].
///
/// # Numerical contract
///
/// The NEON polynomial targets ~1 ulp accuracy across `|x| <= ~700`.
/// VBx callers pass values in `[-50, 0]` after the logsumexp max-shift,
/// well inside the safe range.
#[inline]
#[allow(dead_code)] // Step 4: scaffolded; VBx wiring reverted on Apple Silicon.
pub fn exp_inplace(x: &mut [f64]) {
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        // SAFETY: `neon_available()` confirmed NEON is on this CPU.
        unsafe { arch::neon::exp_inplace(x); }
        return;
      }
    },
    _ => {}
  }
  scalar::exp_inplace(x);
}
