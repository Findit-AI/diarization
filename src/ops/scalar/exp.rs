//! Scalar in-place `exp` over an f64 slice.
//!
//! Reference / fallback implementation. Defers to `f64::exp` element by
//! element. The arch backends (NEON in Step 4) implement vectorized
//! polynomial approximations and dispatchers route here when no SIMD
//! backend is available for the host arch.

/// In-place element-wise `exp` over an f64 slice: `x[i] = x[i].exp()`.
#[inline]
#[allow(dead_code)] // Step 4: scaffolded; VBx wiring reverted on Apple Silicon (libm exp wins).
pub fn exp_inplace(x: &mut [f64]) {
  for v in x.iter_mut() {
    *v = v.exp();
  }
}
