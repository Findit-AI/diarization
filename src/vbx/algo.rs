//! VBx variational EM iterations.

use crate::vbx::error::Error;

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
