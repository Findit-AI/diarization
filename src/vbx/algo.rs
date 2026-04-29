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

/// Per-iteration ELBO regression beyond which the algorithm errors
/// rather than treating the change as float roundoff. Empirically
/// the captured Phase-0 trajectory's smallest *positive* delta is
/// ~1.1e-4 (last meaningful iteration before convergence at
/// epsilon=1e-4), so any negative delta worse than `-1e-9` is well
/// outside float-roundoff territory. Codex review MEDIUM round 2.
const ELBO_REGRESSION_TOLERANCE: f64 = 1.0e-9;

/// Outcome of comparing one EM iteration's ELBO against the previous.
#[derive(Debug, PartialEq)]
pub(super) enum ElboStep {
  /// Improvement >= `epsilon` — keep iterating.
  Continue,
  /// Improvement < `epsilon` (including tiny negative deltas within
  /// `ELBO_REGRESSION_TOLERANCE`) — converged, exit cleanly.
  Converged,
  /// Negative delta beyond `ELBO_REGRESSION_TOLERANCE` — VB EM's
  /// monotonicity invariant is violated. Carries the offending delta.
  Regressed(f64),
}

/// Classify an ELBO step into the three convergence regimes.
///
/// Pyannote's `vbx.py:133-136` uses `if ELBO - prev < epsilon: break`
/// for both small-positive convergence AND any negative regression,
/// printing a warning for the regression case. The Rust port treats a
/// regression beyond `ELBO_REGRESSION_TOLERANCE` as an error instead
/// (no print mechanism, and downstream clustering should not silently
/// consume a regressed posterior). Tiny negative deltas inside the
/// tolerance are float-roundoff and treated as `Converged`.
pub(super) fn classify_elbo_step(delta: f64, epsilon: f64) -> ElboStep {
  if delta < -ELBO_REGRESSION_TOLERANCE {
    ElboStep::Regressed(delta)
  } else if delta < epsilon {
    ElboStep::Converged
  } else {
    ElboStep::Continue
  }
}

/// Row-wise `logsumexp` (numerically stable). For each row `r`:
///
/// ```text
/// out[r] = log(sum_j exp(m[r, j] - max_j m[r, j])) + max_j m[r, j]
/// ```
///
/// Matches `scipy.special.logsumexp(m, axis=-1)` modulo float roundoff
/// for finite or `-inf` rows. An all-NaN row returns `-inf` here vs
/// `NaN` in scipy — VBx callers reject NaN inputs upstream via
/// `Error::NonFinite`, so this divergence is unreachable in production.
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

/// Variational Bayes HMM speaker clustering (the VBx EM core).
///
/// Mirrors `pyannote.audio.utils.vbx.VBx` (`utils/vbx.py:27-137` in
/// pyannote.audio 4.0.4). Inputs:
///
/// - `x`: `(T, D)` post-PLDA features (output of
///   `dia::plda::PldaTransform::project()` stacked into a matrix).
/// - `phi`: `(D,)` eigenvalue diagonal (output of
///   `dia::plda::PldaTransform::phi()`). Must be strictly positive.
/// - `qinit`: `(T, S)` initial responsibility matrix. Each row should
///   sum to 1 (the algorithm doesn't enforce this — pyannote's caller
///   pre-softmaxes a smoothed one-hot AHC initialization).
/// - `fa`: sufficient-statistics scale (community-1 uses 0.07).
/// - `fb`: speaker regularization (community-1 uses 0.8).
/// - `max_iters`: hard iteration cap. Inner convergence triggers early
///   exit when `ELBO_i - ELBO_{i-1} < 1e-4`.
///
/// Returns final `gamma`, `pi`, and the ELBO trajectory (one entry per
/// iteration actually run; length ≤ `max_iters`).
///
/// # Errors
///
/// - [`Error::Shape`] on mismatched dimensions, or on an `Fa`/`Fb`/
///   `qinit` value that fails the input contract (non-positive or
///   non-finite scalar; `qinit` row that doesn't sum to 1; `qinit`
///   entry that's negative).
/// - [`Error::NonFinite`] if `qinit` contains a NaN/`±inf` entry,
///   or if a non-finite value appears in an algorithm intermediate
///   (the algorithm has no recovery; treat as a hard failure).
/// - [`Error::NonPositivePhi`] if any `phi[d] <= 0`.
///
/// `qinit` row-sum tolerance is `1e-9` — pyannote's caller produces
/// a softmaxed initializer that is unit-normalized to within float
/// roundoff, and Phase-0 captured rows are within `~1e-15` of 1.0.
/// This rejects a degraded or hand-crafted initializer that biases
/// the first speaker-model update — Codex review MEDIUM (round 1 of
/// Phase 2).
pub fn vbx_iterate(
  x: &DMatrix<f64>,
  phi: &DVector<f64>,
  qinit: &DMatrix<f64>,
  fa: f64,
  fb: f64,
  max_iters: usize,
) -> Result<VbxOutput, Error> {
  let (t, d) = x.shape();
  if phi.len() != d {
    return Err(Error::Shape("Phi.len() must equal X.ncols()"));
  }
  if qinit.nrows() != t {
    return Err(Error::Shape("qinit.nrows() must equal X.nrows()"));
  }
  let s = qinit.ncols();
  if s == 0 {
    return Err(Error::Shape("qinit must have at least one cluster column"));
  }
  if !fa.is_finite() || fa <= 0.0 {
    return Err(Error::Shape("Fa must be a positive finite scalar"));
  }
  if !fb.is_finite() || fb <= 0.0 {
    return Err(Error::Shape("Fb must be a positive finite scalar"));
  }
  for (i, p) in phi.iter().enumerate() {
    if *p <= 0.0 || p.is_nan() {
      return Err(Error::NonPositivePhi(*p, i));
    }
  }
  // qinit value validation: each row must be a discrete probability
  // distribution over speakers (finite, nonnegative, row-sum ≈ 1).
  // Without this, a malformed initializer (negative entries, rows
  // not summing to 1, NaN) produces finite-looking posteriors after
  // the first update and biases the speaker model silently. Also
  // matters at `max_iters == 0`, which returns `qinit` directly as
  // the output `gamma`. Codex review MEDIUM (round 1 of Phase 2).
  const QINIT_ROW_SUM_TOLERANCE: f64 = 1.0e-9;
  for tt in 0..t {
    let mut row_sum = 0.0;
    for sj in 0..s {
      let v = qinit[(tt, sj)];
      if !v.is_finite() {
        return Err(Error::NonFinite("qinit"));
      }
      if v < 0.0 {
        return Err(Error::Shape("qinit entries must be nonnegative"));
      }
      row_sum += v;
    }
    if (row_sum - 1.0).abs() > QINIT_ROW_SUM_TOLERANCE {
      return Err(Error::Shape("qinit rows must sum to 1"));
    }
  }

  // Pre-compute G[t] = -0.5 * (sum(X[t]^2) + D * log(2*pi))
  let log_2pi = (2.0_f64 * std::f64::consts::PI).ln();
  let mut g = DVector::<f64>::zeros(t);
  for r in 0..t {
    let row_sq: f64 = x.row(r).iter().map(|v| v * v).sum();
    g[r] = -0.5 * (row_sq + d as f64 * log_2pi);
  }
  // V = sqrt(Phi); rho[t,d] = X[t,d] * V[d]
  let v_sqrt: DVector<f64> = phi.map(|p| p.sqrt());
  let mut rho = x.clone();
  for r in 0..t {
    for c in 0..d {
      rho[(r, c)] *= v_sqrt[c];
    }
  }

  let mut gamma = qinit.clone();
  // pi starts as the uniform 1/S vector (matches pyannote's
  // `if type(pi) is int: pi = ones(pi)/pi` for the integer-pi path).
  let mut pi = DVector::<f64>::from_element(s, 1.0 / s as f64);

  let mut elbo_trajectory: Vec<f64> = Vec::with_capacity(max_iters);
  let epsilon = 1e-4_f64;
  let eps_log = 1e-8_f64;
  let fa_over_fb = fa / fb;

  for ii in 0..max_iters {
    // ── E-step (speaker-model update) ────────────────────────────
    // gamma_sum, invL, alpha
    // gamma_sum[s] = column-sum of gamma over T rows (Eq. 17 input).
    let gamma_sum = DVector::<f64>::from_vec((0..s).map(|j| gamma.column(j).sum()).collect());

    // invL[s,d] = 1 / (1 + Fa/Fb * gamma_sum[s] * Phi[d])  (Eq. 17)
    let mut inv_l = DMatrix::<f64>::zeros(s, d);
    for sj in 0..s {
      for dk in 0..d {
        let denom = 1.0 + fa_over_fb * gamma_sum[sj] * phi[dk];
        inv_l[(sj, dk)] = 1.0 / denom;
      }
    }

    // alpha[s,d] = Fa/Fb * invL[s,d] * (gamma.T @ rho)[s,d]  (Eq. 16)
    let prod = gamma.transpose() * &rho; // (S, D)
    let mut alpha = DMatrix::<f64>::zeros(s, d);
    for sj in 0..s {
      for dk in 0..d {
        alpha[(sj, dk)] = fa_over_fb * inv_l[(sj, dk)] * prod[(sj, dk)];
      }
    }

    // ── log_p_ (per-(frame, speaker) log-likelihood, Eq. 23) ─────
    // log_p_[t,s] = Fa * (rho @ alpha.T - 0.5*(invL+alpha**2)@Phi + G) (Eq. 23)
    let rho_alpha_t = &rho * alpha.transpose(); // (T, S)
    // (invL + alpha**2) @ Phi : (S, D) · (D,) → (S,)
    let mut sa_phi = DVector::<f64>::zeros(s);
    for sj in 0..s {
      let mut acc = 0.0;
      for dk in 0..d {
        let inv = inv_l[(sj, dk)];
        let a2 = alpha[(sj, dk)] * alpha[(sj, dk)];
        acc += (inv + a2) * phi[dk];
      }
      sa_phi[sj] = acc;
    }
    let mut log_p = DMatrix::<f64>::zeros(t, s);
    for tt in 0..t {
      for sj in 0..s {
        log_p[(tt, sj)] = fa * (rho_alpha_t[(tt, sj)] - 0.5 * sa_phi[sj] + g[tt]);
      }
    }

    // ── Responsibility update ────────────────────────────────────
    // log_pi, log_p_x via logsumexp, new gamma, new pi
    // log_pi[s] = log(pi[s] + eps_log)
    let log_pi: DVector<f64> = pi.map(|p| (p + eps_log).ln());
    // Fold log_pi into log_p in place — log_p is not referenced
    // outside this block, so we save the (T, S) clone.
    for tt in 0..t {
      for sj in 0..s {
        log_p[(tt, sj)] += log_pi[sj];
      }
    }
    // log_p_x[t] = logsumexp_t(log_p[t,:] + log_pi[:])
    let log_p_x = logsumexp_rows(&log_p);
    // gamma[t,s] = exp(log_p_[t,s] + log_pi[s] - log_p_x[t])
    let mut new_gamma = DMatrix::<f64>::zeros(t, s);
    for tt in 0..t {
      for sj in 0..s {
        // log_p now contains log_p + log_pi.
        new_gamma[(tt, sj)] = (log_p[(tt, sj)] - log_p_x[tt]).exp();
      }
    }
    gamma = new_gamma;
    // pi = gamma.sum(0); pi /= pi.sum()
    let mut new_pi = DVector::<f64>::zeros(s);
    for sj in 0..s {
      new_pi[sj] = gamma.column(sj).sum();
    }
    let pi_sum = new_pi.sum();
    if !pi_sum.is_finite() || pi_sum <= 0.0 {
      return Err(Error::NonFinite("pi sum"));
    }
    pi = new_pi / pi_sum;

    // ── ELBO (Eq. 25) ────────────────────────────────────────────
    // ELBO = sum(log_p_x) + Fb * 0.5 * sum_{s,d}(log(invL) - invL - alpha**2 + 1)  (Eq. 25)
    let log_p_x_total: f64 = log_p_x.iter().sum();
    let mut bracket = 0.0;
    for sj in 0..s {
      for dk in 0..d {
        let inv = inv_l[(sj, dk)];
        let a2 = alpha[(sj, dk)] * alpha[(sj, dk)];
        bracket += inv.ln() - inv - a2 + 1.0;
      }
    }
    let elbo = log_p_x_total + fb * 0.5 * bracket;
    if !elbo.is_finite() {
      return Err(Error::NonFinite("ELBO"));
    }
    elbo_trajectory.push(elbo);

    // ── Convergence check ────────────────────────────────────────
    if ii > 0 {
      let prev = elbo_trajectory[elbo_trajectory.len() - 2];
      let delta = elbo - prev;
      match classify_elbo_step(delta, epsilon) {
        ElboStep::Continue => {}
        ElboStep::Converged => break,
        ElboStep::Regressed(d) => {
          return Err(Error::ElboRegression { iter: ii, delta: d });
        }
      }
    }
  }

  Ok(VbxOutput {
    gamma,
    pi,
    elbo_trajectory,
  })
}
