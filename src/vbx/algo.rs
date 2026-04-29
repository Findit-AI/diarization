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

/// Absolute floor for the ELBO regression tolerance. Caps the band
/// for tiny ELBOs where the relative term is negligible.
const ELBO_REGRESSION_ATOL: f64 = 1.0e-9;

/// Relative scaling for the ELBO regression tolerance. ELBO is an
/// accumulated sum over `T * S * D` matrix entries plus `T` per-frame
/// terms; float roundoff therefore scales with the working magnitude
/// of the ELBO itself. Codex round 3 reproduced a final delta of
/// `~-2.47e-8` for finite community-Fa/Fb inputs at |ELBO| ≈ 2700,
/// well outside an absolute `1e-9` band but ~9× *inside* the
/// scale-aware band `1e-9 + 1e-9 * 2700 ≈ 2.7e-6`. The previous
/// fixture-only calibration would have rejected that as an algorithm
/// failure.
const ELBO_REGRESSION_RTOL: f64 = 1.0e-9;

/// Compute the regression tolerance for a given ELBO magnitude.
/// `band(prev, elbo) = atol + rtol * max(|prev|, |elbo|)`.
fn regression_tolerance(prev_elbo: f64, elbo: f64) -> f64 {
  ELBO_REGRESSION_ATOL + ELBO_REGRESSION_RTOL * prev_elbo.abs().max(elbo.abs())
}

/// Outcome of comparing one EM iteration's ELBO against the previous.
#[derive(Debug, PartialEq)]
pub(super) enum ElboStep {
  /// Improvement >= `epsilon` — keep iterating.
  Continue,
  /// Improvement < `epsilon` (including small negative deltas within
  /// the scale-aware regression-tolerance band) — converged, exit
  /// cleanly.
  Converged,
  /// Negative delta beyond the scale-aware regression-tolerance band
  /// — VB EM's monotonicity invariant is violated. Carries the
  /// offending delta.
  Regressed(f64),
}

/// Classify an ELBO step into the three convergence regimes.
///
/// The regression boundary is scale-aware: any delta within
/// `±(atol + rtol * max(|prev|, |elbo|))` is treated as float
/// roundoff and routed to `Converged`. Beyond that band on the
/// negative side: `Regressed`. This matters because ELBO accumulates
/// over `T * S * D` matrix entries plus `T` per-frame terms; float
/// roundoff therefore scales with magnitude, and an absolute
/// tolerance calibrated against a single fixture would error out on
/// numerically awkward but otherwise valid inputs (Codex review
/// MEDIUM round 3).
///
/// Pyannote's `vbx.py:133-136` uses `if ELBO - prev < epsilon: break`
/// for both small-positive convergence AND any negative regression,
/// printing a warning for the regression case. The Rust port treats
/// a regression *beyond the float-roundoff band* as an error (no
/// print mechanism, and downstream clustering should not silently
/// consume a materially regressed posterior).
pub(super) fn classify_elbo_step(delta: f64, prev_elbo: f64, elbo: f64, epsilon: f64) -> ElboStep {
  let regression_tol = regression_tolerance(prev_elbo, elbo);
  if delta < -regression_tol {
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
///   entry that's negative; `qinit` speaker column with zero total
///   mass).
/// - [`Error::NonFinite`] if `x` or `qinit` contains a NaN/`±inf`
///   entry, or if a non-finite value appears in an algorithm
///   intermediate (the algorithm has no recovery; treat as a hard
///   failure).
/// - [`Error::NonPositivePhi`] if any `phi[d]` is not strictly
///   positive *and* finite (zero, negative, NaN, or `±inf`).
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
  if d == 0 {
    // VBx without feature evidence collapses to "iterate the uniform
    // prior". With D=0 the per-dimension loops are no-ops, the
    // likelihood term vanishes, and gamma/pi are driven only by `pi`
    // and `eps_log` — a finite-but-empty cluster posterior dressed
    // up as a clustering result. Reject at the boundary so a schema
    // drift or feature-construction bug doesn't return plausible
    // garbage. Codex review MEDIUM round 6 of Phase 2.
    return Err(Error::Shape("X must have at least one feature dimension"));
  }
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
  // Phi must be strictly positive AND finite. The previous check
  // accepted `+inf` because `inf > 0.0` is true and `inf.is_nan()`
  // is false; an infinite eigenvalue from a corrupted PLDA upstream
  // would have flowed into `sqrt(Phi)` and `1 + Fa/Fb * gamma_sum *
  // Phi`, producing NaN/Inf intermediates downstream. Codex review
  // MEDIUM round 5.
  for (i, p) in phi.iter().enumerate() {
    if !p.is_finite() || *p <= 0.0 {
      return Err(Error::NonPositivePhi(*p, i));
    }
  }
  // X must be entirely finite. Without this, NaN/Inf in the
  // post-PLDA features would either:
  //   - silently return Ok at `max_iters = 0` with the unvalidated
  //     qinit as "gamma", or
  //   - poison G/rho in the pre-loop and surface as a generic
  //     `NonFinite("ELBO")` later instead of a clear input error.
  // The boundary contract is "non-finite intermediates are hard
  // failures"; admitting non-finite inputs violates that. Codex
  // review MEDIUM round 5.
  if x.iter().any(|v| !v.is_finite()) {
    return Err(Error::NonFinite("x"));
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
  // Reject "dead" or "near-dead" speaker columns. A column whose
  // total initial mass is below the calibrated `QINIT_COL_SUM_MIN`
  // floor cannot be a legitimate speaker initialization: pyannote's
  // softmax-of-smoothed-one-hot caller produces every real-speaker
  // column with `col_sum ≥ ~1.16` (a single-frame speaker; observed
  // empirically across the Phase-0 fixture has min `1.158`), while
  // a column with no real-frame assignment under the same smoothing
  // sits around `T * exp(0) / (exp(7) + (S-1))` ≈ `0.175` for `S=19,
  // T=195`. The `0.5` threshold cleanly separates the two regimes
  // (~2.3× margin to the real minimum, ~2.9× margin above the
  // residue floor). Without this guard, the uniform `pi = 1/S`
  // initialization gives a `1/S` prior to a speaker that has only
  // numerical residue in qinit, and the first EM update can
  // resurrect it on weak/symmetric features — fabricating an extra
  // speaker. Round 4 added the exact-zero check; Codex round 6
  // pointed out that exact-zero is too narrow.
  const QINIT_COL_SUM_MIN: f64 = 0.5;
  for sj in 0..s {
    let col_sum: f64 = (0..t).map(|tt| qinit[(tt, sj)]).sum();
    if col_sum < QINIT_COL_SUM_MIN {
      return Err(Error::Shape(
        "qinit speaker column has below-floor total mass (would be \
         resurrected by uniform-pi initialization)",
      ));
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

  // `Vec::new()` rather than `Vec::with_capacity(max_iters)` —
  // `max_iters` is caller-controlled, and `with_capacity` would
  // allocate `max_iters * size_of::<f64>()` bytes up front. A
  // misconfigured caller passing `usize::MAX` would crash with
  // "capacity overflow" or OOM before the loop ran. The iteration
  // count is bounded in practice (the captured trajectory converges
  // in 16 of 20 iterations); amortized push cost is O(1). Codex
  // review MEDIUM round 4.
  let mut elbo_trajectory: Vec<f64> = Vec::new();
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
      match classify_elbo_step(delta, prev, elbo, epsilon) {
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
