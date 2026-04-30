# Phase 2: VBx HMM clustering Rust port

> **For agentic workers:** REQUIRED SUB-SKILL — use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Port `pyannote.audio.utils.vbx.VBx` (`utils/vbx.py:27-137`) to Rust as a standalone `diarization::vbx` module. Verify byte-for-byte (within float-cast tolerance) parity against the Phase-0 captured `vbx_state.npz` artifacts.

**Architecture:** New top-level module `diarization::vbx`, crate-private (`pub(crate) mod vbx;` in `src/lib.rs`, mirroring `diarization::plda`'s pattern from Phase 1). Single function `vbx_iterate(x, phi, qinit, fa, fb, max_iters) -> (gamma, pi, elbo)` plus a small set of types. Uses existing `nalgebra` for matrices; no new dependencies.

**Tech Stack:** Rust + `nalgebra` 0.34 (already a dep). f64 throughout. `logsumexp` implemented as a small helper (numpy/scipy version is `log(sum(exp(x - max))) + max`).

**Out of scope:**
- Integration with `Diarizer` (Phase 5).
- AHC initialization that produces `qinit` (Phase 4 — Phase-0 fixture provides `qinit` directly).
- Constrained Hungarian assignment (Phase 3).
- DER measurement / end-to-end accuracy (Phase 5).

---

## Pyannote 4.0.4 facts (relevant to this phase)

`VBx(X, Phi, Fa, Fb, pi, gamma, maxIters, ...)` (`utils/vbx.py:27-137`) is the variational EM core. The community-1 pipeline calls it with hyperparameters from `config.yaml`:

```yaml
params:
  clustering:
    threshold: 0.6   # AHC linkage threshold (Phase 4, NOT used by VBx itself)
    Fa: 0.07         # VBx sufficient-statistics scale
    Fb: 0.8          # VBx speaker regularization
```

`maxIters=20` is set by `cluster_vbx`'s call site in `pyannote/audio/pipelines/clustering.py:613`, overriding VBx's default of 10. `epsilon=1e-4` is the VBx default; `pi=qinit.shape[1]` is an integer (initial speaker count).

### Algorithm (paper: Landini, Profant, Diez, Burget — "Bayesian HMM clustering of x-vector sequences")

```python
D = X.shape[1]                  # 128 (post-PLDA dim)
G = -0.5 * (sum(X**2, 1, keepdims=True) + D * log(2*pi))   # (T, 1) per-frame constant
V = sqrt(Phi)                   # (D,)
rho = X * V                     # (T, D), broadcast

for ii in range(maxIters):
    # E-step (speaker-model update from previous gamma)
    invL = 1 / (1 + Fa/Fb * gamma.sum(0).T * Phi)        # (1, S, D)? actually (S, D)
    alpha = Fa/Fb * invL * (gamma.T @ rho)               # (S, D)
    log_p_ = Fa * (rho @ alpha.T - 0.5 * (invL + alpha**2) @ Phi + G)  # (T, S)
    # responsibility update
    log_pi = log(pi + 1e-8)                              # (S,)
    log_p_x = logsumexp(log_p_ + log_pi, axis=-1)        # (T,)
    gamma = exp(log_p_ + log_pi - log_p_x[:, None])       # (T, S)
    pi = gamma.sum(0); pi /= pi.sum()                     # (S,)
    # ELBO
    ELBO = sum(log_p_x) + Fb * 0.5 * sum(log(invL) - invL - alpha**2 + 1)
    if ii > 0 and ELBO - prev_ELBO < epsilon: break
return gamma, pi, [ELBO_history]
```

Concrete shapes for the Phase-0 fixture (`vbx_state.npz`):
- T=195 frames, D=128 PLDA dim, N_speakers=19 (initial; converges to 2 with Fa=0.07, Fb=0.8)
- `qinit.shape == (195, 19)`, `q_final.shape == (195, 19)`, `sp_final.shape == (19,)`
- ELBO trajectory: 16 iterations (converged before maxIters=20)

The captured `sp_final` confirms the algorithm correctly drives the redundant 17 speakers' priors to ~1.76e-14 while keeping the two real speakers' priors at ~0.85 and ~0.15.

### `logsumexp` implementation

scipy: `log(sum(exp(x - max(x)))) + max(x)`, computed along an axis. For numerical stability we shift by the per-row max before exponentiating.

---

## File structure

```
src/vbx/
  mod.rs                # module root, public API re-exports, #[cfg(test)] mod tests/parity_tests
  algo.rs               # vbx_iterate + helpers
  error.rs              # Error enum (Bounds checks, NaN intermediates)
  tests.rs              # model-free invariants (logsumexp, ELBO monotonicity, shape contracts)
  parity_tests.rs       # tests against tests/parity/fixtures/01_dialogue/vbx_state.npz
```

`src/lib.rs` adds `pub(crate) mod vbx;` (matches `diarization::plda`'s gating).

No new dependencies. No fixture changes (Phase-0 already captured everything VBx needs, except Fa/Fb which we hardcode as constants pinned to community-1).

---

## Tasks

### Task 0: Capture Fa/Fb hyperparameters into vbx_state.npz

**Why:** the community-1 model's `config.yaml` pins `Fa=0.07, Fb=0.8`, but those values are not in any captured fixture. Hardcoding them in Rust would let a future model upgrade silently desync the parity test from the captured `q_final` / `sp_final` (those are the *outputs* produced under those hyperparameters). Instead capture the inputs alongside the outputs.

**Files:**
- Modify: `tests/parity/python/capture_intermediates.py`
- Modify: `tests/parity/fixtures/01_dialogue/vbx_state.npz` (one-shot augment with known values from config.yaml)

- [ ] **Step 1: Update the capture script to save Fa/Fb**

In `capture_intermediates.py`, locate the `np.savez_compressed(out_dir / "vbx_state.npz", ...)` call and add:

```python
np.savez_compressed(
    out_dir / "vbx_state.npz",
    qinit=buf.qinit,
    q_final=buf.q_final,
    sp_final=buf.sp_final,
    elbo_trajectory=np.array(buf.elbo_trajectory, dtype=np.float64),
    # `Fa`, `Fb` are pipeline hyperparameters from
    # pyannote/speaker-diarization-community-1/config.yaml. They
    # are inputs to VBx, while q_final / sp_final / elbo_trajectory
    # are the outputs produced *under* those values; capturing the
    # inputs here keeps the parity test self-contained when the
    # model is upgraded. Codex review MEDIUM (Phase 2 plan, Task 0).
    fa=np.float64(cap.Fa),
    fb=np.float64(cap.Fb),
    max_iters=np.int64(20),  # cluster_vbx() override; see clustering.py:613
)
```

- [ ] **Step 2: One-shot augment the existing fixture**

Run via the existing venv:

```bash
tests/parity/python/.venv/bin/python -c "
import numpy as np
p = 'tests/parity/fixtures/01_dialogue/vbx_state.npz'
existing = dict(np.load(p))
# Values pinned to community-1's config.yaml at the snapshot revision
# in models/plda/SOURCE.md. Independently verifiable from the cached
# config.yaml under ~/.cache/huggingface/hub/.
existing['fa'] = np.float64(0.07)
existing['fb'] = np.float64(0.8)
existing['max_iters'] = np.int64(20)
np.savez_compressed(p, **existing)
v = np.load(p)
print('keys:', list(v.keys()))
print(f'fa={float(v[\"fa\"])} fb={float(v[\"fb\"])} max_iters={int(v[\"max_iters\"])}')"
```

Expected output: `keys: ['qinit', 'q_final', 'sp_final', 'elbo_trajectory', 'fa', 'fb', 'max_iters']` and `fa=0.07 fb=0.8 max_iters=20`.

- [ ] **Step 3: Commit**

```bash
git add tests/parity/python/capture_intermediates.py \
        tests/parity/fixtures/01_dialogue/vbx_state.npz
git commit -m "vbx-fixture: capture Fa/Fb/max_iters alongside VBx outputs"
```

---

### Task 1: Module scaffold + Error enum + lib.rs export

**Files:**
- Create: `src/vbx/mod.rs`
- Create: `src/vbx/error.rs`
- Modify: `src/lib.rs` (add `pub(crate) mod vbx;`)

- [ ] **Step 1: Write `src/vbx/error.rs`**

```rust
//! Error variants for `diarization::vbx`.

use thiserror::Error;

/// Errors produced by `vbx_iterate`.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum Error {
  /// Input shapes do not satisfy the contract.
  #[error("shape mismatch: {0}")]
  Shape(&'static str),

  /// A non-finite value (NaN / ±inf) appeared in an intermediate
  /// (rho, alpha, log_p_, ELBO, …). The algorithm has no recovery
  /// path; the caller should treat this as a hard failure.
  #[error("non-finite intermediate: {0}")]
  NonFinite(&'static str),

  /// `Phi` (the eigenvalue diagonal from `PldaTransform::phi()`) had
  /// a non-positive entry. The algorithm requires `Phi[d] > 0` for
  /// `sqrt(Phi)` and `1 + … * Phi` to be well-defined.
  #[error("Phi must be strictly positive; saw {0:.3e} at index {1}")]
  NonPositivePhi(f64, usize),
}
```

- [ ] **Step 2: Write `src/vbx/mod.rs`**

```rust
//! Variational Bayes HMM speaker clustering (VBx).
//!
//! Ports `pyannote.audio.utils.vbx.VBx` (`utils/vbx.py:27-137` in
//! pyannote.audio 4.0.4) to Rust. Consumes the post-PLDA features
//! produced by `diarization::plda::PldaTransform::project()` plus the
//! eigenvalue diagonal `diarization::plda::PldaTransform::phi()`, runs
//! variational EM iterations, and returns final speaker
//! responsibilities + priors + ELBO trajectory.
//!
//! ## Standalone — no `Diarizer` integration yet
//!
//! Phase 2 ships VBx as a pure-math module. The integration
//! (`Diarizer` consuming VBx output → cluster centroids → per-frame
//! diarization) lands in Phase 5. Until then `diarization::vbx` is
//! crate-private (see `src/lib.rs:62-72`).

#![allow(dead_code, unused_imports)]

mod algo;
mod error;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod parity_tests;

pub use algo::{vbx_iterate, VbxOutput};
pub use error::Error;
```

- [ ] **Step 3: Wire into `src/lib.rs`**

Add `pub(crate) mod vbx;` next to `pub(crate) mod plda;`.

- [ ] **Step 4: Empty `algo.rs` shell so the build compiles**

```rust
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
```

- [ ] **Step 5: Build + commit**

Run: `cargo build`
Expected: clean build with the new module.

```bash
git add src/vbx/ src/lib.rs
git commit -m "vbx: scaffold diarization::vbx module with crate-private gating"
```

---

### Task 2: `logsumexp` + shape-validation helpers

**Files:**
- Modify: `src/vbx/algo.rs`
- Create: `src/vbx/tests.rs`

- [ ] **Step 1: Write the failing test**

```rust
// src/vbx/tests.rs
use super::algo::logsumexp_rows;
use nalgebra::DMatrix;

/// scipy.special.logsumexp on a 2x3 matrix along axis=-1 returns a
/// length-2 vector. Reference value computed in Python:
///
/// ```python
/// >>> from scipy.special import logsumexp
/// >>> logsumexp([[1.0, 2.0, 3.0], [-100.0, -101.0, -102.0]], axis=-1)
/// array([3.40760596e+00, -9.95923940e+01])
/// ```
#[test]
fn logsumexp_rows_matches_scipy_reference() {
  let m = DMatrix::<f64>::from_row_slice(2, 3, &[
    1.0, 2.0, 3.0,
    -100.0, -101.0, -102.0,
  ]);
  let lse = logsumexp_rows(&m);
  assert!((lse[0] - 3.40760596_f64).abs() < 1e-8, "row0: {}", lse[0]);
  assert!((lse[1] - (-99.59239403555561_f64)).abs() < 1e-10, "row1: {}", lse[1]);
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test --lib vbx::tests::logsumexp`
Expected: FAIL with "function not defined".

- [ ] **Step 3: Implement `logsumexp_rows` in `src/vbx/algo.rs`**

```rust
use nalgebra::{DMatrix, DVector};

/// Row-wise `logsumexp` (numerically stable). For each row `r`:
///
/// ```text
/// out[r] = log(sum_j exp(m[r, j] - max_j m[r, j])) + max_j m[r, j]
/// ```
///
/// Matches `scipy.special.logsumexp(m, axis=-1)` modulo float roundoff.
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cargo test --lib vbx::tests::logsumexp`
Expected: PASS.

- [ ] **Step 5: Add an all-`-inf`-row test (edge case)**

```rust
#[test]
fn logsumexp_rows_all_neg_inf_returns_neg_inf() {
  let m = DMatrix::<f64>::from_row_slice(1, 3, &[
    f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY,
  ]);
  let lse = logsumexp_rows(&m);
  assert!(lse[0].is_infinite() && lse[0] < 0.0, "got {}", lse[0]);
}
```

- [ ] **Step 6: Commit**

```bash
git add src/vbx/algo.rs src/vbx/tests.rs
git commit -m "vbx: stable logsumexp_rows helper + scipy parity test"
```

---

### Task 3: Implement `vbx_iterate` (the EM loop)

**Files:**
- Modify: `src/vbx/algo.rs`
- Modify: `src/vbx/tests.rs`

- [ ] **Step 1: Write the failing test (input validation)**

```rust
// src/vbx/tests.rs
use crate::vbx::{vbx_iterate, Error};
use nalgebra::{DMatrix, DVector};

#[test]
fn vbx_rejects_phi_with_non_positive_entry() {
  let x = DMatrix::<f64>::zeros(5, 4);
  let mut phi = DVector::<f64>::from_element(4, 1.0);
  phi[2] = -0.5;
  let qinit = DMatrix::<f64>::from_element(5, 2, 0.5);
  let result = vbx_iterate(&x, &phi, &qinit, 0.07, 0.8, 20);
  assert!(matches!(result, Err(Error::NonPositivePhi(_, 2))), "got {result:?}");
}

#[test]
fn vbx_rejects_shape_mismatch_x_vs_qinit() {
  let x = DMatrix::<f64>::zeros(5, 4);          // T=5
  let phi = DVector::<f64>::from_element(4, 1.0);
  let qinit = DMatrix::<f64>::from_element(6, 2, 0.5); // T=6 ≠ 5
  let result = vbx_iterate(&x, &phi, &qinit, 0.07, 0.8, 20);
  assert!(matches!(result, Err(Error::Shape(_))), "got {result:?}");
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test --lib vbx`
Expected: FAIL on `vbx_rejects_*`.

- [ ] **Step 3: Implement the validated iteration**

Add to `src/vbx/algo.rs`:

```rust
pub fn vbx_iterate(
  x: &DMatrix<f64>,
  phi: &DVector<f64>,
  qinit: &DMatrix<f64>,
  fa: f64,
  fb: f64,
  max_iters: usize,
) -> Result<VbxOutput, Error> {
  // ── Shape validation ──────────────────────────────────────────
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

  // ── Phi positivity ────────────────────────────────────────────
  for (i, p) in phi.iter().enumerate() {
    if !(*p > 0.0) {
      return Err(Error::NonPositivePhi(*p, i));
    }
  }

  // ── Pre-compute G and rho ─────────────────────────────────────
  // G = -0.5 * (sum(X**2, axis=1) + D * log(2*pi))
  let log_2pi = (2.0 * std::f64::consts::PI).ln();
  let mut g = DVector::<f64>::zeros(t);
  for r in 0..t {
    let row_sq: f64 = x.row(r).iter().map(|v| v * v).sum();
    g[r] = -0.5 * (row_sq + d as f64 * log_2pi);
  }
  // V = sqrt(Phi) — element-wise. rho = X * V (row-wise scaling).
  let v_sqrt: DVector<f64> = phi.map(|p| p.sqrt());
  let mut rho = x.clone();
  for r in 0..t {
    for c in 0..d {
      rho[(r, c)] *= v_sqrt[c];
    }
  }

  // ── Initial state ─────────────────────────────────────────────
  let mut gamma = qinit.clone();
  // pi starts as 1/S vector (matches `if type(pi) is int`).
  let mut pi = DVector::<f64>::from_element(s, 1.0 / s as f64);

  let mut elbo_trajectory: Vec<f64> = Vec::with_capacity(max_iters);
  let epsilon = 1e-4_f64;
  let eps_log = 1e-8_f64;
  let fa_over_fb = fa / fb;

  for ii in 0..max_iters {
    // ── E-step (speaker-model update) ───────────────────────────
    // gamma_sum: (S,) — column sums of gamma
    let gamma_sum = DVector::<f64>::from_vec(
      (0..s).map(|j| gamma.column(j).sum()).collect(),
    );
    // invL[s, d] = 1 / (1 + Fa/Fb * gamma_sum[s] * Phi[d])
    let mut inv_l = DMatrix::<f64>::zeros(s, d);
    for sj in 0..s {
      for dk in 0..d {
        let denom = 1.0 + fa_over_fb * gamma_sum[sj] * phi[dk];
        inv_l[(sj, dk)] = 1.0 / denom;
      }
    }
    // alpha[s, d] = Fa/Fb * invL[s, d] * (gamma.T @ rho)[s, d]
    // gamma.T @ rho : (S, T) @ (T, D) = (S, D)
    let alpha = {
      let prod = gamma.transpose() * &rho; // (S, D)
      let mut a = DMatrix::<f64>::zeros(s, d);
      for sj in 0..s {
        for dk in 0..d {
          a[(sj, dk)] = fa_over_fb * inv_l[(sj, dk)] * prod[(sj, dk)];
        }
      }
      a
    };

    // ── log_p_[t, s] = Fa * (rho @ alpha.T - 0.5*(invL+alpha**2)@Phi + G) ─
    // rho @ alpha.T : (T, D) @ (D, S) = (T, S)
    let rho_alpha_t = &rho * alpha.transpose();
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
    // log_p_[t, s] = Fa * (rho_alpha_t[t, s] - 0.5 * sa_phi[s] + G[t])
    let mut log_p = DMatrix::<f64>::zeros(t, s);
    for tt in 0..t {
      for sj in 0..s {
        log_p[(tt, sj)] = fa * (rho_alpha_t[(tt, sj)] - 0.5 * sa_phi[sj] + g[tt]);
      }
    }

    // ── Responsibility update ───────────────────────────────────
    // log_pi[s] = log(pi[s] + eps)
    let log_pi: DVector<f64> = pi.map(|p| (p + eps_log).ln());
    // log_p_x[t] = logsumexp_t(log_p[t, :] + log_pi[:])
    let mut log_p_plus_pi = log_p.clone();
    for tt in 0..t {
      for sj in 0..s {
        log_p_plus_pi[(tt, sj)] += log_pi[sj];
      }
    }
    let log_p_x = logsumexp_rows(&log_p_plus_pi);
    // gamma[t, s] = exp(log_p_[t, s] + log_pi[s] - log_p_x[t])
    let mut new_gamma = DMatrix::<f64>::zeros(t, s);
    for tt in 0..t {
      for sj in 0..s {
        new_gamma[(tt, sj)] = (log_p[(tt, sj)] + log_pi[sj] - log_p_x[tt]).exp();
      }
    }
    gamma = new_gamma;
    // pi = gamma.sum(0); pi /= pi.sum()
    let mut new_pi = DVector::<f64>::zeros(s);
    for sj in 0..s {
      new_pi[sj] = gamma.column(sj).sum();
    }
    let pi_sum = new_pi.sum();
    pi = new_pi / pi_sum;

    // ── ELBO ────────────────────────────────────────────────────
    let log_p_x_total: f64 = log_p_x.iter().sum();
    let mut bracket = 0.0; // sum(log(invL) - invL - alpha**2 + 1) over all (s, d)
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

    // ── Convergence check ──────────────────────────────────────
    if ii > 0 {
      let prev = elbo_trajectory[elbo_trajectory.len() - 2];
      if elbo - prev < epsilon {
        // Pyannote prints a warning if ELBO decreased — we just stop.
        break;
      }
    }
  }

  Ok(VbxOutput { gamma, pi, elbo_trajectory })
}
```

- [ ] **Step 4: Run validation tests**

Run: `cargo test --lib vbx::tests`
Expected: all PASS (validation tests + logsumexp tests).

- [ ] **Step 5: Commit**

```bash
git add src/vbx/algo.rs src/vbx/tests.rs
git commit -m "vbx: implement vbx_iterate (variational EM core)"
```

---

### Task 4: Parity test against captured Phase-0 fixtures

**Files:**
- Create: `src/vbx/parity_tests.rs`

- [ ] **Step 1: Write the parity test**

```rust
//! Parity tests for `diarization::vbx` against the Phase-0 captured artifacts.
//!
//! Loads `tests/parity/fixtures/01_dialogue/{plda_embeddings, vbx_state}.npz`
//! and asserts that `vbx_iterate(post_plda, phi, qinit, Fa=0.07, Fb=0.8,
//! max_iters=20)` reproduces pyannote's q_final, sp_final, and
//! elbo_trajectory within float-cast tolerance.
//!
//! **Hard-fails** when fixtures are absent (same convention as
//! `src/plda/parity_tests.rs`). Codex review MEDIUM round 8b: a silent
//! skip would let CI/packaging silently stop checking the high-risk
//! algorithm port.

use std::{fs::File, io::BufReader, path::PathBuf};

use nalgebra::{DMatrix, DVector};
use npyz::npz::NpzArchive;

use crate::vbx::vbx_iterate;

fn repo_root() -> PathBuf { PathBuf::from(env!("CARGO_MANIFEST_DIR")) }
fn fixture(rel: &str) -> PathBuf { repo_root().join(rel) }

fn require_fixtures() {
  let required = [
    "tests/parity/fixtures/01_dialogue/plda_embeddings.npz",
    "tests/parity/fixtures/01_dialogue/vbx_state.npz",
  ];
  let missing: Vec<&str> = required.iter().copied()
    .filter(|p| !repo_root().join(p).exists())
    .collect();
  assert!(
    missing.is_empty(),
    "VBx parity fixtures missing: {missing:?}. \
     These ship with the crate via `cargo publish`; a missing \
     fixture is a packaging error, not an opt-out."
  );
}

fn read_npz_array<T: npyz::Deserialize>(path: &PathBuf, key: &str) -> (Vec<T>, Vec<u64>) {
  let f = File::open(path).expect("open npz");
  let mut z = NpzArchive::new(BufReader::new(f)).expect("read npz");
  let npy = z.by_name(key).expect("query archive")
    .unwrap_or_else(|| panic!("array `{key}` not in {}", path.display()));
  let shape: Vec<u64> = npy.shape().to_vec();
  let data: Vec<T> = npy.into_vec().expect("decode array");
  (data, shape)
}

#[test]
fn vbx_iterate_matches_pyannote_q_final_and_elbo() {
  require_fixtures();

  // Inputs.
  let plda_path = fixture("tests/parity/fixtures/01_dialogue/plda_embeddings.npz");
  let (post_plda_flat, post_plda_shape) = read_npz_array::<f64>(&plda_path, "post_plda");
  let t = post_plda_shape[0] as usize;
  let d = post_plda_shape[1] as usize;
  assert_eq!(d, 128);
  let x = DMatrix::<f64>::from_row_slice(t, d, &post_plda_flat);

  let (phi_flat, phi_shape) = read_npz_array::<f64>(&plda_path, "phi");
  assert_eq!(phi_shape, vec![128]);
  let phi = DVector::<f64>::from_vec(phi_flat);

  let vbx_path = fixture("tests/parity/fixtures/01_dialogue/vbx_state.npz");
  let (qinit_flat, qinit_shape) = read_npz_array::<f64>(&vbx_path, "qinit");
  let s = qinit_shape[1] as usize;
  let qinit = DMatrix::<f64>::from_row_slice(t, s, &qinit_flat);

  // Hyperparameters captured alongside the VBx outputs (Task 0).
  // Reading them from the fixture means a future model upgrade
  // surfaces as a parity-failure rather than a silent drift.
  let (fa_flat, _) = read_npz_array::<f64>(&vbx_path, "fa");
  let (fb_flat, _) = read_npz_array::<f64>(&vbx_path, "fb");
  let (max_iters_flat, _) = read_npz_array::<i64>(&vbx_path, "max_iters");
  let fa = fa_flat[0];
  let fb = fb_flat[0];
  let max_iters = max_iters_flat[0] as usize;

  // Run.
  let out = vbx_iterate(&x, &phi, &qinit, fa, fb, max_iters).expect("vbx_iterate");

  // Compare gamma.
  let (q_final_flat, _) = read_npz_array::<f64>(&vbx_path, "q_final");
  let q_final = DMatrix::<f64>::from_row_slice(t, s, &q_final_flat);
  let mut gamma_max_err = 0.0f64;
  for tt in 0..t {
    for sj in 0..s {
      let err = (out.gamma[(tt, sj)] - q_final[(tt, sj)]).abs();
      if err > gamma_max_err { gamma_max_err = err; }
    }
  }
  eprintln!("[parity_vbx] gamma max_abs_err = {gamma_max_err:.3e}");
  assert!(gamma_max_err < 1e-6, "gamma parity failed: {gamma_max_err:.3e}");

  // Compare pi.
  let (sp_final_flat, _) = read_npz_array::<f64>(&vbx_path, "sp_final");
  let mut pi_max_err = 0.0f64;
  for sj in 0..s {
    let err = (out.pi[sj] - sp_final_flat[sj]).abs();
    if err > pi_max_err { pi_max_err = err; }
  }
  eprintln!("[parity_vbx] pi max_abs_err = {pi_max_err:.3e}");
  assert!(pi_max_err < 1e-9, "pi parity failed: {pi_max_err:.3e}");

  // Compare ELBO trajectory.
  let (elbo_flat, _) = read_npz_array::<f64>(&vbx_path, "elbo_trajectory");
  assert_eq!(out.elbo_trajectory.len(), elbo_flat.len(),
    "ELBO iteration count mismatch");
  let mut elbo_max_err = 0.0f64;
  for (got, want) in out.elbo_trajectory.iter().zip(elbo_flat.iter()) {
    let err = (got - want).abs();
    if err > elbo_max_err { elbo_max_err = err; }
  }
  eprintln!("[parity_vbx] ELBO max_abs_err = {elbo_max_err:.3e}");
  assert!(elbo_max_err < 1e-6, "ELBO parity failed: {elbo_max_err:.3e}");
}
```

- [ ] **Step 2: Run the parity test**

Run: `cargo test --lib vbx::parity_tests`
Expected: PASS, with eprintln residuals comparable to the PLDA parity (gamma ≤ 1e-6, pi ≤ 1e-9, ELBO ≤ 1e-6).

If it fails: the most likely culprits are (a) a transposed matmul (gamma.T @ rho vs gamma @ rho), (b) a dropped `Fa` / `Fb` factor, (c) an off-by-one in the convergence check, (d) `log(pi + eps)` vs `log(pi)` (pyannote adds `1e-8`). Bisect by reading the captured ELBO trajectory: if iter-0 disagrees, the speaker-model update is wrong; if iter-1+ disagrees but iter-0 matches, the responsibility update is wrong.

- [ ] **Step 3: Commit**

```bash
git add src/vbx/parity_tests.rs
git commit -m "vbx: parity test against captured Phase-0 vbx_state.npz"
```

---

### Task 5: Model-free invariants + finishing touches

**Files:**
- Modify: `src/vbx/tests.rs`

- [ ] **Step 1: ELBO monotonicity test**

```rust
/// VBx must produce a monotonically non-decreasing ELBO (up to the
/// epsilon convergence stop). A regression that, e.g., reuses the
/// previous iteration's gamma in the alpha update would break this.
#[test]
fn vbx_elbo_is_monotonically_non_decreasing() {
  // 50 frames × 8 dim × 3 speakers, random-ish but deterministic.
  let t = 50;
  let d = 8;
  let s = 3;
  let mut x = DMatrix::<f64>::zeros(t, d);
  for i in 0..t {
    for j in 0..d {
      x[(i, j)] = ((i * 7 + j * 13) as f64 % 11.0) - 5.0;
    }
  }
  let phi = DVector::<f64>::from_element(d, 2.0);
  let qinit = DMatrix::<f64>::from_element(t, s, 1.0 / s as f64);
  let out = vbx_iterate(&x, &phi, &qinit, 0.07, 0.8, 20).expect("vbx_iterate");
  for w in out.elbo_trajectory.windows(2) {
    // Allow tiny float wobble at convergence (≤ 1e-6) before the
    // epsilon-based stop fires.
    assert!(
      w[1] - w[0] > -1.0e-6,
      "ELBO must not decrease: {} → {}", w[0], w[1]
    );
  }
}
```

- [ ] **Step 2: Gamma row-sum invariant**

```rust
/// At every iteration, `gamma[t, :]` is a discrete probability over
/// speakers, so each row must sum to 1 (within float roundoff).
#[test]
fn vbx_gamma_rows_sum_to_one() {
  let t = 30;
  let d = 4;
  let s = 4;
  let mut x = DMatrix::<f64>::zeros(t, d);
  for i in 0..t {
    for j in 0..d {
      x[(i, j)] = ((i + j) as f64).sin();
    }
  }
  let phi = DVector::<f64>::from_element(d, 1.5);
  let qinit = DMatrix::<f64>::from_element(t, s, 1.0 / s as f64);
  let out = vbx_iterate(&x, &phi, &qinit, 0.1, 0.5, 10).expect("vbx_iterate");
  for r in 0..t {
    let row_sum: f64 = (0..s).map(|c| out.gamma[(r, c)]).sum();
    assert!((row_sum - 1.0).abs() < 1e-12,
      "gamma row {r} sums to {row_sum}");
  }
}
```

- [ ] **Step 3: Pi sums to 1 invariant**

```rust
#[test]
fn vbx_pi_sums_to_one() {
  let t = 20;
  let d = 4;
  let s = 5;
  let x = DMatrix::<f64>::from_fn(t, d, |i, j| ((i * 3 + j) as f64).cos());
  let phi = DVector::<f64>::from_element(d, 1.0);
  let qinit = DMatrix::<f64>::from_element(t, s, 1.0 / s as f64);
  let out = vbx_iterate(&x, &phi, &qinit, 0.07, 0.8, 20).expect("vbx_iterate");
  let pi_sum: f64 = out.pi.iter().sum();
  assert!((pi_sum - 1.0).abs() < 1e-12, "pi sums to {pi_sum}");
}
```

- [ ] **Step 4: Determinism — same input twice gives byte-identical output**

```rust
#[test]
fn vbx_is_deterministic() {
  let t = 15;
  let d = 4;
  let s = 3;
  let x = DMatrix::<f64>::from_fn(t, d, |i, j| (i + 2 * j) as f64 * 0.1);
  let phi = DVector::<f64>::from_element(d, 2.0);
  let qinit = DMatrix::<f64>::from_element(t, s, 1.0 / s as f64);
  let a = vbx_iterate(&x, &phi, &qinit, 0.07, 0.8, 10).expect("a");
  let b = vbx_iterate(&x, &phi, &qinit, 0.07, 0.8, 10).expect("b");
  assert_eq!(a.elbo_trajectory, b.elbo_trajectory);
  for r in 0..t {
    for c in 0..s {
      assert_eq!(a.gamma[(r, c)], b.gamma[(r, c)]);
    }
  }
  for c in 0..s {
    assert_eq!(a.pi[c], b.pi[c]);
  }
}
```

- [ ] **Step 5: Run full test suite + clippy**

Run: `cargo test --lib vbx && cargo clippy --lib --all-targets -- -D warnings`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/vbx/tests.rs
git commit -m "vbx: model-free invariants (ELBO monotonicity, gamma/pi normalization, determinism)"
```

---

### Task 6: PR + review iteration

- [ ] **Step 1: Push the branch**

```bash
git push -u origin feat/phase2-vbx
```

- [ ] **Step 2: Open PR targeting `0.1.0`**

```bash
gh pr create --base 0.1.0 --title "feat(vbx): Phase 2 — VBx HMM Rust port" --body ...
```

- [ ] **Step 3: Run `/codex:adversarial-review --base 0.1.0`**

Address findings as they come in (same iterative pattern as Phase 1).

- [ ] **Step 4: Merge to `0.1.0`**

After review converges; same merge convention as Phase 1.
