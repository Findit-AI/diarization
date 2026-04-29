//! Parity tests for `dia::vbx` against the Phase-0 captured artifacts.
//!
//! Loads `tests/parity/fixtures/01_dialogue/{plda_embeddings, vbx_state}.npz`
//! and asserts that `vbx_iterate(post_plda, phi, qinit, fa, fb, max_iters)`
//! reproduces pyannote's `q_final`, `sp_final`, and `elbo_trajectory`
//! within float-cast tolerance.
//!
//! **Hard-fails** when fixtures are absent (same convention as
//! `src/plda/parity_tests.rs`). The fixtures are committed to the
//! repo and ship via `cargo publish`; a missing one is a packaging
//! error, not an opt-out.

use std::{fs::File, io::BufReader, path::PathBuf};

use nalgebra::{DMatrix, DVector};
use npyz::npz::NpzArchive;

use crate::vbx::vbx_iterate;

fn repo_root() -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn fixture(rel: &str) -> PathBuf {
  repo_root().join(rel)
}

/// Hard-fail if the Phase-0 fixtures are absent. Mirrors
/// `src/plda/parity_tests.rs::require_fixtures`.
fn require_fixtures() {
  let required = [
    "tests/parity/fixtures/01_dialogue/plda_embeddings.npz",
    "tests/parity/fixtures/01_dialogue/vbx_state.npz",
  ];
  let missing: Vec<&str> = required
    .iter()
    .copied()
    .filter(|p| !repo_root().join(p).exists())
    .collect();
  assert!(
    missing.is_empty(),
    "VBx parity fixtures missing: {missing:?}. \
     These ship with the crate via `cargo publish`; a missing \
     fixture is a packaging error, not an opt-out. Re-run \
     `tests/parity/python/capture_intermediates.py` against the \
     Phase-0 clip to regenerate, or restore the files from a \
     full checkout."
  );
}

fn read_npz_array<T>(path: &PathBuf, key: &str) -> (Vec<T>, Vec<u64>)
where
  T: npyz::Deserialize,
{
  let f = File::open(path).expect("open npz");
  let mut z = NpzArchive::new(BufReader::new(f)).expect("read npz");
  let npy = z
    .by_name(key)
    .expect("query archive")
    .unwrap_or_else(|| panic!("array `{key}` not in {}", path.display()));
  let shape: Vec<u64> = npy.shape().to_vec();
  let data: Vec<T> = npy.into_vec().expect("decode array");
  (data, shape)
}

#[test]
fn vbx_iterate_matches_pyannote_q_final_pi_elbo() {
  require_fixtures();

  // ── Inputs (post_plda, phi from PLDA stage; qinit, fa, fb,
  //    max_iters from the captured VBx run) ────────────────────────
  let plda_path = fixture("tests/parity/fixtures/01_dialogue/plda_embeddings.npz");
  let (post_plda_flat, post_plda_shape) = read_npz_array::<f64>(&plda_path, "post_plda");
  assert_eq!(post_plda_shape.len(), 2);
  let t = post_plda_shape[0] as usize;
  let d = post_plda_shape[1] as usize;
  assert_eq!(d, 128);
  let x = DMatrix::<f64>::from_row_slice(t, d, &post_plda_flat);

  let (phi_flat, phi_shape) = read_npz_array::<f64>(&plda_path, "phi");
  assert_eq!(phi_shape, vec![128]);
  let phi = DVector::<f64>::from_vec(phi_flat);

  let vbx_path = fixture("tests/parity/fixtures/01_dialogue/vbx_state.npz");
  let (qinit_flat, qinit_shape) = read_npz_array::<f64>(&vbx_path, "qinit");
  assert_eq!(qinit_shape.len(), 2);
  assert_eq!(qinit_shape[0] as usize, t);
  let s = qinit_shape[1] as usize;
  let qinit = DMatrix::<f64>::from_row_slice(t, s, &qinit_flat);

  // Hyperparameters were captured alongside the VBx outputs (Task 0).
  // Reading from the fixture means a future model upgrade surfaces
  // as a parity failure rather than a silent drift.
  let (fa_flat, _) = read_npz_array::<f64>(&vbx_path, "fa");
  let (fb_flat, _) = read_npz_array::<f64>(&vbx_path, "fb");
  let (max_iters_flat, _) = read_npz_array::<i64>(&vbx_path, "max_iters");
  let fa = fa_flat[0];
  let fb = fb_flat[0];
  let max_iters = max_iters_flat[0] as usize;

  // ── Run ────────────────────────────────────────────────────────
  let out = vbx_iterate(&x, &phi, &qinit, fa, fb, max_iters).expect("vbx_iterate");

  // ── Compare gamma (T x S) ──────────────────────────────────────
  let (q_final_flat, q_final_shape) = read_npz_array::<f64>(&vbx_path, "q_final");
  assert_eq!(q_final_shape, vec![t as u64, s as u64]);
  let q_final = DMatrix::<f64>::from_row_slice(t, s, &q_final_flat);
  let mut gamma_max_err = 0.0f64;
  for tt in 0..t {
    for sj in 0..s {
      let err = (out.gamma[(tt, sj)] - q_final[(tt, sj)]).abs();
      if err > gamma_max_err {
        gamma_max_err = err;
      }
    }
  }
  eprintln!("[parity_vbx] gamma max_abs_err = {gamma_max_err:.3e}");
  assert!(
    gamma_max_err < 1.0e-6,
    "gamma parity failed: max_abs_err = {gamma_max_err:.3e}"
  );

  // ── Compare pi (S,) ────────────────────────────────────────────
  let (sp_final_flat, sp_final_shape) = read_npz_array::<f64>(&vbx_path, "sp_final");
  assert_eq!(sp_final_shape, vec![s as u64]);
  let mut pi_max_err = 0.0f64;
  for (sj, &want) in sp_final_flat.iter().enumerate() {
    let err = (out.pi[sj] - want).abs();
    if err > pi_max_err {
      pi_max_err = err;
    }
  }
  eprintln!("[parity_vbx] pi max_abs_err = {pi_max_err:.3e}");
  assert!(
    pi_max_err < 1.0e-9,
    "pi parity failed: max_abs_err = {pi_max_err:.3e}"
  );

  // ── Compare ELBO trajectory ────────────────────────────────────
  let (elbo_flat, elbo_shape) = read_npz_array::<f64>(&vbx_path, "elbo_trajectory");
  assert_eq!(elbo_shape.len(), 1);
  assert_eq!(
    out.elbo_trajectory.len(),
    elbo_flat.len(),
    "ELBO iteration count mismatch: rust={} pyannote={}",
    out.elbo_trajectory.len(),
    elbo_flat.len()
  );
  let mut elbo_max_err = 0.0f64;
  for (got, want) in out.elbo_trajectory.iter().zip(elbo_flat.iter()) {
    let err = (got - want).abs();
    if err > elbo_max_err {
      elbo_max_err = err;
    }
  }
  eprintln!("[parity_vbx] ELBO max_abs_err = {elbo_max_err:.3e}");
  assert!(
    elbo_max_err < 1.0e-6,
    "ELBO parity failed: max_abs_err = {elbo_max_err:.3e}"
  );
}
