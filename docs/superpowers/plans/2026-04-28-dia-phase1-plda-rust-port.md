# Phase 1: PLDA Rust port

> **For agentic workers:** REQUIRED SUB-SKILL — use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Port `pyannote/audio` PLDA transformation
(`pyannote.audio.utils.vbx.vbx_setup` + `xvec_tf` + `plda_tf` —
`utils/vbx.py:181-218`) to Rust as a standalone `dia::plda` module.
Verify byte-for-byte (within float-cast tolerance) parity against the
Phase-0 captured `plda_embeddings.npz` artifacts.

**Architecture:** New top-level module `dia::plda` with three responsibilities — load the two `.npz` weight files, expose `xvec_transform(&[f32; 256]) -> [f64; 128]` and `plda_transform(&[f64; 128]) -> [f64; 128]`, and pre-compute `phi` (the descending-sorted eigenvalue diagonal that VBx will consume in Phase 2). Internally uses `nalgebra` for matrix ops and the generalized-eigenvalue solve, plus a new `npyz` dependency for `.npz` parsing.

**Tech Stack:** Rust + `nalgebra` 0.34 (already a dep) + `npyz` (new). f64 throughout to match numpy's default; f32→f64 promotion happens at the input boundary.

**Out of scope:**
- Integration with `Diarizer` (the existing pipeline still uses raw cosine on the WeSpeaker embeddings; PLDA-projected embeddings get wired in via Phase 2 VBx + Phase 5 final integration).
- VBx clustering itself (Phase 2).
- Constrained Hungarian (Phase 3).
- Centroid AHC (Phase 4).
- DER measurement (Phase 5 end-to-end).

---

## Pyannote 4.0.4 facts (relevant to this phase only)

`vbx_setup(transform_npz, plda_npz)` (`utils/vbx.py:181-218`) loads two `.npz` files and returns three things — two functions plus the eigenvalue diagonal:

```python
x = np.load(transform_npz)
mean1, mean2, lda = x["mean1"], x["mean2"], x["lda"]   # (256,), (128,), (256,128)

p = np.load(plda_npz)
plda_mu, plda_tr, plda_psi = p["mu"], p["tr"], p["psi"]   # (128,), (128,128), (128,)

W = inv(plda_tr.T @ plda_tr)              # (128,128) symmetric positive-definite
B = inv((plda_tr.T / plda_psi) @ plda_tr) # (128,128) symmetric

acvar, wccn = eigh(B, W)                  # generalized eigh
plda_psi = acvar[::-1]                    # reversed (descending)
plda_tr  = wccn.T[::-1]                   # reversed (rows match psi order)

xvec_tf = lambda x: sqrt(lda.shape[1]) * l2_norm(
    lda.T @ (sqrt(lda.shape[0]) * l2_norm(x - mean1).T).T - mean2
)
plda_tf = lambda x0, lda_dim=lda.shape[1]: ((x0 - plda_mu) @ plda_tr.T)[:, :lda_dim]
```

**Concrete dimensions** (verified against the snapshot):
- `lda.shape == (256, 128)` ⇒ `lda.shape[0] = 256` (input dim), `lda.shape[1] = 128` (output dim).
- `mean1: (256,)`, `mean2: (128,)`, `mu: (128,)`, `tr: (128, 128)`, `psi: (128,)`.
- `lda_dim` defaults to 128, so `plda_tf`'s output slice is a no-op for the standard case.

**`xvec_tf` output norm** (verified against Phase-0 capture):
- `sqrt(D_out) * l2_norm(...) ⇒ ‖post_xvec‖ ≈ sqrt(128) ≈ 11.313708`.
- The "L2-normed" pseudocode in the original analysis doc was incomplete — the outer `sqrt(D_out)` factor scales the unit vector.

**Generalized-eigh in Rust** — nalgebra has only ordinary `SymmetricEigen<T>`. The standard reduction:
1. Cholesky decompose `W = L L^T` (W is symmetric positive-definite).
2. Form `M = L^{-1} B L^{-T}` (also symmetric).
3. Eigendecompose `M = Y Λ Y^T` via `SymmetricEigen`.
4. Recover the generalized eigenvectors `X = L^{-T} Y`. Eigenvalues are `Λ`.

scipy's `eigh(B, W)` returns ascending; pyannote then reverses. nalgebra's `SymmetricEigen` returns unsorted, so we sort descending and reorder eigenvectors to match — which makes the final ordering equivalent to `acvar[::-1]` / `wccn.T[::-1]`.

**Eigenvector sign indeterminacy.** Eigenvectors are unique only up to scalar sign in `eigh`. Different LAPACK implementations may return `v` vs `-v`. This means **per-element** parity for `post_plda` may fail by sign in some columns even when the algorithm is correct. Fall-backs:
- Per-element absolute-value match.
- Gram-matrix match: `post_plda_rust @ post_plda_rust.T == post_plda_python @ post_plda_python.T` is sign-invariant.

The xvec stage has no eigen step, so it should match per-element.

---

## File Structure

- Create: `src/plda/mod.rs` — module root, public re-exports.
- Create: `src/plda/error.rs` — `Error` enum.
- Create: `src/plda/loader.rs` — `.npz` parsing of the six named arrays.
- Create: `src/plda/transform.rs` — `PldaTransform` struct + load/eigh/transform methods.
- Create: `src/plda/tests.rs` — unit tests (small synthetic inputs).
- Create: `tests/parity_plda.rs` — integration test against Phase-0 captured artifacts.
- Modify: `src/lib.rs` — add `pub mod plda;`.
- Modify: `Cargo.toml` — add `npyz = "0.8"` (or whatever the current stable major is — Task 1 confirms).

---

## Task 1: Cargo dep + module scaffolding

**Files:**
- Modify: `Cargo.toml`
- Modify: `src/lib.rs`
- Create: `src/plda/mod.rs`, `src/plda/error.rs`

- [ ] **Step 1: Confirm npyz is the right crate**

```bash
curl -fsSL https://crates.io/api/v1/crates/npyz | python3 -c "import sys,json; m=json.load(sys.stdin); print('latest:', m['crate']['max_stable_version']); print('description:', m['crate']['description'])"
```

Expected: a stable version (likely 0.8.x). If `npyz` is unmaintained or a better alternative shows up, pivot — but `npyz` is the current canonical pure-Rust `.npy`/`.npz` reader.

- [ ] **Step 2: Add the dep to Cargo.toml**

```toml
[dependencies]
# ... existing entries ...
npyz = "0.8"
```

- [ ] **Step 3: Create empty module skeleton**

`src/plda/mod.rs`:

```rust
//! PLDA (Probabilistic Linear Discriminant Analysis) transform.
//!
//! Ports `pyannote.audio.utils.vbx.vbx_setup` plus its inner
//! `xvec_tf` / `plda_tf` lambdas (`utils/vbx.py:181-218`) to Rust.
//! Loads two `.npz` weight files (shipped with
//! `pyannote/speaker-diarization-community-1`, redistributed under
//! `models/plda/`) and exposes a deterministic transformation chain:
//!
//! ```text
//! 256-d WeSpeaker embedding (f32)
//!         │
//!         ▼  xvec_transform
//! 128-d PLDA-stage-1 (f64, sqrt(128)-scaled L2-norm)
//!         │
//!         ▼  plda_transform
//! 128-d PLDA-stage-2 (f64, whitened — input to VBx in Phase 2)
//! ```

mod error;
mod loader;
mod transform;

#[cfg(test)]
mod tests;

pub use error::Error;
pub use transform::PldaTransform;

/// PLDA stage-1 / stage-2 dimension. Pyannote's
/// `pyannote/speaker-diarization-community-1` always uses 128.
pub const PLDA_DIMENSION: usize = 128;

/// WeSpeaker embedding dimension (input to `xvec_transform`).
pub const EMBEDDING_DIMENSION: usize = 256;
```

`src/plda/error.rs`:

```rust
//! Error type for `dia::plda`.

use std::path::PathBuf;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    /// Failed to read or parse a `.npz` weight file.
    #[error("failed to load PLDA weights from {path}: {source}")]
    LoadNpz {
        path: PathBuf,
        #[source]
        source: NpzError,
    },

    /// A required array is missing from the `.npz` archive.
    #[error("PLDA weights at {path} missing array `{key}`")]
    MissingArray { path: PathBuf, key: &'static str },

    /// Array shape is not what we expect.
    #[error(
        "PLDA weights at {path}: array `{key}` has shape {got:?}, expected {expected:?}"
    )]
    ShapeMismatch {
        path: PathBuf,
        key: &'static str,
        expected: &'static [usize],
        got: Vec<usize>,
    },

    /// The within-class covariance matrix `W` is not positive-definite,
    /// so its Cholesky factor doesn't exist. This means the PLDA
    /// weights are corrupted or inconsistent.
    #[error("PLDA weights produced a non-positive-definite W matrix; cannot run generalized eigh")]
    WNotPositiveDefinite,
}

/// Wrapper for `npyz` errors so the public `Error` doesn't expose
/// `npyz` types directly.
#[derive(Debug, Error)]
pub enum NpzError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Npyz(#[from] npyz::DTypeError),
    #[error(transparent)]
    NpyzRead(#[from] npyz::ReadNpyError),
}
```

(Adjust `NpzError` variants once Task 2 confirms the exact error types `npyz` exposes — this is the load-bearing part the loader will populate.)

- [ ] **Step 4: Wire into the lib**

In `src/lib.rs`, add (alphabetically next to existing `pub mod` lines):

```rust
pub mod plda;
```

- [ ] **Step 5: Verify the skeleton compiles**

```bash
cargo build --lib 2>&1 | tail -20
```

Expected: `Finished dev profile`. Some unused-import warnings are acceptable since loader/transform are still empty stubs.

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml src/lib.rs src/plda/mod.rs src/plda/error.rs
git commit -m "plda: scaffold dia::plda module + npyz dep"
```

No `Co-Authored-By` trailer.

---

## Task 2: `.npz` loader

**Files:** Create `src/plda/loader.rs`.

- [ ] **Step 1: Inspect the actual `.npz` files**

```bash
cd /Users/user/Develop/findit-studio/dia
python3 -c "
import numpy as np
for path in ['models/plda/xvec_transform.npz', 'models/plda/plda.npz']:
    print(path)
    a = np.load(path)
    for k in a.keys():
        v = a[k]
        print(f'  {k}: shape={v.shape} dtype={v.dtype}')
"
```

Expected:
```
models/plda/xvec_transform.npz
  mean1: shape=(256,) dtype=float64
  mean2: shape=(128,) dtype=float64
  lda:   shape=(256, 128) dtype=float64
models/plda/plda.npz
  mu:  shape=(128,) dtype=float64
  tr:  shape=(128, 128) dtype=float64
  psi: shape=(128,) dtype=float64
```

If shapes or dtypes differ, the loader needs adjusting; record the actual values and proceed.

- [ ] **Step 2: Write the loader**

Create `src/plda/loader.rs`:

```rust
//! `.npz` weight loading for `PldaTransform`.

use std::{
    fs::File,
    io::{BufReader, Read, Seek},
    path::{Path, PathBuf},
};

use nalgebra::{DMatrix, DVector};
use npyz::{npz::NpzArchive, NpyFile};

use crate::plda::error::{Error, NpzError};

/// Raw arrays loaded from `xvec_transform.npz`.
pub(super) struct XvecWeights {
    pub mean1: DVector<f64>,  // (256,)
    pub mean2: DVector<f64>,  // (128,)
    pub lda: DMatrix<f64>,    // (256, 128)
}

/// Raw arrays loaded from `plda.npz` (pre-eigh; transform.rs runs the
/// eigh in `PldaTransform::new`).
pub(super) struct PldaWeights {
    pub mu: DVector<f64>,    // (128,)
    pub tr: DMatrix<f64>,    // (128, 128)
    pub psi: DVector<f64>,   // (128,)
}

pub(super) fn load_xvec(path: &Path) -> Result<XvecWeights, Error> {
    let mut npz = open_npz(path)?;
    let mean1 = read_vector(&mut npz, path, "mean1", 256)?;
    let mean2 = read_vector(&mut npz, path, "mean2", 128)?;
    let lda = read_matrix(&mut npz, path, "lda", 256, 128)?;
    Ok(XvecWeights { mean1, mean2, lda })
}

pub(super) fn load_plda(path: &Path) -> Result<PldaWeights, Error> {
    let mut npz = open_npz(path)?;
    let mu = read_vector(&mut npz, path, "mu", 128)?;
    let tr = read_matrix(&mut npz, path, "tr", 128, 128)?;
    let psi = read_vector(&mut npz, path, "psi", 128)?;
    Ok(PldaWeights { mu, tr, psi })
}

fn open_npz(path: &Path) -> Result<NpzArchive<BufReader<File>>, Error> {
    let f = File::open(path).map_err(|e| Error::LoadNpz {
        path: path.to_path_buf(),
        source: NpzError::Io(e),
    })?;
    NpzArchive::new(BufReader::new(f)).map_err(|e| Error::LoadNpz {
        path: path.to_path_buf(),
        source: NpzError::Io(e),
    })
}

fn read_vector<R: Read + Seek>(
    npz: &mut NpzArchive<R>,
    path: &Path,
    key: &'static str,
    expected_len: usize,
) -> Result<DVector<f64>, Error> {
    let npy = npz_array(npz, path, key)?;
    let shape = npy.shape().to_vec();
    if shape.as_slice() != [expected_len as u64] {
        return Err(Error::ShapeMismatch {
            path: path.to_path_buf(),
            key,
            expected: vector_shape_for(expected_len),
            got: shape.into_iter().map(|d| d as usize).collect(),
        });
    }
    let data: Vec<f64> = read_f64(npy, path, key)?;
    Ok(DVector::from_vec(data))
}

fn read_matrix<R: Read + Seek>(
    npz: &mut NpzArchive<R>,
    path: &Path,
    key: &'static str,
    rows: usize,
    cols: usize,
) -> Result<DMatrix<f64>, Error> {
    let npy = npz_array(npz, path, key)?;
    let shape = npy.shape().to_vec();
    if shape.as_slice() != [rows as u64, cols as u64] {
        return Err(Error::ShapeMismatch {
            path: path.to_path_buf(),
            key,
            expected: matrix_shape_for(rows, cols),
            got: shape.into_iter().map(|d| d as usize).collect(),
        });
    }
    let data: Vec<f64> = read_f64(npy, path, key)?;
    // npyz returns C-order data; nalgebra's DMatrix is column-major.
    Ok(DMatrix::from_row_slice(rows, cols, &data))
}

fn npz_array<R: Read + Seek>(
    npz: &mut NpzArchive<R>,
    path: &Path,
    key: &'static str,
) -> Result<NpyFile<impl Read + '_>, Error> {
    npz.by_name(key)
        .map_err(|e| Error::LoadNpz {
            path: path.to_path_buf(),
            source: NpzError::NpyzRead(e),
        })?
        .ok_or(Error::MissingArray {
            path: path.to_path_buf(),
            key,
        })
}

fn read_f64<R: Read>(
    npy: NpyFile<R>,
    path: &Path,
    _key: &'static str,
) -> Result<Vec<f64>, Error> {
    npy.into_vec::<f64>().map_err(|e| Error::LoadNpz {
        path: path.to_path_buf(),
        source: NpzError::NpyzRead(e),
    })
}

// `expected: &'static [usize]` in the error type forces these to be
// `&'static`, so we precompute the small set we actually use.
fn vector_shape_for(n: usize) -> &'static [usize] {
    match n {
        128 => &[128],
        256 => &[256],
        _ => &[], // unreachable for the dimensions we use
    }
}
fn matrix_shape_for(r: usize, c: usize) -> &'static [usize] {
    match (r, c) {
        (128, 128) => &[128, 128],
        (256, 128) => &[256, 128],
        _ => &[],
    }
}

// Allow the compiler to error if PathBuf isn't used (it is — through Error variants).
#[allow(dead_code)]
fn _path_buf_lint_anchor(_p: PathBuf) {}
```

(The npyz API has changed between minor versions — Task 2 Step 1's confirmation lets this body be adjusted before commit. The shape and ordering invariants — C-order data, row-by-row matrix layout — must be preserved either way.)

- [ ] **Step 3: Verify it compiles**

```bash
cargo build --lib 2>&1 | tail -20
```

Expected: clean build. If npyz's API differs, fix the call sites; the structure stays.

- [ ] **Step 4: Smoke-test loading**

Add a temporary inline test (will be replaced by `tests.rs` later):

```rust
// at the bottom of loader.rs, or in a fresh `cargo test plda::loader`
#[cfg(test)]
mod loader_tests {
    use super::*;
    use std::path::PathBuf;

    fn repo_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    }

    #[test]
    fn xvec_npz_loads_with_expected_shapes() {
        let path = repo_root().join("models/plda/xvec_transform.npz");
        if !path.exists() {
            eprintln!("skip: {} not present", path.display());
            return;
        }
        let w = load_xvec(&path).expect("xvec npz loads");
        assert_eq!(w.mean1.len(), 256);
        assert_eq!(w.mean2.len(), 128);
        assert_eq!(w.lda.shape(), (256, 128));
    }

    #[test]
    fn plda_npz_loads_with_expected_shapes() {
        let path = repo_root().join("models/plda/plda.npz");
        if !path.exists() {
            eprintln!("skip: {} not present", path.display());
            return;
        }
        let w = load_plda(&path).expect("plda npz loads");
        assert_eq!(w.mu.len(), 128);
        assert_eq!(w.tr.shape(), (128, 128));
        assert_eq!(w.psi.len(), 128);
    }
}
```

Run:

```bash
cargo test --lib plda::loader 2>&1 | tail -10
```

Expected: 2 passed. If `npyz` returns a row-major storage but nalgebra's `from_row_slice` expects something different, sanity-check by also asserting `w.mean1[0]` matches a Python-printed reference value. (That guards against silent transposition of `lda`.)

- [ ] **Step 5: Sanity-check first matrix values match Python**

```bash
python3 -c "
import numpy as np
x = np.load('models/plda/xvec_transform.npz')
print('lda[0,0]:', x['lda'][0, 0])
print('lda[0,1]:', x['lda'][0, 1])
print('lda[1,0]:', x['lda'][1, 0])
print('lda[255,127]:', x['lda'][255, 127])
print('mean1[0]:', x['mean1'][0])
print('mean1[255]:', x['mean1'][255])
"
```

Add an assertion in the loader test that the first/last elements of `mean1` and the corner elements of `lda` (specifically `lda[0,0]` and `lda[1,0]`) match those values within `1e-12`. The point is to catch silent row/column swaps in `DMatrix::from_row_slice` — if the matrix is loaded transposed, `lda[0,1]` would land where `lda[1,0]` should be.

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml src/plda/loader.rs
git commit -m "plda: load xvec_transform.npz + plda.npz with shape checks"
```

---

## Task 3: `xvec_transform` + parity test

**Files:**
- Create: `src/plda/transform.rs` (initial version with only xvec_transform).
- Update: `src/plda/mod.rs` (re-export `PldaTransform`).

- [ ] **Step 1: Write `PldaTransform` skeleton with `xvec_transform`**

`src/plda/transform.rs`:

```rust
//! `PldaTransform` — the load-time setup + per-embedding projection.

use std::path::Path;

use nalgebra::{DMatrix, DVector};

use crate::plda::{
    error::Error,
    loader::{load_plda, load_xvec, PldaWeights, XvecWeights},
    EMBEDDING_DIMENSION, PLDA_DIMENSION,
};

/// Centering + LDA + whitening transform from `pyannote.audio`'s
/// `vbx_setup` (`pyannote/audio/utils/vbx.py:181-218`).
///
/// Built once from the two `.npz` weight files; thereafter
/// `xvec_transform` and `plda_transform` are pure read-only mappings.
pub struct PldaTransform {
    // xvec_tf factors
    mean1: DVector<f64>,
    mean2: DVector<f64>,
    lda: DMatrix<f64>,
    sqrt_in_dim: f64,   // sqrt(EMBEDDING_DIMENSION)
    sqrt_out_dim: f64,  // sqrt(PLDA_DIMENSION)

    // plda_tf factors (filled by Task 4 — the eigh).
    #[allow(dead_code)]
    plda_mu: DVector<f64>,
    #[allow(dead_code)]
    plda_tr_t: DMatrix<f64>,  // post-eigh, reversed, transposed for matmul
    #[allow(dead_code)]
    phi: DVector<f64>,
}

impl PldaTransform {
    /// Construct from disk. `xvec_path` is the
    /// `xvec_transform.npz`-equivalent file; `plda_path` is the
    /// `plda.npz` equivalent. Both ship under `models/plda/` for
    /// `pyannote/speaker-diarization-community-1`.
    pub fn from_npz_files(xvec_path: &Path, plda_path: &Path) -> Result<Self, Error> {
        let XvecWeights { mean1, mean2, lda } = load_xvec(xvec_path)?;
        let PldaWeights { mu, tr, psi } = load_plda(plda_path)?;

        // Phase-1 placeholder for the post-eigh state — Task 4 replaces
        // this with the actual generalized-eigh result.
        let plda_tr_t = DMatrix::<f64>::zeros(PLDA_DIMENSION, PLDA_DIMENSION);
        let phi = DVector::<f64>::from_iterator(PLDA_DIMENSION, psi.iter().copied());

        Ok(Self {
            mean1,
            mean2,
            lda,
            sqrt_in_dim: (EMBEDDING_DIMENSION as f64).sqrt(),
            sqrt_out_dim: (PLDA_DIMENSION as f64).sqrt(),
            plda_mu: mu,
            plda_tr_t,
            phi,
        })
    }

    /// First PLDA stage: center, scale by `sqrt(D_in)`, L2-normalize,
    /// apply `lda.T`, recenter, scale by `sqrt(D_out)`, L2-normalize.
    /// Matches `xvec_tf` in `utils/vbx.py:211-213`.
    ///
    /// `‖output‖ == sqrt(PLDA_DIMENSION) ≈ 11.31` (NOT 1.0).
    pub fn xvec_transform(&self, input: &[f32; EMBEDDING_DIMENSION]) -> [f64; PLDA_DIMENSION] {
        // Step 1: x - mean1 (256-d)
        let mut x = DVector::<f64>::from_iterator(
            EMBEDDING_DIMENSION,
            input.iter().map(|v| *v as f64),
        );
        x -= &self.mean1;

        // Step 2: l2_norm(x - mean1) * sqrt(D_in)
        l2_normalize_in_place(&mut x);
        x *= self.sqrt_in_dim;

        // Step 3: lda.T @ x  (output dim = 128)
        let mut y = self.lda.transpose() * &x;

        // Step 4: subtract mean2
        y -= &self.mean2;

        // Step 5: l2_norm(...) * sqrt(D_out)
        l2_normalize_in_place(&mut y);
        y *= self.sqrt_out_dim;

        let mut out = [0.0f64; PLDA_DIMENSION];
        for (o, v) in out.iter_mut().zip(y.iter()) {
            *o = *v;
        }
        out
    }
}

/// In-place L2 normalization. Identity for zero-norm input (matches
/// `pyannote.audio.utils.vbx.l2_norm`'s implicit divide; pyannote will
/// produce NaN on zero, but real WeSpeaker outputs are never exactly
/// zero, so divergence on this edge case is acceptable for parity).
fn l2_normalize_in_place(v: &mut DVector<f64>) {
    let n = v.norm();
    if n > 0.0 {
        *v /= n;
    }
}
```

Note: the `lda.transpose() * &x` allocates an intermediate. If profiling later flags this as hot, replace with `lda.tr_mul(&x)` (no transpose copy). Per the YAGNI rule, do that only after measurement.

- [ ] **Step 2: Re-export from `mod.rs`**

Already done in Task 1 (`pub use transform::PldaTransform;`). No change.

- [ ] **Step 3: Add the parity test**

The Phase-0 capture wrote `tests/parity/fixtures/01_dialogue/raw_embeddings.npz` (`(218, 3, 256)` f32) and `plda_embeddings.npz` containing `post_xvec` (`(195, 128)` f64) plus the `train_chunk_idx` / `train_speaker_idx` arrays that index into raw_embeddings. The test loads all three, runs Rust `xvec_transform` on each train embedding, and compares to `post_xvec` element-wise.

Create `tests/parity_plda.rs`:

```rust
//! Parity tests for `dia::plda` against the Phase-0 captured artifacts.
//!
//! Skipped (with eprintln warning) if the captured artifacts are not
//! present — Phase 0 must have been run for these to exist.

use std::path::PathBuf;

use dia::plda::{PldaTransform, EMBEDDING_DIMENSION, PLDA_DIMENSION};
use nalgebra::DMatrix;
use npyz::npz::NpzArchive;
use std::{fs::File, io::BufReader};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn fixture(p: &str) -> PathBuf {
    repo_root().join(p)
}

fn fixtures_present() -> bool {
    [
        "models/plda/xvec_transform.npz",
        "models/plda/plda.npz",
        "tests/parity/fixtures/01_dialogue/raw_embeddings.npz",
        "tests/parity/fixtures/01_dialogue/plda_embeddings.npz",
    ]
    .iter()
    .all(|p| repo_root().join(p).exists())
}

fn load_npz_f32_3d(
    path: &PathBuf,
    key: &str,
) -> (Vec<f32>, Vec<u64>) {
    let f = File::open(path).expect("open raw_embeddings.npz");
    let mut z = NpzArchive::new(BufReader::new(f)).expect("read npz");
    let npy = z.by_name(key).expect("by_name").expect("array present");
    let shape = npy.shape().to_vec();
    let data: Vec<f32> = npy.into_vec().expect("read f32 array");
    (data, shape)
}

fn load_npz_f64_2d(path: &PathBuf, key: &str) -> (DMatrix<f64>, (usize, usize)) {
    let f = File::open(path).expect("open npz");
    let mut z = NpzArchive::new(BufReader::new(f)).expect("read npz");
    let npy = z.by_name(key).expect("by_name").expect("array present");
    let shape = npy.shape().to_vec();
    assert_eq!(shape.len(), 2);
    let (rows, cols) = (shape[0] as usize, shape[1] as usize);
    let data: Vec<f64> = npy.into_vec().expect("read f64 array");
    (DMatrix::from_row_slice(rows, cols, &data), (rows, cols))
}

fn load_npz_i64_1d(path: &PathBuf, key: &str) -> Vec<i64> {
    let f = File::open(path).expect("open npz");
    let mut z = NpzArchive::new(BufReader::new(f)).expect("read npz");
    let npy = z.by_name(key).expect("by_name").expect("array present");
    npy.into_vec().expect("read i64 array")
}

#[test]
fn xvec_transform_matches_pyannote_on_train_embeddings() {
    if !fixtures_present() {
        eprintln!("skip: Phase-0 fixtures not present");
        return;
    }

    let plda = PldaTransform::from_npz_files(
        &fixture("models/plda/xvec_transform.npz"),
        &fixture("models/plda/plda.npz"),
    )
    .expect("load PLDA");

    // Load (218, 3, 256) raw embeddings. C-order: index by chunk*3*256 + slot*256 + d.
    let raw_path = fixture("tests/parity/fixtures/01_dialogue/raw_embeddings.npz");
    let (raw_flat, raw_shape) = load_npz_f32_3d(&raw_path, "embeddings");
    let (chunks, slots, dim) = (
        raw_shape[0] as usize,
        raw_shape[1] as usize,
        raw_shape[2] as usize,
    );
    assert_eq!(dim, EMBEDDING_DIMENSION);

    // Load expected (195, 128) post-xvec + train indices.
    let plda_emb_path = fixture("tests/parity/fixtures/01_dialogue/plda_embeddings.npz");
    let (post_xvec_expected, (n_train, post_dim)) =
        load_npz_f64_2d(&plda_emb_path, "post_xvec");
    assert_eq!(post_dim, PLDA_DIMENSION);
    let train_chunk_idx = load_npz_i64_1d(&plda_emb_path, "train_chunk_idx");
    let train_speaker_idx = load_npz_i64_1d(&plda_emb_path, "train_speaker_idx");
    assert_eq!(train_chunk_idx.len(), n_train);
    assert_eq!(train_speaker_idx.len(), n_train);

    // Run xvec_transform on each (chunk, slot) train pair and compare.
    let mut max_abs_err = 0.0f64;
    for i in 0..n_train {
        let c = train_chunk_idx[i] as usize;
        let s = train_speaker_idx[i] as usize;
        assert!(c < chunks && s < slots);
        let off = (c * slots + s) * dim;
        let mut input = [0.0f32; EMBEDDING_DIMENSION];
        input.copy_from_slice(&raw_flat[off..off + EMBEDDING_DIMENSION]);

        let actual = plda.xvec_transform(&input);
        for d in 0..PLDA_DIMENSION {
            let want = post_xvec_expected[(i, d)];
            let got = actual[d];
            let err = (want - got).abs();
            if err > max_abs_err {
                max_abs_err = err;
            }
        }
    }

    // Tolerance: pyannote does the entire computation in f64; our Rust
    // port matches except for f32→f64 promotion of the input. The
    // expected error floor is ~1e-7 (single-precision input → f64
    // computation). 1e-5 is comfortably above that.
    assert!(
        max_abs_err < 1e-5,
        "xvec_transform parity failed: max_abs_err = {max_abs_err}"
    );
    eprintln!(
        "[parity_plda] xvec_transform max_abs_err = {max_abs_err:.3e} \
         over {n_train} embeddings"
    );
}
```

- [ ] **Step 4: Run the parity test**

```bash
cargo test --test parity_plda xvec_transform 2>&1 | tail -20
```

Expected: pass with `max_abs_err < 1e-5`. If it fails:
- A larger error (e.g. 1e-2) suggests a fundamental algorithm mismatch — re-read `utils/vbx.py:211-213` and check the order of operations.
- A close-but-no-cigar error (e.g. 5e-5) might be a per-operation accumulation; bump the tolerance or investigate.

- [ ] **Step 5: Run all lib tests to confirm no regression**

```bash
cargo test --lib 2>&1 | tail -5
```

Expected: 238 passed (no change vs the merged Phase-0 baseline) plus our new loader tests.

- [ ] **Step 6: Commit**

```bash
git add src/plda/transform.rs tests/parity_plda.rs
git commit -m "plda: xvec_transform + parity test against captured post_xvec"
```

---

## Task 4: `plda_transform` (with generalized eigh) + parity test

**Files:**
- Modify: `src/plda/transform.rs` (replace the `plda_tr_t = zeros(...)` placeholder).
- Modify: `tests/parity_plda.rs` (add a second test for `plda_transform`).

- [ ] **Step 1: Add the eigh helper**

In `src/plda/transform.rs`, before `impl PldaTransform`:

```rust
/// Solve the generalized symmetric eigenvalue problem `B*v = λ*W*v`,
/// returning eigenvalues and eigenvectors sorted DESCENDING by λ.
/// Matches scipy `eigh(B, W)` followed by `[::-1]` reversal.
///
/// `W` must be symmetric positive-definite (we Cholesky-decompose it).
fn generalized_eigh_descending(
    b: &DMatrix<f64>,
    w: &DMatrix<f64>,
) -> Result<(DVector<f64>, DMatrix<f64>), Error> {
    use nalgebra::{Cholesky, SymmetricEigen};

    let n = b.nrows();
    debug_assert_eq!(b.ncols(), n);
    debug_assert_eq!(w.shape(), (n, n));

    // Step 1: Cholesky W = L L^T.
    let chol = Cholesky::new(w.clone()).ok_or(Error::WNotPositiveDefinite)?;
    let l = chol.l();

    // Step 2: M = L^{-1} B L^{-T}, computed via two triangular solves.
    // Using nalgebra's `solve_lower_triangular` and friends keeps it
    // O(n^3) without forming L^{-1} explicitly.
    // Right-solve: L X = B  =>  X = L^{-1} B.
    let x = l
        .clone()
        .solve_lower_triangular(b)
        .expect("L is unit lower-triangular by Cholesky construction");
    // Left-solve: M^T L^T = X^T  =>  M = (L^{-T} X^T)^T = (L^{-T})X… hmm,
    // simpler: M = X * L^{-T}  =>  L^T M^T = X^T.
    let m_t = l
        .transpose()
        .solve_lower_triangular(&x.transpose())
        .expect("L^T is upper-triangular; solve via L^T y = z");
    // The previous solve_lower_triangular call on L^T does not actually
    // work — L^T is upper-triangular. Use the upper-triangular solver:
    let l_t = l.transpose();
    let m = l_t
        .solve_upper_triangular(&x.transpose())
        .expect("upper-triangular solve")
        .transpose();

    // Step 3: Symmetric eigen of M.
    let SymmetricEigen {
        eigenvalues,
        eigenvectors,
    } = SymmetricEigen::new(m);

    // Step 4: Recover X = L^{-T} Y. Solve L^T X = Y.
    let x_recovered = l_t
        .solve_upper_triangular(&eigenvectors)
        .expect("upper-triangular solve");

    // Step 5: Sort descending by eigenvalue.
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        eigenvalues[b]
            .partial_cmp(&eigenvalues[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut sorted_vals = DVector::<f64>::zeros(n);
    let mut sorted_vecs = DMatrix::<f64>::zeros(n, n);
    for (out_col, &src_col) in idx.iter().enumerate() {
        sorted_vals[out_col] = eigenvalues[src_col];
        sorted_vecs.set_column(out_col, &x_recovered.column(src_col));
    }

    Ok((sorted_vals, sorted_vecs))
}
```

(The body above mixes lower/upper-triangular solves; double-check by stepping through with a 2x2 test before relying on it. nalgebra's API is `lhs.solve_lower_triangular(&rhs)` which solves `lhs * X = rhs` with `lhs` lower-triangular.)

- [ ] **Step 2: Replace the `plda_tr_t = zeros(...)` placeholder**

In `PldaTransform::from_npz_files`, replace the placeholder block with:

```rust
        // Generalized eigh setup.
        // W = inv(tr.T @ tr)  — within-class precision.
        // B = inv((tr.T / psi) @ tr)  — between-class precision (psi rescales rows of tr.T).
        // Then `eigh(B, W)` produces the whitening eigenvectors `wccn`
        // and corresponding eigenvalues. We reverse to descending order.

        let tr_t = tr.transpose();
        let w_mat = (&tr_t * &tr)
            .try_inverse()
            .ok_or(Error::WNotPositiveDefinite)?;

        // (tr.T / psi).dot(tr): scale each row of tr.T by 1/psi[i], then
        // matmul with tr. Equivalent to (diag(1/psi) @ tr.T) @ tr.
        let mut tr_t_scaled = tr_t.clone();
        for i in 0..PLDA_DIMENSION {
            for j in 0..PLDA_DIMENSION {
                tr_t_scaled[(i, j)] /= psi[i];
            }
        }
        let b_mat_inv = &tr_t_scaled * &tr;
        let b_mat = b_mat_inv
            .try_inverse()
            .ok_or(Error::WNotPositiveDefinite)?;

        let (eigenvalues_desc, eigenvectors_desc) =
            generalized_eigh_descending(&b_mat, &w_mat)?;

        // pyannote: `plda_tr = wccn.T[::-1]`. wccn columns are
        // eigenvectors; .T flips to rows; [::-1] reverses row order to
        // descending. The reversal-then-transpose is equivalent to
        // taking the descending-sorted eigenvectors as ROWS — i.e.
        // eigenvectors_desc.transpose() if the eigenvectors are
        // column-stored.
        //
        // Then `plda_tf` does `(x - mu) @ plda_tr.T`. Substituting:
        // `plda_tr.T = ((wccn.T[::-1])).T = wccn[:, ::-1]` — eigenvectors
        // in descending column order. Multiplying x by this is exactly
        // `eigenvectors_desc * (x - mu)` if x is treated as a column,
        // OR `(x - mu).T @ eigenvectors_desc` if row.
        //
        // We store `plda_tr_t = eigenvectors_desc` (so multiplying a
        // 128-row, 1-col centered vector by it produces a 128-row
        // result — see plda_transform body below).
        let plda_tr_t = eigenvectors_desc;
        let phi = eigenvalues_desc;
```

**Critical note on the `plda_tr.T` stored convention.** Working out the algebra:
- pyannote: `plda_tr_python = wccn.T[::-1]` → shape `(128, 128)`, descending eigenvectors as **rows**.
- `plda_tf`: `(x0 - mu) @ plda_tr_python.T = (x0 - mu) @ (wccn.T[::-1]).T`.
- `(wccn.T[::-1]).T == wccn[:, ::-1]` — descending eigenvectors as **columns**.
- So the matmul is `(x0 - mu) @ <eigenvectors-as-columns>`.
- Treating `(x0 - mu)` as a row vector: row × matrix = row. Output is `(x0 - mu)` projected onto each eigenvector → 128-d row.
- Equivalent: as a column vector, `<eigenvectors-as-columns>.T @ (x0 - mu) = eigenvectors-as-rows @ (x0 - mu)`.
- We store `plda_tr_t = eigenvectors_desc` (columns = descending eigenvectors). The transform body uses `plda_tr_t.transpose() * &(x - mu)` for the standard column-vector interpretation. Verified against Python in Step 4.

- [ ] **Step 3: Add `plda_transform`**

After `xvec_transform` in `impl PldaTransform`:

```rust
    /// Second PLDA stage: center by `plda_mu`, project via the
    /// eigenvectors of the generalized eigenvalue problem
    /// `eigh(B, W)` (descending). Matches `plda_tf` in
    /// `utils/vbx.py:215-217`.
    ///
    /// Output is whitened (NOT L2-normed). Input is the output of
    /// [`Self::xvec_transform`].
    pub fn plda_transform(&self, post_xvec: &[f64; PLDA_DIMENSION]) -> [f64; PLDA_DIMENSION] {
        let mut x = DVector::<f64>::from_iterator(
            PLDA_DIMENSION,
            post_xvec.iter().copied(),
        );
        x -= &self.plda_mu;

        // out = eigenvectors_desc.T * x   (column-vector convention)
        // Equivalent to row-vector x @ eigenvectors_desc in pyannote.
        let y = self.plda_tr_t.transpose() * &x;

        let mut out = [0.0f64; PLDA_DIMENSION];
        for (o, v) in out.iter_mut().zip(y.iter()) {
            *o = *v;
        }
        out
    }

    /// Eigenvalue diagonal `phi` consumed by VBx (Phase 2). Same as
    /// pyannote's `PLDA.phi` — the descending-sorted generalized
    /// eigenvalues, sliced to the first `PLDA_DIMENSION` (no-op for
    /// the standard 128-d case).
    pub fn phi(&self) -> &[f64] {
        self.phi.as_slice()
    }
```

- [ ] **Step 4: Add the parity test for `plda_transform`**

In `tests/parity_plda.rs`, after the xvec test:

```rust
#[test]
fn plda_transform_matches_pyannote_modulo_signs() {
    if !fixtures_present() {
        eprintln!("skip: Phase-0 fixtures not present");
        return;
    }

    let plda = PldaTransform::from_npz_files(
        &fixture("models/plda/xvec_transform.npz"),
        &fixture("models/plda/plda.npz"),
    )
    .expect("load PLDA");

    let plda_emb_path = fixture("tests/parity/fixtures/01_dialogue/plda_embeddings.npz");
    // We use the captured post_xvec as input to plda_transform — that
    // way we test plda_transform in isolation (any xvec drift is
    // already validated in the xvec test).
    let (post_xvec_in, (n_train, _)) = load_npz_f64_2d(&plda_emb_path, "post_xvec");
    let (post_plda_expected, _) = load_npz_f64_2d(&plda_emb_path, "post_plda");

    // Per-element absolute-value comparison (sign-invariant), AND the
    // Gram matrix comparison for additional rigor (eigenvector signs
    // can flip per column independently between LAPACK implementations).
    let mut per_elem_abs_max_err = 0.0f64;
    let mut rust_post_plda = nalgebra::DMatrix::<f64>::zeros(n_train, PLDA_DIMENSION);
    for i in 0..n_train {
        let mut input = [0.0f64; PLDA_DIMENSION];
        for d in 0..PLDA_DIMENSION {
            input[d] = post_xvec_in[(i, d)];
        }
        let actual = plda.plda_transform(&input);
        for d in 0..PLDA_DIMENSION {
            let want = post_plda_expected[(i, d)];
            let got = actual[d];
            let err = (want.abs() - got.abs()).abs();
            if err > per_elem_abs_max_err {
                per_elem_abs_max_err = err;
            }
            rust_post_plda[(i, d)] = got;
        }
    }
    assert!(
        per_elem_abs_max_err < 1e-4,
        "plda_transform |abs| parity failed: max err = {per_elem_abs_max_err:.3e}"
    );

    // Gram matrix: G_rust = post_plda_rust @ post_plda_rust.T,
    // G_python = post_plda_python @ post_plda_python.T. Sign flips in
    // any eigenvector column cancel in G.
    let g_rust = &rust_post_plda * rust_post_plda.transpose();
    let g_py = &post_plda_expected * post_plda_expected.transpose();
    let mut gram_max_err = 0.0f64;
    for i in 0..n_train {
        for j in 0..n_train {
            let err = (g_rust[(i, j)] - g_py[(i, j)]).abs();
            if err > gram_max_err {
                gram_max_err = err;
            }
        }
    }
    assert!(
        gram_max_err < 1e-3,
        "plda_transform Gram-matrix parity failed: max err = {gram_max_err:.3e}"
    );
    eprintln!(
        "[parity_plda] plda_transform |abs| max err = {per_elem_abs_max_err:.3e}, \
         Gram max err = {gram_max_err:.3e}"
    );
}
```

The Gram tolerance is looser than per-element because Gram entries are sums of `n_train * 128` products, so accumulated float error scales with the matrix size. 1e-3 is comfortable for n_train ≈ 200.

- [ ] **Step 5: Run + iterate**

```bash
cargo test --test parity_plda 2>&1 | tail -30
```

Expected: both tests pass. If the per-element abs test passes but the Gram one fails, look for a missing `[::-1]` (eigenvalue/eigenvector ordering) or a transposition bug. If both fail, the eigh helper is wrong — write a 3x3 Python+Rust crosscheck to localize.

- [ ] **Step 6: Commit**

```bash
git add src/plda/transform.rs tests/parity_plda.rs
git commit -m "plda: plda_transform via generalized eigh + parity test"
```

---

## Task 5: Module-level test consolidation + doc comments

**Files:** Create `src/plda/tests.rs`. Polish doc comments in `mod.rs` / `transform.rs`.

- [ ] **Step 1: Move loader smoke tests into `src/plda/tests.rs`**

Pull the inline `mod loader_tests` from Task 2 into a shared `src/plda/tests.rs`:

```rust
//! Unit tests for `dia::plda`. Heavy parity tests live in
//! `tests/parity_plda.rs`; this module covers the small, model-free
//! invariants that the load/transform paths must hold for any input.

use std::path::PathBuf;

use crate::plda::{
    loader::{load_plda, load_xvec},
    PldaTransform, EMBEDDING_DIMENSION, PLDA_DIMENSION,
};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn xvec_path() -> PathBuf {
    repo_root().join("models/plda/xvec_transform.npz")
}

fn plda_path() -> PathBuf {
    repo_root().join("models/plda/plda.npz")
}

fn weights_present() -> bool {
    xvec_path().exists() && plda_path().exists()
}

#[test]
fn xvec_npz_shapes_match_pyannote_4_0_4() {
    if !weights_present() {
        eprintln!("skip: PLDA weights missing under models/plda/");
        return;
    }
    let w = load_xvec(&xvec_path()).expect("xvec npz");
    assert_eq!(w.mean1.len(), 256);
    assert_eq!(w.mean2.len(), 128);
    assert_eq!(w.lda.shape(), (256, 128));
}

#[test]
fn plda_npz_shapes_match_pyannote_4_0_4() {
    if !weights_present() {
        eprintln!("skip: PLDA weights missing under models/plda/");
        return;
    }
    let w = load_plda(&plda_path()).expect("plda npz");
    assert_eq!(w.mu.len(), 128);
    assert_eq!(w.tr.shape(), (128, 128));
    assert_eq!(w.psi.len(), 128);
}

/// `xvec_transform` output norm is `sqrt(PLDA_DIMENSION) ≈ 11.31` —
/// see `utils/vbx.py:211-213`. Test guards against silent regressions
/// where the outer `sqrt(D_out)` factor is dropped.
#[test]
fn xvec_transform_norm_is_sqrt_d_out() {
    if !weights_present() {
        eprintln!("skip: PLDA weights missing");
        return;
    }
    let plda = PldaTransform::from_npz_files(&xvec_path(), &plda_path())
        .expect("load PLDA");

    // Use a non-trivial input (all 0.1 values, then scaled to a
    // reasonable WeSpeaker-like magnitude after L2-norm in xvec_tf).
    let input = [0.1f32; EMBEDDING_DIMENSION];
    let out = plda.xvec_transform(&input);
    let norm: f64 = out.iter().map(|v| v * v).sum::<f64>().sqrt();
    let expected = (PLDA_DIMENSION as f64).sqrt();
    assert!(
        (norm - expected).abs() < 1e-6,
        "xvec output norm = {norm}, expected {expected}"
    );
}

/// `phi` (eigenvalues) must be sorted DESCENDING after the
/// generalized-eigh + reversal step.
#[test]
fn phi_is_sorted_descending() {
    if !weights_present() {
        eprintln!("skip: PLDA weights missing");
        return;
    }
    let plda = PldaTransform::from_npz_files(&xvec_path(), &plda_path())
        .expect("load PLDA");
    let phi = plda.phi();
    assert_eq!(phi.len(), PLDA_DIMENSION);
    for w in phi.windows(2) {
        assert!(
            w[0] >= w[1],
            "phi not descending: {:?} < {:?}",
            w[0],
            w[1]
        );
    }
}
```

Remove the loader-internal `#[cfg(test)] mod loader_tests` from `loader.rs` — it's superseded by `tests.rs`.

- [ ] **Step 2: Run all module tests**

```bash
cargo test --lib plda 2>&1 | tail -10
```

Expected: 4 passed (`xvec_npz_shapes...`, `plda_npz_shapes...`, `xvec_transform_norm...`, `phi_is_sorted_descending`).

- [ ] **Step 3: Polish public docs**

In `src/plda/mod.rs`, ensure the top-level docstring covers:
- What this module is (PLDA transform from pyannote).
- The two-stage pipeline diagram.
- Where the weight files come from (`models/plda/`, CC-BY-4.0).
- That the implementation is pinned to pyannote.audio 4.0.4 — Phase 0 is the canonical parity reference.
- Cross-link to `Phase 1 plan doc` for context.

- [ ] **Step 4: Run the full test suite (lib + integration)**

```bash
cargo test 2>&1 | tail -15
```

Expected: all lib tests + parity_plda tests pass. If any pre-existing test broke, investigate before committing.

- [ ] **Step 5: Run clippy**

```bash
cargo clippy --lib --all-targets 2>&1 | tail -10
```

Expected: clean. Common warnings to watch for: needless `clone()` (the eigh helper has several intentional ones — annotate or refactor), unused imports.

- [ ] **Step 6: Commit**

```bash
git add src/plda/tests.rs src/plda/mod.rs src/plda/loader.rs src/plda/transform.rs
git commit -m "plda: consolidate module tests + polish public docs"
```

---

## Task 6: Push + open PR

- [ ] **Step 1: Push**

```bash
git push -u origin feat/phase1-plda
```

- [ ] **Step 2: Open PR (target `0.1.0`, matching Phase-0 convention)**

```bash
gh pr create --base 0.1.0 --head feat/phase1-plda \
  --title "plda: Rust port of pyannote vbx_setup + xvec_tf + plda_tf (Phase 1)" \
  --body "$(cat <<'EOF'
## Summary

Implements **Phase 1** of the Option-A pyannote-parity work
(`docs/superpowers/plans/2026-04-28-dia-phase1-plda-rust-port.md`).

This PR adds the new `dia::plda` module: a standalone Rust port of
`pyannote.audio.utils.vbx.vbx_setup` + the inner `xvec_tf` and `plda_tf`
lambdas (`utils/vbx.py:181-218` in pyannote.audio 4.0.4). The module is
not yet integrated into `Diarizer` — that wiring lands in a later phase.

## What's in the module

- `PldaTransform::from_npz_files(xvec_path, plda_path)` — loads the
  six numpy arrays from the two `.npz` weight files, runs the
  generalized eigenvalue solve `eigh(B, W)` once, and stores the
  descending-sorted whitening matrix + eigenvalue diagonal `phi`.
- `xvec_transform(&[f32; 256]) -> [f64; 128]` — center, scale by
  `sqrt(256)`, L2-norm, apply `lda.T`, recenter, scale by `sqrt(128)`,
  L2-norm. Output `‖·‖ = sqrt(128)`.
- `plda_transform(&[f64; 128]) -> [f64; 128]` — center by `mu`,
  project onto descending eigenvectors. Output is whitened (NOT
  L2-normed) and is what VBx will consume in Phase 2.
- `phi() -> &[f64]` — descending eigenvalues, length 128.

The generalized eigh is implemented via Cholesky reduction:
1. `W = L L^T` (Cholesky).
2. `M = L^{-1} B L^{-T}` (two triangular solves).
3. `M = Y Λ Y^T` (`SymmetricEigen`).
4. `X = L^{-T} Y` (recover original eigenvectors).
5. Sort descending by `λ`, reorder eigenvectors to match.

## Parity validation

`tests/parity_plda.rs` loads the Phase-0 captured artifacts
(`tests/parity/fixtures/01_dialogue/{raw_embeddings,plda_embeddings}.npz`)
and asserts:
- **xvec_transform**: per-element absolute error vs `post_xvec` over all
  195 train embeddings, max < 1e-5.
- **plda_transform**: per-element |abs| match vs `post_plda` < 1e-4
  (sign-invariant — eigenvectors are unique only up to scalar sign);
  Gram-matrix match < 1e-3 (rigorous sign-invariant whitening parity).

Plus 4 model-free unit tests in `src/plda/tests.rs` covering shape
checks, output norm = sqrt(D_out), and `phi` descending order.

## Dependencies

Adds `npyz` for `.npz` reading. Uses `nalgebra` (already a dep) for
matrices, Cholesky, and SymmetricEigen.

## Test plan

- [ ] `cargo test --lib plda` — module unit tests
- [ ] `cargo test --test parity_plda` — captured-artifact parity tests
- [ ] `cargo clippy --lib --all-targets` — clean
- [ ] `cargo test` — no pre-existing test broken

## Out of scope

- Phase 2: VBx HMM port. Will consume `phi()` and `plda_transform`.
- Phase 3: constrained Hungarian.
- Phase 4: centroid AHC.
- Wiring PLDA into `Diarizer` end-to-end. Done after Phase 4.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 3: Link to Phase 0**

In the PR body or as a top-level comment, link to PR #1 (Phase 0) and to the analysis doc that motivates the whole effort.

---

## Self-review

**1. Spec coverage.** Phase 1 of the analysis doc requires porting PLDA. The plan covers:
- Loading both `.npz` files (Task 2).
- `xvec_transform` matching `xvec_tf` (Task 3) — verified against captured `post_xvec`.
- `plda_transform` matching `plda_tf` (Task 4) — verified against captured `post_plda` modulo eigenvector signs.
- `phi` exposed for Phase 2 (Task 4).
- Module-level smoke tests (Task 5).
- PR (Task 6).

**2. Placeholder scan.** No `<placeholder>`, `TBD`, `# Replace this with...`. The "double-check this" notes in Task 4's eigh helper (mixing lower/upper triangular solves) are real review-anchor flags — running the test catches errors. Not the same as deferred design.

**3. Type consistency.** Public types are stable across tasks: `PldaTransform`, `EMBEDDING_DIMENSION`, `PLDA_DIMENSION`, `Error`, function signatures. The internal `XvecWeights` / `PldaWeights` are `pub(super)`. The private `generalized_eigh_descending` returns `(DVector<f64>, DMatrix<f64>)` consumed only by `from_npz_files`.

**4. Risk areas explicitly called out:**
- Eigenvector sign indeterminacy — the parity test compares `|abs|` AND Gram matrix (Task 4 Step 4).
- Triangular-solve direction confusion — flagged inline; the parity test catches it (Task 4 Step 1, "double-check by stepping through with a 2x2 test").
- npyz API drift — Task 1 confirms current version; Task 2 adapts.
- DMatrix row/col-major confusion — Task 2 Step 5 cross-checks against Python-printed corner values.

**5. Dimensions consistency.** Verified against the actual loaded arrays:
`mean1: 256`, `mean2: 128`, `lda: 256×128`, `mu: 128`, `tr: 128×128`, `psi: 128`, `phi: 128 (descending)`.

---

**Plan complete.** Phase 2 (VBx) will be planned after this lands; it inherits trust in Phase 1's parity contract.
