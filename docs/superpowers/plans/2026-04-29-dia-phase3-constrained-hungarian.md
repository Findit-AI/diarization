# Phase 3 — Constrained Hungarian Assignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port pyannote's `constrained_argmax` (a per-chunk maximum-weight bipartite matching that maps speakers → clusters) to Rust, validated at exact equality against the Phase-0 captured `hard_clusters` fixture.

**Architecture:** Use the `pathfinding` crate's `kuhn_munkres` (O(n³) Hungarian). Wrap `f64` costs in `ordered_float::NotNan` for the `Ord` bound. When `num_speakers > num_clusters` (the captured fixture's case: 3 × 2), transpose the cost matrix so rows ≤ columns (the crate's hard requirement) and invert the assignment afterwards.

**Tech Stack:** `pathfinding = "4.15"`, `ordered-float = "5.3"`, `nalgebra` (existing), `npyz` (existing), pyannote 4.0.4 reference.

---

## Pyannote 4.0.4 reference (the contract we must match)

`pyannote/audio/pipelines/clustering.py:127-140`:

```python
def constrained_argmax(self, soft_clusters: np.ndarray) -> np.ndarray:
    soft_clusters = np.nan_to_num(soft_clusters, nan=np.nanmin(soft_clusters))
    num_chunks, num_speakers, num_clusters = soft_clusters.shape

    hard_clusters = -2 * np.ones((num_chunks, num_speakers), dtype=np.int8)

    for c, cost in enumerate(soft_clusters):
        speakers, clusters = linear_sum_assignment(cost, maximize=True)
        for s, k in zip(speakers, clusters):
            hard_clusters[c, s] = k

    return hard_clusters
```

Key facts:
- `linear_sum_assignment(cost, maximize=True)` with `cost.shape == (num_speakers, num_clusters)` returns `min(num_speakers, num_clusters)` matched `(speaker, cluster)` pairs.
- Unmatched speakers (when `num_speakers > num_clusters`) keep the sentinel `-2`.
- NaN values in `soft_clusters` are pre-replaced with `np.nanmin(soft_clusters)` (the *global* min across chunks).
- Captured fixture: `soft_clusters.shape == (218, 3, 2)`, `hard_clusters ∈ {-2, 0, 1}`.

## Out of scope

- Centroid/AHC computation (Phase 4).
- Embedding-to-centroid distance (Phase 4 — soft_clusters is captured directly here).
- `Diarizer` integration (Phase 5).

---

## File Structure

- `src/hungarian/mod.rs` — module entry, `#[cfg(test)] mod tests; mod parity_tests;` plus `pub(crate)` re-exports. Crate-private until Phase 5.
- `src/hungarian/algo.rs` — `constrained_argmax` and helpers.
- `src/hungarian/error.rs` — `Error` enum (`Shape`, `NonFinite`).
- `src/hungarian/tests.rs` — model-free invariants (shape, NaN, distinct-cluster, edge cases).
- `src/hungarian/parity_tests.rs` — full-fixture exact-equality test against `clustering.npz`.
- `Cargo.toml` — add `pathfinding = "4.15"` and `ordered-float = "5.3"` to `[dependencies]`.
- `src/lib.rs` — add `mod hungarian;` (crate-private, no `pub use`; mirror `mod vbx;` placement).

---

## Task 0: Cargo deps + lib.rs wiring

**Files:**
- Modify: `Cargo.toml` (add `pathfinding = "4.15"` and `ordered-float = "5.3"` under `[dependencies]`)
- Modify: `src/lib.rs` (add `mod hungarian;` next to `mod vbx;`)
- Create: `src/hungarian/mod.rs`
- Create: `src/hungarian/error.rs`

- [ ] **Step 1: Add deps to Cargo.toml**

Find the alphabetically-sorted `[dependencies]` block. Insert:

```toml
ordered-float = "5.3"
pathfinding = "4.15"
```

Both crates are pure-Rust, no system libs. `pathfinding` pulls `num-traits` (already transitively used by nalgebra) and `indexmap`/`fxhash`.

- [ ] **Step 2: Verify deps resolve**

Run: `cargo check 2>&1 | tail -10`
Expected: dependencies fetched and compiled, no errors.

- [ ] **Step 3: Create src/hungarian/error.rs**

```rust
//! Errors for `diarization::hungarian`.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
  /// Input shape is invalid (e.g., 0 speakers, 0 clusters, mismatched dims).
  #[error("hungarian: shape error: {0}")]
  Shape(&'static str),
  /// A NaN/±inf entry was found in the cost matrix.
  #[error("hungarian: non-finite value in {0}")]
  NonFinite(&'static str),
}
```

- [ ] **Step 4: Create src/hungarian/mod.rs**

```rust
//! Constrained Hungarian assignment — per-chunk speaker → cluster mapping.
//!
//! Ports `pyannote.audio.pipelines.clustering.SpeakerEmbedding.constrained_argmax`
//! (`clustering.py:127-140` in pyannote.audio 4.0.4) to Rust. Given a per-chunk
//! `(num_speakers, num_clusters)` cost matrix (typically `2 - cosine_distance`
//! between embeddings and centroids), returns the maximum-weight bipartite
//! matching as `Vec<i32>` of length `num_speakers`. Unmatched speakers
//! (possible when `num_speakers > num_clusters`) carry the sentinel `-2`.
//!
//! ## Standalone — no `Diarizer` integration yet
//!
//! Phase 3 ships this as a pure-math module. The integration
//! (`Diarizer` consuming VBx + Hungarian → centroid AHC → per-frame diarization)
//! lands in Phase 5. Until then `diarization::hungarian` is crate-private.

#![allow(dead_code, unused_imports)]

mod algo;
mod error;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod parity_tests;

pub use algo::{UNMATCHED, constrained_argmax};
pub use error::Error;
```

- [ ] **Step 5: Wire into lib.rs**

Find the `mod vbx;` line in `src/lib.rs`. Add directly below it:

```rust
mod hungarian;
```

(crate-private — no `pub use` until Phase 5).

- [ ] **Step 6: Stub `src/hungarian/algo.rs` so `cargo check` passes**

```rust
//! Constrained Hungarian assignment (per-chunk maximum-weight matching).

use crate::hungarian::error::Error;
use nalgebra::DMatrix;

/// Sentinel value for an unmatched speaker. Matches pyannote's
/// `-2 * np.ones(...)` initializer in `constrained_argmax`.
pub const UNMATCHED: i32 = -2;

pub fn constrained_argmax(_soft_clusters: &DMatrix<f64>) -> Result<Vec<i32>, Error> {
  todo!("Task 2")
}
```

- [ ] **Step 7: Verify it builds**

Run: `cargo check 2>&1 | tail -5`
Expected: `Finished` with no errors (warnings about unused are fine — `#![allow(dead_code, unused_imports)]` covers them).

- [ ] **Step 8: Commit**

```bash
git add Cargo.toml Cargo.lock src/lib.rs src/hungarian
git commit -m "hungarian: scaffold module + add pathfinding/ordered-float deps"
```

---

## Task 1: Validation tests (write failing tests first)

**Files:**
- Create: `src/hungarian/tests.rs`

These tests exercise the boundary contract before the implementation lands. They will fail compilation/runtime now and start passing in Task 2.

- [ ] **Step 1: Write the model-free test module**

```rust
//! Model-free unit tests for `diarization::hungarian`.
//!
//! Heavy parity against pyannote's captured `hard_clusters` lives in
//! `src/hungarian/parity_tests.rs`. This module covers smaller invariants
//! that should hold for any input.

use crate::hungarian::{Error, UNMATCHED, constrained_argmax};
use nalgebra::DMatrix;

#[test]
fn rejects_empty_speakers() {
  let cost = DMatrix::<f64>::zeros(0, 3);
  let result = constrained_argmax(&cost);
  assert!(matches!(result, Err(Error::Shape(_))), "got {result:?}");
}

#[test]
fn rejects_empty_clusters() {
  let cost = DMatrix::<f64>::zeros(3, 0);
  let result = constrained_argmax(&cost);
  assert!(matches!(result, Err(Error::Shape(_))), "got {result:?}");
}

#[test]
fn rejects_nan_entry() {
  let mut cost = DMatrix::<f64>::from_element(2, 2, 0.5);
  cost[(0, 1)] = f64::NAN;
  let result = constrained_argmax(&cost);
  assert!(
    matches!(result, Err(Error::NonFinite(_))),
    "got {result:?}"
  );
}

#[test]
fn rejects_pos_inf_entry() {
  let mut cost = DMatrix::<f64>::from_element(2, 2, 0.5);
  cost[(1, 0)] = f64::INFINITY;
  let result = constrained_argmax(&cost);
  assert!(
    matches!(result, Err(Error::NonFinite(_))),
    "got {result:?}"
  );
}

#[test]
fn rejects_neg_inf_entry() {
  let mut cost = DMatrix::<f64>::from_element(2, 2, 0.5);
  cost[(0, 0)] = f64::NEG_INFINITY;
  let result = constrained_argmax(&cost);
  assert!(
    matches!(result, Err(Error::NonFinite(_))),
    "got {result:?}"
  );
}

/// Square 2x2 — direct kuhn_munkres path. Expected: speaker 0 → cluster 0
/// (0.9 vs 0.1), speaker 1 → cluster 1 (0.8 vs 0.2). Both must be matched.
#[test]
fn square_2x2_picks_diagonal_when_diagonal_dominates() {
  let cost = DMatrix::<f64>::from_row_slice(2, 2, &[0.9, 0.1, 0.2, 0.8]);
  let assign = constrained_argmax(&cost).expect("constrained_argmax");
  assert_eq!(assign, vec![0, 1]);
}

/// Square 2x2 anti-diagonal: speaker 0 → cluster 1 (0.9 vs 0.2),
/// speaker 1 → cluster 0 (0.8 vs 0.1). Hungarian must avoid the greedy trap.
#[test]
fn square_2x2_picks_anti_diagonal_when_off_diagonal_dominates() {
  let cost = DMatrix::<f64>::from_row_slice(2, 2, &[0.2, 0.9, 0.8, 0.1]);
  let assign = constrained_argmax(&cost).expect("constrained_argmax");
  assert_eq!(assign, vec![1, 0]);
}

/// Tall (S < K): 2 speakers, 3 clusters. Both speakers must be matched to
/// distinct clusters; the unused cluster index is just dropped.
#[test]
fn tall_2x3_assigns_both_speakers_to_distinct_clusters() {
  // speaker 0 prefers cluster 2 (1.0); speaker 1 prefers cluster 0 (0.9).
  let cost = DMatrix::<f64>::from_row_slice(2, 3, &[0.1, 0.5, 1.0, 0.9, 0.4, 0.3]);
  let assign = constrained_argmax(&cost).expect("constrained_argmax");
  assert_eq!(assign, vec![2, 0]);
  // No UNMATCHED — every speaker got a cluster.
  assert!(!assign.iter().any(|&v| v == UNMATCHED));
}

/// Wide (S > K): 3 speakers, 2 clusters. Two speakers must be matched to
/// distinct clusters; the third must be UNMATCHED. This is the captured
/// fixture's shape (3, 2) and exercises the transpose path.
#[test]
fn wide_3x2_leaves_one_speaker_unmatched() {
  // speaker 0: top in cluster 0 (0.95). speaker 1: top in cluster 1 (0.95).
  // speaker 2: weak in both (0.1, 0.1). Expected assignment 0→0, 1→1, 2→-2.
  let cost = DMatrix::<f64>::from_row_slice(
    3,
    2,
    &[0.95, 0.05, 0.05, 0.95, 0.10, 0.10],
  );
  let assign = constrained_argmax(&cost).expect("constrained_argmax");
  assert_eq!(assign, vec![0, 1, UNMATCHED]);
}

/// Wide (S > K) with the optimal assignment leaving a *different* speaker
/// unmatched than the trivially-weakest one. Speaker 2 has the highest
/// individual cell (0.99 in cluster 0), but assigning {2→0, 1→1} gives
/// 0.99 + 0.95 = 1.94, while {0→0, 1→1} gives 0.95 + 0.95 = 1.90, so
/// optimal leaves *speaker 0* unmatched. Catches a "leave the lowest-row
/// speaker unmatched" greedy bug.
#[test]
fn wide_3x2_optimal_unmatches_non_weakest_speaker() {
  let cost = DMatrix::<f64>::from_row_slice(
    3,
    2,
    &[0.95, 0.10, 0.05, 0.95, 0.99, 0.10],
  );
  let assign = constrained_argmax(&cost).expect("constrained_argmax");
  // Speaker 2 → cluster 0 (0.99), speaker 1 → cluster 1 (0.95),
  // speaker 0 unmatched.
  assert_eq!(assign, vec![UNMATCHED, 1, 0]);
}

/// Distinct-cluster invariant: every matched assignment must use a
/// different cluster index. Holds for square, tall, and wide shapes.
/// Property check on a 4x4 with deterministic costs.
#[test]
fn matched_speakers_are_assigned_distinct_clusters() {
  let cost = DMatrix::<f64>::from_fn(4, 4, |i, j| ((i * 7 + j * 13) % 17) as f64 * 0.1);
  let assign = constrained_argmax(&cost).expect("constrained_argmax");
  let mut used = std::collections::HashSet::new();
  for &k in &assign {
    if k != UNMATCHED {
      assert!(
        used.insert(k),
        "cluster {k} assigned twice in {assign:?}"
      );
    }
  }
  // 4x4 square — every speaker should be matched.
  assert!(!assign.iter().any(|&v| v == UNMATCHED));
}

/// Single speaker, single cluster: must assign 0 → 0.
#[test]
fn single_speaker_single_cluster() {
  let cost = DMatrix::<f64>::from_element(1, 1, 0.42);
  let assign = constrained_argmax(&cost).expect("constrained_argmax");
  assert_eq!(assign, vec![0]);
}

/// Single speaker, multiple clusters: speaker must be assigned to the
/// cluster with maximum cost.
#[test]
fn single_speaker_multiple_clusters_picks_max() {
  let cost = DMatrix::<f64>::from_row_slice(1, 4, &[0.1, 0.5, 0.9, 0.3]);
  let assign = constrained_argmax(&cost).expect("constrained_argmax");
  assert_eq!(assign, vec![2]);
}

/// Single cluster, multiple speakers: only the speaker with maximum
/// cost gets matched; the rest are UNMATCHED.
#[test]
fn single_cluster_multiple_speakers_matches_max_speaker() {
  let cost = DMatrix::<f64>::from_row_slice(3, 1, &[0.1, 0.9, 0.5]);
  let assign = constrained_argmax(&cost).expect("constrained_argmax");
  assert_eq!(assign, vec![UNMATCHED, 0, UNMATCHED]);
}

/// Determinism: two calls with the same cost matrix must produce
/// identical assignments. kuhn_munkres has no RNG, but tie-breaking
/// determinism still warrants an explicit check.
#[test]
fn deterministic_on_repeated_calls() {
  let cost = DMatrix::<f64>::from_fn(5, 4, |i, j| ((i + 2 * j) as f64 * 0.13).cos());
  let a = constrained_argmax(&cost).expect("a");
  let b = constrained_argmax(&cost).expect("b");
  assert_eq!(a, b);
}
```

- [ ] **Step 2: Run the test module — verify all fail with `todo!`**

Run: `cargo test --lib hungarian 2>&1 | tail -30`
Expected: 14 tests, all fail (panicked with "Task 2" todo or compile error if module wiring incomplete).

- [ ] **Step 3: Commit**

```bash
git add src/hungarian/tests.rs
git commit -m "hungarian: failing tests for constrained_argmax contract"
```

---

## Task 2: Implement `constrained_argmax`

**Files:**
- Modify: `src/hungarian/algo.rs` (replace the `todo!()` stub)

- [ ] **Step 1: Replace algo.rs with the full implementation**

```rust
//! Constrained Hungarian assignment (per-chunk maximum-weight matching).
//!
//! Ports `pyannote.audio.pipelines.clustering.SpeakerEmbedding.constrained_argmax`
//! (`clustering.py:127-140` in pyannote.audio 4.0.4). The pyannote function
//! takes a `(num_chunks, num_speakers, num_clusters)` cost tensor and runs
//! `scipy.optimize.linear_sum_assignment(cost, maximize=True)` on each chunk.
//! This Rust port operates per-chunk; the caller iterates chunks.

use crate::hungarian::error::Error;
use nalgebra::DMatrix;
use ordered_float::NotNan;
use pathfinding::prelude::{Matrix, kuhn_munkres};

/// Sentinel value for an unmatched speaker. Matches pyannote's
/// `-2 * np.ones((num_chunks, num_speakers), dtype=np.int8)`
/// initialization in `constrained_argmax`.
pub const UNMATCHED: i32 = -2;

/// Per-chunk constrained Hungarian assignment.
///
/// Given a `(num_speakers, num_clusters)` cost matrix (typically
/// `2 - cosine_distance(embedding, centroid)`), returns the maximum-weight
/// bipartite matching as `Vec<i32>` of length `num_speakers`. Each entry is
/// the cluster index assigned to that speaker, or [`UNMATCHED`] (`-2`) if
/// the speaker had no cluster left (only possible when
/// `num_speakers > num_clusters`).
///
/// # Errors
///
/// - [`Error::Shape`] if either dimension is zero.
/// - [`Error::NonFinite`] if any entry is NaN/`±inf`. (Pyannote pre-replaces
///   NaN with `np.nanmin(soft_clusters)`; this Rust port rejects fail-fast at
///   the boundary instead — production embeddings are always finite, so a
///   non-finite cost indicates upstream corruption that should not silently
///   proceed.)
///
/// # Algorithm
///
/// `pathfinding::kuhn_munkres` requires `rows <= columns`. When
/// `num_speakers > num_clusters` we transpose the cost matrix to
/// `(num_clusters, num_speakers)`, run kuhn_munkres, and invert the
/// resulting assignment.
pub fn constrained_argmax(soft_clusters: &DMatrix<f64>) -> Result<Vec<i32>, Error> {
  let (num_speakers, num_clusters) = soft_clusters.shape();
  if num_speakers == 0 {
    return Err(Error::Shape("num_speakers must be at least 1"));
  }
  if num_clusters == 0 {
    return Err(Error::Shape("num_clusters must be at least 1"));
  }
  for s in 0..num_speakers {
    for k in 0..num_clusters {
      if !soft_clusters[(s, k)].is_finite() {
        return Err(Error::NonFinite("soft_clusters"));
      }
    }
  }

  let mut assignment = vec![UNMATCHED; num_speakers];

  // Build a NotNan<f64> Matrix in the orientation kuhn_munkres requires.
  // Safety: the loop above already rejected NaN, so NotNan::new_unchecked
  // would be sound — but the cost is one extra branch per cell, negligible
  // for the typical S * K = 6 cells per chunk, and the safe variant keeps
  // the local invariants simpler to reason about.
  if num_speakers <= num_clusters {
    // Direct path: rows = speakers, columns = clusters.
    let mut data = Vec::with_capacity(num_speakers * num_clusters);
    for s in 0..num_speakers {
      for k in 0..num_clusters {
        data.push(NotNan::new(soft_clusters[(s, k)]).expect("finite (checked above)"));
      }
    }
    let weights =
      Matrix::from_vec(num_speakers, num_clusters, data).expect("matrix dims match data length");
    let (_total, speaker_to_cluster) = kuhn_munkres(&weights);
    for (s, &k) in speaker_to_cluster.iter().enumerate() {
      assignment[s] = i32::try_from(k).expect("cluster idx fits in i32");
    }
  } else {
    // Transpose path: rows = clusters, columns = speakers.
    let mut data = Vec::with_capacity(num_clusters * num_speakers);
    for k in 0..num_clusters {
      for s in 0..num_speakers {
        data.push(NotNan::new(soft_clusters[(s, k)]).expect("finite (checked above)"));
      }
    }
    let weights =
      Matrix::from_vec(num_clusters, num_speakers, data).expect("matrix dims match data length");
    let (_total, cluster_to_speaker) = kuhn_munkres(&weights);
    for (k, &s) in cluster_to_speaker.iter().enumerate() {
      assignment[s] = i32::try_from(k).expect("cluster idx fits in i32");
    }
  }

  Ok(assignment)
}
```

- [ ] **Step 2: Run unit tests — must all pass**

Run: `cargo test --lib hungarian::tests 2>&1 | tail -25`
Expected: all 14 tests pass.

- [ ] **Step 3: Run clippy**

Run: `cargo clippy --lib -- -D warnings 2>&1 | tail -10`
Expected: no warnings.

- [ ] **Step 4: Commit**

```bash
git add src/hungarian/algo.rs
git commit -m "hungarian: constrained_argmax (per-chunk Kuhn-Munkres)"
```

---

## Task 3: Parity test against `clustering.npz`

**Files:**
- Create: `src/hungarian/parity_tests.rs`

- [ ] **Step 1: Write parity test loading captured fixture**

```rust
//! Parity test: `diarization::hungarian::constrained_argmax` against pyannote's
//! captured `hard_clusters` (Phase-0 fixture).
//!
//! Loads `tests/parity/fixtures/01_dialogue/clustering.npz` and asserts that
//! running `constrained_argmax` on each captured `soft_clusters[c]` chunk
//! reproduces the captured `hard_clusters[c]` exactly. **Hard-fails** on
//! missing fixtures (same convention as `src/plda/parity_tests.rs` and
//! `src/vbx/parity_tests.rs`).

use std::{fs::File, io::BufReader, path::PathBuf};

use nalgebra::DMatrix;
use npyz::npz::NpzArchive;

use crate::hungarian::constrained_argmax;

fn repo_root() -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn fixture(rel: &str) -> PathBuf {
  repo_root().join(rel)
}

fn require_fixtures() {
  let required = ["tests/parity/fixtures/01_dialogue/clustering.npz"];
  let missing: Vec<&str> = required
    .iter()
    .copied()
    .filter(|p| !repo_root().join(p).exists())
    .collect();
  assert!(
    missing.is_empty(),
    "Hungarian parity fixture missing: {missing:?}. \
     Ships with the crate via `cargo publish`; a missing fixture is a \
     packaging error, not an opt-out. Re-run \
     `tests/parity/python/capture_intermediates.py` to regenerate."
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
fn constrained_argmax_matches_pyannote_hard_clusters() {
  require_fixtures();

  let path = fixture("tests/parity/fixtures/01_dialogue/clustering.npz");
  let (soft_flat, soft_shape) = read_npz_array::<f64>(&path, "soft_clusters");
  let (hard_flat, hard_shape) = read_npz_array::<i8>(&path, "hard_clusters");

  assert_eq!(soft_shape.len(), 3, "soft_clusters must be 3D");
  let num_chunks = soft_shape[0] as usize;
  let num_speakers = soft_shape[1] as usize;
  let num_clusters = soft_shape[2] as usize;

  assert_eq!(hard_shape.len(), 2, "hard_clusters must be 2D");
  assert_eq!(hard_shape[0] as usize, num_chunks);
  assert_eq!(hard_shape[1] as usize, num_speakers);

  let chunk_stride = num_speakers * num_clusters;
  let mut mismatches: Vec<(usize, Vec<i32>, Vec<i32>)> = Vec::new();
  for c in 0..num_chunks {
    let chunk_slice = &soft_flat[c * chunk_stride..(c + 1) * chunk_stride];
    let cost = DMatrix::<f64>::from_row_slice(num_speakers, num_clusters, chunk_slice);

    let got = constrained_argmax(&cost).expect("constrained_argmax");
    let want: Vec<i32> = (0..num_speakers)
      .map(|s| hard_flat[c * num_speakers + s] as i32)
      .collect();
    if got != want {
      mismatches.push((c, got, want));
    }
  }

  if !mismatches.is_empty() {
    let preview: String = mismatches
      .iter()
      .take(5)
      .map(|(c, got, want)| format!("  chunk {c}: got {got:?}, want {want:?}"))
      .collect::<Vec<_>>()
      .join("\n");
    panic!(
      "constrained_argmax parity failed on {}/{} chunks:\n{preview}",
      mismatches.len(),
      num_chunks
    );
  }

  eprintln!(
    "[parity_hungarian] all {num_chunks} chunks match (shape {num_speakers}x{num_clusters})"
  );
}
```

- [ ] **Step 2: Run parity test**

Run: `cargo test --lib hungarian::parity_tests 2>&1 | tail -15`
Expected: 1 test passes; eprintln shows "all 218 chunks match (shape 3x2)".

- [ ] **Step 3: Run full vbx + hungarian to confirm no regressions**

Run: `cargo test --lib hungarian vbx 2>&1 | grep "test result"`
Expected: all green; vbx still 41 passing, hungarian 15 passing (14 unit + 1 parity).

- [ ] **Step 4: Run clippy**

Run: `cargo clippy --lib --all-targets -- -D warnings 2>&1 | tail -10`
Expected: no warnings.

- [ ] **Step 5: Commit**

```bash
git add src/hungarian/parity_tests.rs
git commit -m "hungarian: parity test vs captured pyannote hard_clusters"
```

---

## Task 4: Push + open PR + Codex iteration

**Files:** none (git/CI only)

- [ ] **Step 1: Push the branch**

```bash
git push -u origin feat/phase3-hungarian
```

- [ ] **Step 2: Open PR against `0.1.0`**

Using `gh pr create --base 0.1.0`.

PR title: `Phase 3: Constrained Hungarian assignment (pyannote parity)`

PR body:

```markdown
## Summary
- Ports pyannote's `constrained_argmax` (clustering.py:127-140) to Rust.
- Uses `pathfinding::kuhn_munkres` + `ordered_float::NotNan<f64>` (no scaling/quantization).
- Handles `num_speakers > num_clusters` via transpose path (matches scipy's behavior on non-square cost matrices).
- Crate-private (`mod hungarian;` in lib.rs); integration with `Diarizer` lands in Phase 5.

## Test plan
- [x] 14 unit tests for shape, NaN/Inf rejection, square/tall/wide shapes, distinct-cluster invariant, single-speaker / single-cluster degenerate cases, determinism.
- [x] Parity test against captured `clustering.npz` (218 chunks of shape 3×2): exact int8 equality.
- [x] `cargo clippy --lib --all-targets -- -D warnings` clean.
```

- [ ] **Step 3: Run `/codex:adversarial-review` and iterate**

Same workflow as Phase 2: `/codex:adversarial-review --base 0.1.0`. Each iteration: run, review the findings, decide whether to fix or push back, commit fixes, re-run. Stop when Codex returns `looks-good` or when the remaining findings are deferred to a future phase.

---

## Self-Review

**Spec coverage:**
- ✅ NaN replacement → boundary rejection (deliberate divergence, documented).
- ✅ `linear_sum_assignment(maximize=True)` → `kuhn_munkres` (always maximizing in pathfinding).
- ✅ `(num_chunks, num_speakers, num_clusters)` tensor → per-chunk loop (matches pyannote's loop over `enumerate(soft_clusters)`).
- ✅ `-2` sentinel → `UNMATCHED` const.
- ✅ `int8` output → `Vec<i32>` (i32 is more ergonomic in Rust; cast back to i8 only at the parity boundary).
- ✅ S > K case via transpose.

**Type consistency:** `UNMATCHED: i32`, `Vec<i32>` return, parity test casts captured `i8` to `i32` for comparison. All consistent.

**Placeholder scan:** All steps contain concrete code or commands. No "TBD"/"add tests"/"similar to Task N" — every test body is fully written out.

## Execution Handoff

Plan saved to `docs/superpowers/plans/2026-04-29-dia-phase3-constrained-hungarian.md`. Next: dispatch via `superpowers:subagent-driven-development`.
