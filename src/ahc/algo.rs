//! AHC initialization: L2-normalize → centroid linkage → fcluster + remap.
//!
//! ## Determinism contract w.r.t. `pdist_euclidean`
//!
//! AHC's pairwise-distance computation runs through
//! [`crate::ops::pdist_euclidean`] with `use_simd = true` in
//! production. On aarch64 (the deployment target) scalar and the
//! NEON kernel produce **bit-identical f64 outputs** — Codex
//! adversarial review (multiple rounds 2026-04-30) flagged the hard
//! `<= threshold` cut as a discrete-decision risk under the previous
//! "1e-12 relative tolerance" SIMD contract. The fix replaced
//! scalar's serial `acc += a*b` with `f64::mul_add` and a 4-acc
//! reduction tree matching NEON's lane structure exactly; see
//! [`crate::ops::scalar::dot`] / [`crate::ops::scalar::pdist_euclidean`]
//! module docs.
//!
//! Cross-architecture (NEON 2-lane vs AVX2 4-lane vs AVX-512 8-lane)
//! the lane-width reduction trees still differ by O(1e-15) relative
//! on well-conditioned inputs — the per-element FMA is bit-identical
//! everywhere, but the horizontal reduce structure is lane-bound.
//! [`crate::ops::differential_tests`] enforces the bit-identical
//! contract on aarch64 and a 1e-14 relative bound elsewhere.
//!
//! [`crate::ahc::tests::simd_partition_stability`] additionally
//! verifies that for 50 random seeds (N=20, D=128) and constructed
//! threshold-adjacent inputs, the AHC partition is identical between
//! scalar and SIMD even at threshold offsets of ±1e-13.

use crate::ahc::error::Error;
use kodama::{Method, Step, linkage};
use nalgebra::DMatrix;

/// Run pyannote's AHC initialization.
///
/// Mirrors `pyannote/audio/pipelines/clustering.py:597-604`:
///
/// 1. L2-normalize each row of `embeddings` (shape `(N, D)`).
/// 2. Compute pairwise euclidean distances (the condensed `pdist`-style
///    upper-triangular vector scipy expects).
/// 3. Centroid-method hierarchical linkage via `kodama` (matches scipy's
///    `linkage(..., method="centroid")` Lance-Williams formula).
/// 4. `fcluster` with `criterion="distance"` and the given `threshold`:
///    union pairs whose merge dissimilarity is `≤ threshold`.
/// 5. Remap the resulting partition to encounter-order contiguous labels
///    `0..k`, equivalent to `np.unique(_, return_inverse=True)[1]`.
///
/// # Errors
///
/// - [`Error::Shape`] if `embeddings` is empty, has zero-length rows, has
///   any zero-L2-norm row, or `threshold` is non-positive / non-finite.
/// - [`Error::NonFinite`] if `embeddings` contains a NaN/`±inf`.
///
/// # Single-row degenerate case
///
/// Pyannote short-circuits AHC entirely when `train_embeddings.shape[0]
/// < 2` (`clustering.py:588-594`). This module-level boundary allows
/// `N=1` and returns `vec![0]` (one cluster, one member) so callers can
/// drive `diarization::ahc::ahc_init` uniformly without the special case
/// leaking into them.
pub fn ahc_init(embeddings: &DMatrix<f64>, threshold: f64) -> Result<Vec<usize>, Error> {
  // Production: SIMD on. Scalar and NEON now produce bit-identical
  // pdist outputs on aarch64 (scalar uses `f64::mul_add` and the
  // same 4-acc reduction tree as NEON; verified by
  // `ops::differential_tests::pdist_euclidean_well_conditioned_match`).
  // See module docs for the precision contract.
  ahc_init_inner(embeddings, threshold, true)
}

/// Test-only entrypoint: identical to [`ahc_init`] but with the
/// `use_simd` flag exposed so the differential tests in
/// [`crate::ahc::tests`] can run scalar-vs-SIMD pdist on identical
/// inputs and compare the resulting partitions. Production code goes
/// through [`ahc_init`] which always passes `true`.
#[cfg(test)]
pub(crate) fn ahc_init_with_simd(
  embeddings: &DMatrix<f64>,
  threshold: f64,
  use_simd: bool,
) -> Result<Vec<usize>, Error> {
  ahc_init_inner(embeddings, threshold, use_simd)
}

fn ahc_init_inner(
  embeddings: &DMatrix<f64>,
  threshold: f64,
  use_simd: bool,
) -> Result<Vec<usize>, Error> {
  let (n, d) = embeddings.shape();
  if n == 0 {
    return Err(Error::Shape("embeddings must have at least one row"));
  }
  if d == 0 {
    return Err(Error::Shape("embeddings must have at least one column"));
  }
  if !threshold.is_finite() || threshold <= 0.0 {
    return Err(Error::Shape("threshold must be a positive finite scalar"));
  }
  // Validate finite + nonzero L2 norm per row.
  for r in 0..n {
    let mut sq = 0.0;
    for c in 0..d {
      let v = embeddings[(r, c)];
      if !v.is_finite() {
        return Err(Error::NonFinite("embeddings"));
      }
      sq += v * v;
    }
    if sq == 0.0 {
      return Err(Error::Shape(
        "embeddings row has zero L2 norm; cannot normalize",
      ));
    }
  }

  if n == 1 {
    return Ok(vec![0]);
  }

  let normed_row_major = l2_normalize_to_row_major(embeddings);
  let mut cond = crate::ops::pdist_euclidean(&normed_row_major, n, d, use_simd);
  let dend = linkage(&mut cond, n, Method::Centroid);

  Ok(fcluster_distance_remap(dend.steps(), n, threshold))
}

/// Pack the row-wise L2-normalized embeddings into a row-major flat
/// buffer in a single pass. nalgebra's `DMatrix` is column-major, and
/// [`crate::ops::pdist_euclidean`] (and its eventual SIMD backend)
/// wants a contiguous row-major slice — so we fuse the normalize +
/// transpose into one allocation.
///
/// Caller has already rejected zero-norm rows.
fn l2_normalize_to_row_major(m: &DMatrix<f64>) -> Vec<f64> {
  let (n, d) = m.shape();
  let mut out = Vec::with_capacity(n * d);
  for r in 0..n {
    let mut sq = 0.0;
    for c in 0..d {
      let v = m[(r, c)];
      sq += v * v;
    }
    let inv_norm = sq.sqrt().recip();
    for c in 0..d {
      out.push(m[(r, c)] * inv_norm);
    }
  }
  out
}

/// `fcluster(criterion="distance", t=threshold)` followed by
/// `np.unique(return_inverse=True)`. Mirrors `scipy._hierarchy.cluster_dist`:
/// (1) precompute the *maximum* merge dissimilarity in each subtree,
/// (2) walk top-down, cutting wherever that max exceeds the threshold.
///
/// Why max-per-subtree rather than the root's own dissimilarity:
/// centroid linkage can produce *inversions* (a parent merge has lower
/// dissimilarity than one of its children — Codex review HIGH round 1
/// of Phase 4). A walk that only checks the root's `step.dissimilarity`
/// would merge an entire subtree based on a low-dist parent even when
/// an internal child merge is above the threshold. Scipy's fcluster
/// (`scipy/cluster/_hierarchy.pyx::cluster_dist`) propagates the max
/// dissimilarity up the tree first, then uses that as the cut criterion
/// — i.e. a flat cluster contains pairs whose cophenetic distance is
/// `≤ threshold`, which is the documented contract.
///
/// # Label assignment: leaf-scan encounter order, not scipy's traversal
///
/// The second pass canonicalizes labels via *leaf-scan encounter order*
/// (the first cluster seen while scanning leaves `0..n` becomes label 0).
/// This is the np.unique-on-contiguous-labels formula but assumes scipy
/// already produced canonical scan-order labels — which **scipy does
/// not do**. Scipy's `fcluster` numbers clusters by tree-traversal
/// order; the captured `ahc_init_labels.npy` starts with label `4` for
/// row 0, not `0`. (Codex review MEDIUM round 2 of Phase 4.)
///
/// The captured Phase-4 parity test compares partitions, not exact
/// label assignments — partition equivalence is sufficient for
/// downstream clustering correctness (the labels are arbitrary
/// integers naming the buckets; DER is invariant to relabeling).
///
/// **TODO** (Phase 5 integration): if a Phase-5 end-to-end parity test
/// runs `ahc_init → build qinit → vbx_iterate → q_final` and compares
/// element-wise against captured `q_final`, the `qinit` column ordering
/// will not match (since our labels are a permutation of scipy's). At
/// that point, choose one of:
/// 1. Implement scipy's exact tree-traversal label order here (drop
///    this canonicalization pass; align DFS push order with scipy's
///    `_hierarchy.pyx::cluster_dist`).
/// 2. Have Phase 5 compare `q_final` modulo column permutation
///    (mathematically equivalent — the permutation is recoverable
///    from `(our_labels, scipy_labels)` matching).
/// 3. Have `ahc_init` return `(labels, permutation_to_scipy)` so the
///    caller can build the column-permuted qinit explicitly.
///
/// Either way, the Phase-4 contract is "produce a valid scipy-equivalent
/// partition", and the existing parity test enforces that.
fn fcluster_distance_remap(steps: &[Step<f64>], n: usize, threshold: f64) -> Vec<usize> {
  // Single leaf — no merges; one cluster.
  if n == 1 {
    return vec![0];
  }

  // Precompute the maximum dissimilarity in each subtree. Leaves have 0
  // (they contain no merges); compound id `n + i` has max of its own
  // merge plus the max of its two children.
  let total_nodes = n + steps.len();
  let mut subtree_max = vec![0.0_f64; total_nodes];
  for (i, step) in steps.iter().enumerate() {
    let m1 = subtree_max[step.cluster1];
    let m2 = subtree_max[step.cluster2];
    subtree_max[n + i] = step.dissimilarity.max(m1).max(m2);
  }

  // First pass: top-down DFS labels leaves by partition class.
  let mut raw = vec![usize::MAX; n];
  let mut next_dfs_label = 0usize;
  let root = total_nodes - 1;
  let mut stack: Vec<usize> = vec![root];
  while let Some(node) = stack.pop() {
    if node < n {
      // Bare leaf surfaced via a split — its own cluster.
      raw[node] = next_dfs_label;
      next_dfs_label += 1;
    } else if subtree_max[node] <= threshold {
      // Whole subtree fits within the threshold — one cluster.
      let l = next_dfs_label;
      next_dfs_label += 1;
      paint_leaves(node, n, steps, l, &mut raw);
    } else {
      // Subtree contains a merge above threshold; split into children.
      let step = &steps[node - n];
      stack.push(step.cluster2);
      stack.push(step.cluster1);
    }
  }

  // Second pass: scan leaves 0..n and assign encounter-order labels.
  let mut canonical = vec![0usize; n];
  let mut next_label = 0usize;
  let mut label_of_class: std::collections::HashMap<usize, usize> =
    std::collections::HashMap::new();
  for (i, slot) in canonical.iter_mut().enumerate() {
    *slot = *label_of_class.entry(raw[i]).or_insert_with(|| {
      let l = next_label;
      next_label += 1;
      l
    });
  }
  canonical
}

/// Recursively assign `label` to every leaf reachable from `node`.
/// Uses iterative traversal to avoid stack-depth concerns on deep
/// dendrograms.
fn paint_leaves(node: usize, n: usize, steps: &[Step<f64>], label: usize, labels: &mut [usize]) {
  let mut stack = vec![node];
  while let Some(cur) = stack.pop() {
    if cur < n {
      labels[cur] = label;
    } else {
      let step = &steps[cur - n];
      stack.push(step.cluster1);
      stack.push(step.cluster2);
    }
  }
}
