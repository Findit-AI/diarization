//! AHC initialization: L2-normalize → centroid linkage → fcluster + remap.

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
/// drive `dia::ahc::ahc_init` uniformly without the special case
/// leaking into them.
pub fn ahc_init(embeddings: &DMatrix<f64>, threshold: f64) -> Result<Vec<usize>, Error> {
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

  let normed = l2_normalize_rows(embeddings);
  let mut cond = pdist_euclidean(&normed);
  let dend = linkage(&mut cond, n, Method::Centroid);

  Ok(fcluster_distance_remap(dend.steps(), n, threshold))
}

/// Row-wise L2 normalization. Each row is divided by its L2 norm.
/// Caller has already rejected zero-norm rows.
fn l2_normalize_rows(m: &DMatrix<f64>) -> DMatrix<f64> {
  let mut out = m.clone();
  for r in 0..out.nrows() {
    let norm = out.row(r).iter().map(|v| v * v).sum::<f64>().sqrt();
    for c in 0..out.ncols() {
      out[(r, c)] /= norm;
    }
  }
  out
}

/// Condensed pairwise euclidean distance vector — `pdist`-style ordering:
/// `[d(0,1), d(0,2), ..., d(0,n-1), d(1,2), ..., d(n-2,n-1)]` of length
/// `n * (n - 1) / 2`. This is the format `kodama::linkage` expects.
fn pdist_euclidean(rows: &DMatrix<f64>) -> Vec<f64> {
  let (n, d) = rows.shape();
  let mut out = Vec::with_capacity(n * (n - 1) / 2);
  for i in 0..n {
    for j in (i + 1)..n {
      let mut sq = 0.0;
      for c in 0..d {
        let diff = rows[(i, c)] - rows[(j, c)];
        sq += diff * diff;
      }
      out.push(sq.sqrt());
    }
  }
  out
}

/// `fcluster(criterion="distance", t=threshold)` followed by
/// `np.unique(return_inverse=True)`. Walks the dendrogram steps and
/// unions only those whose dissimilarity is `≤ threshold`, then assigns
/// encounter-order labels `0..k` to leaves.
///
/// kodama returns `n - 1` steps, each merging two cluster ids. Leaf ids
/// are `0..n`; compound ids are `n + step_idx`. We track each cluster
/// id's leaf representative (any leaf in its set) so subsequent steps
/// referencing a compound id can union the right leaves.
fn fcluster_distance_remap(steps: &[Step<f64>], n: usize, threshold: f64) -> Vec<usize> {
  let mut uf = UnionFind::new(n);
  // cluster_leaf_rep[id] = some leaf id in cluster `id`. Pre-fill the n
  // leaves; compound clusters get their rep set as steps are processed.
  let mut cluster_leaf_rep: Vec<usize> = (0..n).collect();
  cluster_leaf_rep.resize(n + steps.len(), usize::MAX);

  for (step_idx, step) in steps.iter().enumerate() {
    let rep1 = cluster_leaf_rep[step.cluster1];
    let rep2 = cluster_leaf_rep[step.cluster2];
    cluster_leaf_rep[n + step_idx] = rep1; // any leaf in the merged set
    if step.dissimilarity <= threshold {
      uf.union(rep1, rep2);
    }
  }

  // Assign encounter-order labels to roots.
  let mut label_of_root: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
  let mut next_label = 0usize;
  let mut labels = vec![0usize; n];
  for (i, slot) in labels.iter_mut().enumerate() {
    let root = uf.find(i);
    *slot = *label_of_root.entry(root).or_insert_with(|| {
      let l = next_label;
      next_label += 1;
      l
    });
  }
  labels
}

/// Minimal union-find with path compression. Sized at construction
/// (one node per leaf); compound dendrogram nodes never carry their own
/// union-find slot — the algorithm unions the leaf reps directly.
struct UnionFind {
  parent: Vec<usize>,
  rank: Vec<u32>,
}

impl UnionFind {
  fn new(n: usize) -> Self {
    Self {
      parent: (0..n).collect(),
      rank: vec![0; n],
    }
  }

  fn find(&mut self, x: usize) -> usize {
    let mut root = x;
    while self.parent[root] != root {
      root = self.parent[root];
    }
    // Path compression.
    let mut cur = x;
    while self.parent[cur] != root {
      let next = self.parent[cur];
      self.parent[cur] = root;
      cur = next;
    }
    root
  }

  fn union(&mut self, a: usize, b: usize) {
    let ra = self.find(a);
    let rb = self.find(b);
    if ra == rb {
      return;
    }
    // Union by rank.
    if self.rank[ra] < self.rank[rb] {
      self.parent[ra] = rb;
    } else if self.rank[ra] > self.rank[rb] {
      self.parent[rb] = ra;
    } else {
      self.parent[rb] = ra;
      self.rank[ra] += 1;
    }
  }
}
