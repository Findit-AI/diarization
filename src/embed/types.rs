//! Public output types for `dia::embed`. All types are `Send + Sync`.

use crate::embed::options::{EMBEDDING_DIM, NORM_EPSILON};

/// A 256-d L2-normalized speaker embedding.
///
/// **Invariant:** `||embedding.as_array()||₂ > NORM_EPSILON`. The crate
/// guarantees this — the only public constructor (`normalize_from`)
/// returns `None` for degenerate inputs. Internal downstream code
/// (e.g., `Clusterer::submit`) can rely on this for similarity
/// computations being well-defined.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Embedding(pub(crate) [f32; EMBEDDING_DIM]);

impl Embedding {
  /// Borrow the raw L2-normalized 256-d vector.
  pub const fn as_array(&self) -> &[f32; EMBEDDING_DIM] {
    &self.0
  }

  /// Borrow as a slice.
  pub fn as_slice(&self) -> &[f32] {
    &self.0
  }

  /// Cosine similarity. Both inputs are L2-normalized (per the
  /// `Embedding` invariant), so this reduces to a dot product.
  /// Returns a value in `[-1.0, 1.0]`.
  pub fn similarity(&self, other: &Embedding) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..EMBEDDING_DIM {
      acc += self.0[i] * other.0[i];
    }
    acc
  }

  /// L2-normalize a raw 256-d inference output and wrap it.
  /// Returns `None` if `||raw||₂ < NORM_EPSILON` (degenerate input).
  /// Use after `EmbedModel::embed_features_batch` + custom aggregation.
  pub fn normalize_from(raw: [f32; EMBEDDING_DIM]) -> Option<Self> {
    // Compute ||raw||₂ in f64 for precision, then divide each
    // component in f32. Matches Python's typical behavior where
    // the L2 norm is computed in float32.
    let sq: f64 = raw.iter().map(|&x| (x as f64) * (x as f64)).sum();
    let n = sq.sqrt() as f32;
    if n < NORM_EPSILON {
      return None;
    }
    let mut out = [0.0f32; EMBEDDING_DIM];
    for (o, &r) in out.iter_mut().zip(raw.iter()) {
      *o = r / n;
    }
    Some(Self(out))
  }
}

/// Free-function form of [`Embedding::similarity`] for callers who
/// prefer it. Both styles are public; pick whichever reads more
/// naturally at the call site. **Bit-exactly equivalent** to the
/// method (same component-order dot product, no FMA rearrangement).
pub fn cosine_similarity(a: &Embedding, b: &Embedding) -> f32 {
  a.similarity(b)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn normalize_from_zero_returns_none() {
    assert!(Embedding::normalize_from([0.0; EMBEDDING_DIM]).is_none());
  }

  #[test]
  fn normalize_from_below_epsilon_returns_none() {
    let mut tiny = [0.0; EMBEDDING_DIM];
    tiny[0] = 1e-13; // < NORM_EPSILON
    assert!(Embedding::normalize_from(tiny).is_none());
  }

  #[test]
  fn normalize_from_unit_vector_round_trips() {
    let mut v = [0.0; EMBEDDING_DIM];
    v[0] = 1.0;
    let e = Embedding::normalize_from(v).unwrap();
    let n2: f32 = e.as_array().iter().map(|x| x * x).sum();
    assert!((n2 - 1.0).abs() < 1e-6, "||result|| ≈ 1, got n2 = {n2}");
    assert!((e.as_array()[0] - 1.0).abs() < 1e-6);
  }

  #[test]
  fn normalize_from_arbitrary_vector_norms_to_one() {
    let mut raw = [0.0; EMBEDDING_DIM];
    for (i, v) in raw.iter_mut().enumerate() {
      *v = (i as f32) * 0.01 + 0.1;
    }
    let e = Embedding::normalize_from(raw).unwrap();
    let n2: f32 = e.as_array().iter().map(|x| x * x).sum();
    assert!((n2 - 1.0).abs() < 1e-5, "n2 = {n2}");
  }

  #[test]
  fn similarity_self_is_one() {
    let mut v = [0.0; EMBEDDING_DIM];
    v[0] = 1.0;
    let e = Embedding::normalize_from(v).unwrap();
    assert!((e.similarity(&e) - 1.0).abs() < 1e-6);
  }

  #[test]
  fn similarity_orthogonal_is_zero() {
    let mut a = [0.0; EMBEDDING_DIM];
    a[0] = 1.0;
    let mut b = [0.0; EMBEDDING_DIM];
    b[1] = 1.0;
    let ea = Embedding::normalize_from(a).unwrap();
    let eb = Embedding::normalize_from(b).unwrap();
    assert!(ea.similarity(&eb).abs() < 1e-6);
  }

  #[test]
  fn similarity_antipodal_is_negative_one() {
    let mut a = [0.0; EMBEDDING_DIM];
    a[0] = 1.0;
    let mut b = [0.0; EMBEDDING_DIM];
    b[0] = -1.0;
    let ea = Embedding::normalize_from(a).unwrap();
    let eb = Embedding::normalize_from(b).unwrap();
    assert!((ea.similarity(&eb) + 1.0).abs() < 1e-6);
  }

  #[test]
  fn similarity_symmetric() {
    let mut a = [0.0; EMBEDDING_DIM];
    a[0] = 0.6;
    a[1] = 0.8;
    let mut b = [0.0; EMBEDDING_DIM];
    b[0] = 0.8;
    b[1] = 0.6;
    let ea = Embedding::normalize_from(a).unwrap();
    let eb = Embedding::normalize_from(b).unwrap();
    assert!((ea.similarity(&eb) - eb.similarity(&ea)).abs() < 1e-7);
  }

  #[test]
  fn cosine_similarity_matches_method() {
    let mut a = [0.0; EMBEDDING_DIM];
    let mut b = [0.0; EMBEDDING_DIM];
    for (i, (av, bv)) in a.iter_mut().zip(b.iter_mut()).enumerate() {
      *av = (i as f32 * 0.01).sin();
      *bv = (i as f32 * 0.013).cos();
    }
    let ea = Embedding::normalize_from(a).unwrap();
    let eb = Embedding::normalize_from(b).unwrap();
    // Free fn must equal method bit-exactly (same dot product,
    // same component order — no fma rearrangement).
    assert_eq!(cosine_similarity(&ea, &eb), ea.similarity(&eb));
  }
}
