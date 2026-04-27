//! Cross-component cluster tests (online + offline) per spec §9.
//!
//! Verifies that on the same well-separated synthetic input the online
//! `Clusterer` and the offline agglomerative path produce semantically
//! equivalent groupings. Cheap sanity check that the two backends agree
//! on a problem they should both find easy.

use super::*;
use crate::embed::{EMBEDDING_DIM, Embedding};

/// Construct a unit-direction embedding `e_i` with a small leak into
/// dimension `(i+1) % EMBEDDING_DIM`. Norm-1 by `Embedding::normalize_from`.
fn perturbed_unit(i: usize, scale: f32) -> Embedding {
  let mut v = [0.0f32; EMBEDDING_DIM];
  v[i] = 1.0;
  v[(i + 1) % EMBEDDING_DIM] = scale;
  Embedding::normalize_from(v).unwrap()
}

#[test]
fn online_separates_two_well_clustered_groups() {
  let mut c = Clusterer::new(ClusterOptions::default());
  // Group A: 5 near-unit(0).
  for s in [0.0, 0.05, -0.05, 0.1, -0.1] {
    c.submit(&perturbed_unit(0, s)).unwrap();
  }
  // Group B: 5 near-unit(10).
  for s in [0.0, 0.05, -0.05, 0.1, -0.1] {
    c.submit(&perturbed_unit(10, s)).unwrap();
  }
  assert_eq!(
    c.num_speakers(),
    2,
    "online clusterer should open exactly 2 speakers on these well-separated groups"
  );
}

#[test]
fn agglomerative_average_matches_two_groups() {
  let mut e = Vec::new();
  for s in [0.0, 0.05, -0.05] {
    e.push(perturbed_unit(0, s));
  }
  for s in [0.0, 0.05, -0.05] {
    e.push(perturbed_unit(10, s));
  }
  let labels = cluster_offline(
    &e,
    &OfflineClusterOptions::default().with_method(OfflineMethod::Agglomerative {
      linkage: Linkage::Average,
    }),
  )
  .unwrap();
  // First three indices share a label, last three share another, and the
  // two groups have different labels.
  assert_eq!(labels[0], labels[1]);
  assert_eq!(labels[1], labels[2]);
  assert_eq!(labels[3], labels[4]);
  assert_eq!(labels[4], labels[5]);
  assert_ne!(
    labels[0], labels[3],
    "two well-separated groups must end up in different clusters"
  );
}
