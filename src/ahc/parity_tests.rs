//! Parity test for `dia::ahc::ahc_init` against pyannote's captured
//! `ahc_init_labels.npy` (Phase-0 fixture).
//!
//! Loads:
//! - `tests/parity/fixtures/01_dialogue/raw_embeddings.npz` (raw 256-dim
//!   embeddings, the input pyannote feeds to `linkage`).
//! - `tests/parity/fixtures/01_dialogue/plda_embeddings.npz`
//!   (`train_chunk_idx` / `train_speaker_idx` for the active-frame
//!   filter pyannote applies before AHC).
//! - `tests/parity/fixtures/01_dialogue/ahc_state.npz` (the `threshold`
//!   pyannote was configured with at capture time — Phase-4 Task 0).
//! - `tests/parity/fixtures/01_dialogue/ahc_init_labels.npy` (the
//!   ground-truth labels after `np.unique(return_inverse=True)`).
//!
//! Asserts exact `Vec<usize>` equality. **Hard-fails** on missing
//! fixtures.

use std::{fs::File, io::BufReader, path::PathBuf};

use nalgebra::DMatrix;
use npyz::npz::NpzArchive;

use crate::ahc::ahc_init;

fn repo_root() -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn fixture(rel: &str) -> PathBuf {
  repo_root().join(rel)
}

fn require_fixtures() {
  let required = [
    "tests/parity/fixtures/01_dialogue/raw_embeddings.npz",
    "tests/parity/fixtures/01_dialogue/plda_embeddings.npz",
    "tests/parity/fixtures/01_dialogue/ahc_state.npz",
    "tests/parity/fixtures/01_dialogue/ahc_init_labels.npy",
  ];
  let missing: Vec<&str> = required
    .iter()
    .copied()
    .filter(|p| !repo_root().join(p).exists())
    .collect();
  assert!(
    missing.is_empty(),
    "AHC parity fixtures missing: {missing:?}. \
     Re-run `tests/parity/python/capture_intermediates.py` to regenerate."
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

fn read_npy_array<T>(path: &PathBuf) -> (Vec<T>, Vec<u64>)
where
  T: npyz::Deserialize,
{
  let f = File::open(path).expect("open npy");
  let npy = npyz::NpyFile::new(BufReader::new(f)).expect("read npy");
  let shape: Vec<u64> = npy.shape().to_vec();
  let data: Vec<T> = npy.into_vec().expect("decode npy");
  (data, shape)
}

#[test]
fn ahc_init_matches_pyannote_ahc_init_labels() {
  require_fixtures();

  // Load raw embeddings (3D: chunks × speakers × dim).
  let raw_path = fixture("tests/parity/fixtures/01_dialogue/raw_embeddings.npz");
  let (raw_flat, raw_shape) = read_npz_array::<f32>(&raw_path, "embeddings");
  assert_eq!(raw_shape.len(), 3, "raw embeddings must be 3D");
  let num_chunks = raw_shape[0] as usize;
  let num_speakers = raw_shape[1] as usize;
  let dim = raw_shape[2] as usize;

  // Load active-frame indices captured alongside the PLDA outputs.
  // `train_chunk_idx[i]` and `train_speaker_idx[i]` together pick a row
  // out of the (chunks × speakers, dim) flattened raw embedding tensor —
  // matching pyannote's `filter_embeddings` projection.
  let plda_path = fixture("tests/parity/fixtures/01_dialogue/plda_embeddings.npz");
  let (chunk_idx, _) = read_npz_array::<i64>(&plda_path, "train_chunk_idx");
  let (speaker_idx, _) = read_npz_array::<i64>(&plda_path, "train_speaker_idx");
  assert_eq!(
    chunk_idx.len(),
    speaker_idx.len(),
    "train_chunk_idx and train_speaker_idx must align"
  );
  let num_train = chunk_idx.len();

  // Project the active embeddings into a (num_train, dim) matrix.
  let mut train = DMatrix::<f64>::zeros(num_train, dim);
  for i in 0..num_train {
    let c = chunk_idx[i] as usize;
    let s = speaker_idx[i] as usize;
    assert!(
      c < num_chunks && s < num_speakers,
      "active idx out of range"
    );
    let base = (c * num_speakers + s) * dim;
    for d in 0..dim {
      train[(i, d)] = raw_flat[base + d] as f64;
    }
  }

  // Load threshold + ground-truth labels.
  let state_path = fixture("tests/parity/fixtures/01_dialogue/ahc_state.npz");
  let (threshold_data, _) = read_npz_array::<f64>(&state_path, "threshold");
  let threshold = threshold_data[0];

  let labels_path = fixture("tests/parity/fixtures/01_dialogue/ahc_init_labels.npy");
  let (want_labels_i64, want_shape) = read_npy_array::<i64>(&labels_path);
  assert_eq!(want_shape.len(), 1);
  assert_eq!(want_shape[0] as usize, num_train);
  let want: Vec<usize> = want_labels_i64.iter().map(|&v| v as usize).collect();

  // Run the port.
  let got = ahc_init(&train, threshold).expect("ahc_init");

  // Compare *partitions*, not exact label assignments. Scipy's fcluster
  // assigns labels via dendrogram tree traversal, which differs from
  // kodama's order; pyannote's `np.unique(fcluster - 1, return_inverse=
  // True)` is a no-op for contiguous 0..k-1 labels and does *not*
  // canonicalize the order. Partition equality (which two leaves end
  // up in the same cluster) is the correctness invariant that matters
  // for downstream VBx + Diarizer.
  //
  // Note: Phase 5 will need to either match scipy's exact label order
  // (for bit-exact qinit parity) or accept column-permuted qinit. For
  // Phase 4 standalone, canonicalizing both via encounter-order makes
  // the comparison rigorous on the partition without locking in
  // scipy's traversal order.
  let got_canon = canonicalize_to_encounter_order(&got);
  let want_canon = canonicalize_to_encounter_order(&want);
  assert_eq!(
    got_canon,
    want_canon,
    "ahc_init partition diverged from pyannote (first 20 got vs want canonicalized: {:?} vs {:?}; threshold={threshold})",
    &got_canon[..20.min(got_canon.len())],
    &want_canon[..20.min(want_canon.len())],
  );

  let unique_count = want_canon.iter().copied().max().unwrap() + 1;
  eprintln!(
    "[parity_ahc] {num_train} labels: partition matches pyannote (k={unique_count}, threshold={threshold})"
  );
}

/// Remap labels to encounter-order: the first label seen becomes 0,
/// the second new label becomes 1, etc. After this transform, two
/// different label arrays representing the same partition compare equal.
fn canonicalize_to_encounter_order(labels: &[usize]) -> Vec<usize> {
  use std::collections::HashMap;
  let mut next = 0usize;
  let mut map: HashMap<usize, usize> = HashMap::new();
  labels
    .iter()
    .map(|&l| {
      *map.entry(l).or_insert_with(|| {
        let v = next;
        next += 1;
        v
      })
    })
    .collect()
}
