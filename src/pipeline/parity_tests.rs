//! End-to-end parity test: `dia::pipeline::assign_embeddings` against
//! pyannote's captured `clustering.npz['hard_clusters']` (Phase-0 fixture).
//!
//! Inputs (all from the captured fixtures):
//! - `raw_embeddings.npz['embeddings']` — 3D (chunks × speakers × dim) raw
//!   x-vectors (f32 → f64).
//! - `segmentations.npz['segmentations']` — 3D (chunks × frames × speakers)
//!   per-frame speaker probabilities.
//! - `plda_embeddings.npz['post_plda', 'phi', 'train_chunk_idx',
//!   'train_speaker_idx']` — pre-PLDA outputs that `cluster_vbx` would
//!   compute internally; we accept them pre-computed because Phase 1
//!   already validated PLDA parity on these exact arrays.
//! - `ahc_state.npz['threshold']` — AHC linkage cutoff (0.6).
//! - `vbx_state.npz['fa', 'fb', 'max_iters']` — VBx hyperparameters.
//!
//! Expected: `clustering.npz['hard_clusters']` (chunks × speakers, int8).
//! Comparison is **partition-equivalent** (canonicalized via
//! encounter-order on each chunk) — same trade-off documented in the
//! Phase 4 ahc parity test (scipy fcluster's traversal-order labels
//! permute the cluster ids; partition is the actual contract).

use std::{fs::File, io::BufReader, path::PathBuf};

use nalgebra::{DMatrix, DVector};
use npyz::npz::NpzArchive;

use crate::{
  hungarian::UNMATCHED,
  pipeline::{AssignEmbeddingsInput, assign_embeddings},
};

fn repo_root() -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn fixture(rel: &str) -> PathBuf {
  repo_root().join(rel)
}

fn require_fixtures() {
  let required = [
    "tests/parity/fixtures/01_dialogue/raw_embeddings.npz",
    "tests/parity/fixtures/01_dialogue/segmentations.npz",
    "tests/parity/fixtures/01_dialogue/plda_embeddings.npz",
    "tests/parity/fixtures/01_dialogue/ahc_state.npz",
    "tests/parity/fixtures/01_dialogue/vbx_state.npz",
    "tests/parity/fixtures/01_dialogue/clustering.npz",
  ];
  let missing: Vec<&str> = required
    .iter()
    .copied()
    .filter(|p| !repo_root().join(p).exists())
    .collect();
  assert!(
    missing.is_empty(),
    "pipeline parity fixtures missing: {missing:?}. \
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

#[test]
fn assign_embeddings_matches_pyannote_hard_clusters() {
  require_fixtures();

  // Raw embeddings (chunks, speakers, embed_dim).
  let raw_path = fixture("tests/parity/fixtures/01_dialogue/raw_embeddings.npz");
  let (raw_flat, raw_shape) = read_npz_array::<f32>(&raw_path, "embeddings");
  assert_eq!(raw_shape.len(), 3);
  let num_chunks = raw_shape[0] as usize;
  let num_speakers = raw_shape[1] as usize;
  let embed_dim = raw_shape[2] as usize;
  let mut embeddings = DMatrix::<f64>::zeros(num_chunks * num_speakers, embed_dim);
  for c in 0..num_chunks {
    for s in 0..num_speakers {
      let row = c * num_speakers + s;
      let base = (c * num_speakers + s) * embed_dim;
      for d in 0..embed_dim {
        embeddings[(row, d)] = raw_flat[base + d] as f64;
      }
    }
  }

  // Segmentations (chunks, frames, speakers).
  let seg_path = fixture("tests/parity/fixtures/01_dialogue/segmentations.npz");
  let (seg_flat_f32, seg_shape) = read_npz_array::<f32>(&seg_path, "segmentations");
  assert_eq!(seg_shape.len(), 3);
  let num_frames = seg_shape[1] as usize;
  assert_eq!(seg_shape[0] as usize, num_chunks);
  assert_eq!(seg_shape[2] as usize, num_speakers);
  let segmentations: Vec<f64> = seg_flat_f32.iter().map(|&v| v as f64).collect();

  // post_plda + phi + train_*idx (pre-filtered, pre-projected).
  let plda_path = fixture("tests/parity/fixtures/01_dialogue/plda_embeddings.npz");
  let (post_plda_flat, post_plda_shape) = read_npz_array::<f64>(&plda_path, "post_plda");
  assert_eq!(post_plda_shape.len(), 2);
  let num_train = post_plda_shape[0] as usize;
  let plda_dim = post_plda_shape[1] as usize;
  let post_plda = DMatrix::<f64>::from_row_slice(num_train, plda_dim, &post_plda_flat);

  let (phi_flat, phi_shape) = read_npz_array::<f64>(&plda_path, "phi");
  assert_eq!(phi_shape, vec![plda_dim as u64]);
  let phi = DVector::<f64>::from_vec(phi_flat);

  let (chunk_idx_i64, _) = read_npz_array::<i64>(&plda_path, "train_chunk_idx");
  let (speaker_idx_i64, _) = read_npz_array::<i64>(&plda_path, "train_speaker_idx");
  assert_eq!(chunk_idx_i64.len(), num_train);
  assert_eq!(speaker_idx_i64.len(), num_train);
  let train_chunk_idx: Vec<usize> = chunk_idx_i64.iter().map(|&v| v as usize).collect();
  let train_speaker_idx: Vec<usize> = speaker_idx_i64.iter().map(|&v| v as usize).collect();

  // Hyperparameters.
  let ahc_path = fixture("tests/parity/fixtures/01_dialogue/ahc_state.npz");
  let (threshold_data, _) = read_npz_array::<f64>(&ahc_path, "threshold");
  let threshold = threshold_data[0];

  let vbx_path = fixture("tests/parity/fixtures/01_dialogue/vbx_state.npz");
  let (fa_arr, _) = read_npz_array::<f64>(&vbx_path, "fa");
  let (fb_arr, _) = read_npz_array::<f64>(&vbx_path, "fb");
  let (max_iters_arr, _) = read_npz_array::<i64>(&vbx_path, "max_iters");
  let fa = fa_arr[0];
  let fb = fb_arr[0];
  let max_iters = max_iters_arr[0] as usize;

  // Run the port.
  let input = AssignEmbeddingsInput {
    embeddings: &embeddings,
    num_chunks,
    num_speakers,
    segmentations: &segmentations,
    num_frames,
    post_plda: &post_plda,
    phi: &phi,
    train_chunk_idx: &train_chunk_idx,
    train_speaker_idx: &train_speaker_idx,
    threshold,
    fa,
    fb,
    max_iters,
  };
  let got = assign_embeddings(&input).expect("assign_embeddings");

  // Captured ground truth.
  let cluster_path = fixture("tests/parity/fixtures/01_dialogue/clustering.npz");
  let (hard_flat_i8, hard_shape) = read_npz_array::<i8>(&cluster_path, "hard_clusters");
  assert_eq!(hard_shape, vec![num_chunks as u64, num_speakers as u64]);

  // Build the captured per-chunk vectors.
  let want: Vec<Vec<i32>> = (0..num_chunks)
    .map(|c| {
      (0..num_speakers)
        .map(|s| hard_flat_i8[c * num_speakers + s] as i32)
        .collect()
    })
    .collect();

  // Compare: partition-equivalent per chunk. The captured labels use
  // scipy's fcluster traversal order; ours use kodama's order remapped
  // through encounter sort. Both produce valid clusterings of the same
  // partition; the integer labels themselves are arbitrary names. We
  // build a global cluster-id permutation by walking chunks and
  // accumulating "got_label X co-occurs with want_label Y" (and vice
  // versa); a consistent partition equivalence requires both maps to
  // be one-to-one across all chunks.
  use std::collections::HashMap;
  let mut got_to_want: HashMap<i32, i32> = HashMap::new();
  let mut want_to_got: HashMap<i32, i32> = HashMap::new();
  for c in 0..num_chunks {
    for s in 0..num_speakers {
      let g = got[c][s];
      let w = want[c][s];
      // UNMATCHED on both sides is consistent.
      if g == UNMATCHED && w == UNMATCHED {
        continue;
      }
      // UNMATCHED only on one side → partition mismatch.
      if g == UNMATCHED || w == UNMATCHED {
        panic!("UNMATCHED mismatch at chunk {c}, speaker {s}: got {g}, want {w}");
      }
      // Establish or verify the consistent permutation.
      match got_to_want.get(&g).copied() {
        Some(existing) => assert_eq!(
          existing, w,
          "partition mismatch at chunk {c}, speaker {s}: got {g} previously mapped to {existing}, now {w}"
        ),
        None => {
          got_to_want.insert(g, w);
        }
      }
      match want_to_got.get(&w).copied() {
        Some(existing) => assert_eq!(
          existing, g,
          "partition mismatch at chunk {c}, speaker {s}: want {w} previously mapped from {existing}, now {g}"
        ),
        None => {
          want_to_got.insert(w, g);
        }
      }
    }
  }
  eprintln!(
    "[parity_pipeline] {} chunks × {} speakers — partition matches pyannote (cluster mapping: {:?})",
    num_chunks, num_speakers, got_to_want
  );
}
