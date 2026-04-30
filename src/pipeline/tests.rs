//! Model-free unit tests for `diarization::pipeline`.

use crate::pipeline::{AssignEmbeddingsInput, assign_embeddings};
use nalgebra::{DMatrix, DVector};

/// Pyannote one-cluster fast path (`clustering.py:588-594`): when
/// fewer than 2 active training embeddings survive `filter_embeddings`,
/// pyannote returns `hard_clusters = np.zeros((num_chunks,
/// num_speakers))`. The Rust port must do the same instead of
/// erroring — short clips, sparse speech, and single-usable-speaker
/// recordings all hit this path. Codex review HIGH round 1 of Phase 5.
#[test]
fn assign_embeddings_returns_one_cluster_when_num_train_lt_2() {
  let num_chunks = 3;
  let num_speakers = 2;
  let embed_dim = 4;
  let plda_dim = 4;
  let num_frames = 8;
  let embeddings = DMatrix::<f64>::from_element(num_chunks * num_speakers, embed_dim, 0.5);
  let segmentations = vec![0.5; num_chunks * num_frames * num_speakers];

  // num_train = 1: only one active embedding survives filter_embeddings.
  let post_plda = DMatrix::<f64>::from_element(1, plda_dim, 0.1);
  let phi = DVector::<f64>::from_element(plda_dim, 1.0);
  let train_chunk_idx = vec![0usize];
  let train_speaker_idx = vec![0usize];
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
    threshold: 0.6,
    fa: 0.07,
    fb: 0.8,
    max_iters: 20,
  };
  let got = assign_embeddings(&input).expect("fast path must succeed, not error");
  assert_eq!(got.len(), num_chunks);
  for chunk_row in &got {
    assert_eq!(chunk_row.len(), num_speakers);
    for &k in chunk_row {
      assert_eq!(k, 0, "every speaker in every chunk must be cluster 0");
    }
  }
}

/// Zero-column `post_plda` is rejected at the boundary — a schema drift
/// or wrong array fed to the pipeline would otherwise let VBx iterate
/// on no PLDA evidence and produce plausible hard_clusters from prior
/// alone. Codex review MEDIUM round 7 of Phase 5.
#[test]
fn rejects_zero_column_post_plda() {
  let num_chunks = 3;
  let num_speakers = 2;
  let embed_dim = 4;
  let num_frames = 8;
  let embeddings = DMatrix::<f64>::from_element(num_chunks * num_speakers, embed_dim, 0.5);
  let segmentations = vec![0.5; num_chunks * num_frames * num_speakers];
  // post_plda has zero columns (PLDA dim = 0).
  let post_plda = DMatrix::<f64>::zeros(2, 0);
  let phi = DVector::<f64>::zeros(0);
  let train_chunk_idx = vec![0usize, 1];
  let train_speaker_idx = vec![0usize, 1];
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
    threshold: 0.6,
    fa: 0.07,
    fb: 0.8,
    max_iters: 20,
  };
  let result = assign_embeddings(&input);
  assert!(matches!(result, Err(crate::pipeline::Error::Shape(_))), "got {result:?}");
}

/// Zero active embeddings (`num_train == 0`) also takes the fast path —
/// pyannote's check is `< 2`, not `== 1`. Skipping AHC/VBx entirely
/// avoids the empty-mean NaN that would otherwise propagate from
/// `np.mean(empty, axis=0)`.
#[test]
fn assign_embeddings_returns_one_cluster_when_num_train_zero() {
  let num_chunks = 2;
  let num_speakers = 3;
  let embed_dim = 4;
  let plda_dim = 4;
  let num_frames = 8;
  let embeddings = DMatrix::<f64>::from_element(num_chunks * num_speakers, embed_dim, 0.5);
  let segmentations = vec![0.5; num_chunks * num_frames * num_speakers];
  let post_plda = DMatrix::<f64>::zeros(0, plda_dim);
  let phi = DVector::<f64>::from_element(plda_dim, 1.0);
  let input = AssignEmbeddingsInput {
    embeddings: &embeddings,
    num_chunks,
    num_speakers,
    segmentations: &segmentations,
    num_frames,
    post_plda: &post_plda,
    phi: &phi,
    train_chunk_idx: &[],
    train_speaker_idx: &[],
    threshold: 0.6,
    fa: 0.07,
    fb: 0.8,
    max_iters: 20,
  };
  let got = assign_embeddings(&input).expect("zero-train fast path must succeed");
  for chunk_row in &got {
    for &k in chunk_row {
      assert_eq!(k, 0);
    }
  }
}
