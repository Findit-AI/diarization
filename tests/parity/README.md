# Pyannote parity test harness

A side-by-side runner that compares dia's diarization output against
`pyannote.audio` on a fixed clip, reporting DER (Diarization Error Rate).

**Spec §15 #43 / #46:** target DER ≤ 0.10 (rev-8 T3-I relaxed threshold)
on a curated multi-speaker clip.

## Layout

- `Cargo.toml` / `src/main.rs` — Rust binary `dia-parity` that runs
  `dia::Diarizer` on a clip and dumps RTTM to stdout.
- `python/pyproject.toml` / `python/reference.py` — pyannote.audio
  reference: same clip → reference RTTM.
- `python/score.py` — DER computation between two RTTMs.
- `run.sh` — end-to-end driver.

## Prerequisites

- The two ONNX models in `dia/models/` (or env vars
  `DIA_SEGMENT_MODEL_PATH` / `DIA_EMBED_MODEL_PATH`).
- A real multi-speaker WAV clip (16 kHz mono).
- `uv` for Python virtualenv management (`brew install uv` or
  `pip install uv`).

## Run

```bash
cd dia
./tests/parity/run.sh tests/fixtures/your_real_clip.wav
```

The script will install pyannote into a per-session venv, run both
reference + dia, and print DER. Exit code 0 iff DER ≤ 0.10.

## Notes

- The harness is **NOT** part of `cargo test`. It's a manual run for
  release-time validation.
- The synthetic 30 s tone fixture from
  `scripts/download-test-fixtures.sh` is **not suitable** — it has no
  real speech, so DER is undefined. Use a real clip from your own
  test corpus.
- Pyannote's API has shifted across versions; if `Pipeline.from_pretrained`
  fails, check the `pyannote.audio` changelog and update
  `python/reference.py`. Spec §15 #43 will be re-validated on each
  pyannote major release.

## Capture hook points (Phase 0, pyannote.audio 4.0.4)

`python/capture_intermediates.py` records pyannote intermediates for the
canonical 2-speaker clip via two complementary mechanisms. If
`pyannote.audio` is bumped past 4.0.4, the line numbers below shift and
both the script and this table must be re-synced; the `==` pin in
`python/pyproject.toml` makes such drift fail loudly.

### Public `hook` callback (`Pipeline.apply`)

`SpeakerDiarization.apply` invokes the user-supplied `hook(name, artefact, file=...)` callback at four named milestones:

| Event | `pipelines/speaker_diarization.py` | Artefact |
|-------|-----------------------------------|----------|
| `"segmentation"` | 594 | `(num_chunks, num_frames, local_num_speakers)` `SlidingWindowFeature` |
| `"speaker_counting"` | 614 | `(num_frames, 1)` `int` counts |
| `"embeddings"` | 637 | `(num_chunks, local_num_speakers, 256)` raw WeSpeaker embeddings (pre-PLDA) |
| `"discrete_diarization"` | 693 | `(num_frames, num_speakers)` post-reconstruct labels |

### `CapturingVBxClustering` subclass

The script replaces `pipeline.clustering` with a `VBxClustering`
subclass whose `__call__` body is a verbatim copy of
`pipelines/clustering.py:572-668` with capture statements interleaved.
That gives access to every interesting local variable inside the
clustering pass:

| Artefact | Source line | Notes |
|----------|-------------|-------|
| `train_embeddings`, `train_chunk_idx`, `train_speaker_idx` | 584 | post-`filter_embeddings` (drops low-quality slots) |
| `ahc_clusters` | 602 | AHC initialization labels |
| `post_xvec`, `post_plda` | 608 (we invoke `_xvec_tf` + `_plda_tf` separately) | PLDA stages: 256 → 128 (`sqrt(D_out)`-scaled L2-normed; D_out=128 → norm≈11.31) → 128 (whitened, not normed) |
| `qinit` | replicated from `utils/vbx.py:142-144` | smoothed one-hot of AHC init |
| `q_final`, `sp_final`, `elbo_trajectory` | invoke `VBx(..., return_model=True)` directly so we keep `Li` | final posteriors + ELBO curve per iteration |
| `soft_clusters` | 651 | input to constrained Hungarian |
| `hard_clusters` | 660-662 | post-`linear_sum_assignment` per chunk |
| `centroids` | 618-619 (or KMeans branch 632-643) | per-cluster centroids |

### Why we do not capture per-iteration VBx posteriors

`cluster_vbx` (`utils/vbx.py:140`) returns only `(gamma, pi)` —
per-iteration `gamma` lives inside `VBx()`'s EM loop and is discarded.
Forking that 80-line numpy function would be brittle. Instead we
capture `qinit` + final `q/sp` + the per-iteration `Li` (ELBO
trajectory). Same init + same final state + same convergence curve ⇒
same algorithm; that is sufficient evidence for a Rust-port parity
check.

### PLDA weight files

The HuggingFace snapshot of
[`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1)
ships:

- `plda/xvec_transform.npz` (134 KB) — keys `mean1`, `mean2`, `lda` (256→128 LDA matrix).
- `plda/plda.npz` (134 KB) — keys `mu`, `tr`, `psi`.

License: CC-BY-4.0 (see `models/plda/SOURCE.md` for attribution). The
capture script copies both to `models/plda/`; the Rust port (Phase 1+)
reads them directly and must reproduce the same transformation.
