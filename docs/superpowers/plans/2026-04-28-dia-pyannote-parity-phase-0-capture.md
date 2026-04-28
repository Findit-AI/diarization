# Pyannote Parity Capture — Phase 0 of Option A (revised)

> **For agentic workers:** REQUIRED SUB-SKILL — use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a frozen snapshot of `pyannote/speaker-diarization-community-1` (pyannote.audio 4.0.4) intermediate artifacts on the canonical 2-speaker clip, so the four subsequent Rust phases (PLDA → VBx → constrained Hungarian → centroid AHC) can be parity-validated commit-by-commit against ground truth instead of flying blind.

**Architecture (revised after reading pyannote source):** Hybrid capture — use pyannote's public `hook` callback for what it gives us (raw embeddings, segmentation, speaker counting, final discrete diarization), and replace `pipeline.clustering` with a `CapturingVBxClustering` subclass whose `__call__` body is a verbatim copy of `pyannote/audio/pipelines/clustering.py:572-668` with capture statements interleaved. The PLDA weight files (`xvec_transform.npz`, `plda.npz`) are copied out of the HuggingFace snapshot to a stable `models/plda/` location.

**Tech Stack:** Python 3.12 via `uv`, `pyannote.audio == 4.0.4` (exact pin), numpy, scipy, scikit-learn (already pulled in by pyannote), einops (already pulled in by pyannote), ffmpeg.

**Out of scope:**
- Any Rust code (Phases 1–4).
- Per-iteration VBx posteriors. `cluster_vbx` only returns final `(gamma, pi)`; per-iter values are inside `pyannote.audio.utils.vbx.VBx`'s EM loop. Forking that 80-line numpy function is brittle. Instead we capture initial `qinit`, final `q/sp`, and the ELBO trajectory `Li` (already returned by `VBx(..., return_model=True)`). Same init + same final state + same convergence curve ⇒ same algorithm; that is sufficient parity evidence.
- Replacing the existing high-level `reference.py` / `run.sh` flow — capture is an additive script, not a rewrite.

## Status (as of this revision)

- ✅ **Task 1** — fixture/dir scaffolding + canonical clip resampled to 16 kHz mono. Commit `e693c21`.
- ✅ **Task 2** — `pyannote.audio == 4.0.4` pinned, `inspect_pyannote.py` written and runs. Commits `9e95071`, `76cbd97`.
- ✅ **Hook research** — done by the controller during plan revision (see `## Pyannote 4.0.4 facts` below). The original Task 3 collapses to a small README write-up.
- ⏳ Tasks 3–9 remain.

---

## Pyannote 4.0.4 facts

**Hook callback events (`SpeakerDiarization.apply` in `pipelines/speaker_diarization.py`):**

| Event | Line | Artifact |
|-------|------|----------|
| `"segmentation"` | 594 | `(num_chunks, num_frames, local_num_speakers)` SlidingWindowFeature |
| `"speaker_counting"` | 614 | `(num_frames, 1)` int counts |
| `"embeddings"` | 637 | `(num_chunks, local_num_speakers, 256)` raw WeSpeaker embeddings (pre-PLDA) |
| `"discrete_diarization"` | 693 | `(num_frames, num_speakers)` post-reconstruct labels |

**Internal artifacts NOT exposed via hook** (must use `CapturingVBxClustering` subclass):

| Artifact | Source | What |
|----------|--------|------|
| `train_embeddings` | `clustering.py:584` | post-`filter_embeddings` (drops low-quality slots) |
| `ahc_clusters` | `clustering.py:602` | AHC init labels per train embedding |
| `post_xvec` | invoked at `clustering.py:608` (`self.plda(...)` calls `_xvec_tf` then `_plda_tf`) | LDA-projected, intermediate (128-d, L2-normed) |
| `post_plda` | same call | whitened PLDA features (128-d, NOT L2-normed) — input to VBx |
| `qinit` | `vbx.py:142-144` | smoothed one-hot encoding of `ahc_clusters` |
| `q_final`, `sp_final` | `clustering.py:609-614` | final VBx posteriors + mixing weights |
| `elbo_trajectory` (`Li`) | `VBx()` returns it; `cluster_vbx` discards it. Capture by re-calling `VBx` ourselves with `return_model=True` | per-iteration ELBO (≤20 entries) |
| `soft_clusters` | `clustering.py:651` | `(num_chunks, num_speakers, num_clusters)` distance scores |
| `hard_clusters` | `clustering.py:660-662` | post-Hungarian `(num_chunks, num_speakers)` labels |
| `centroids` | `clustering.py:618-619` (or post-KMeans branch) | `(num_clusters, 256)` |

**PLDA class** (`core/plda.py`):
- `pipeline._plda: PLDA` is the instance.
- `_plda._xvec_tf` and `_plda._plda_tf` are lambdas (instance attrs set by `vbx_setup`); both can be invoked directly to capture the intermediate.
- `pipeline._plda.lda_dimension == 128`.
- `pipeline._plda.phi` is the eigenvalue diag used by VBx.

**HF snapshot layout** (`pyannote/speaker-diarization-community-1`, license CC-BY-4.0):
```
plda/xvec_transform.npz   (134 KB — keys: mean1, mean2, lda)
plda/plda.npz             (134 KB — keys: mu, tr, psi)
segmentation/pytorch_model.bin
embedding/pytorch_model.bin
config.yaml, README.md
```

CC-BY-4.0 permits redistribution with attribution → safe to commit `models/plda/*.npz` to the dia repo with a `SOURCE.md` provenance file.

---

## File Structure

- ✅ Existing: `tests/parity/python/inspect_pyannote.py`
- ✅ Existing: `tests/parity/python/pyproject.toml` (pinned, documented)
- Create: `tests/parity/python/capture_intermediates.py` — main capture script.
- Create: `tests/parity/python/verify_capture.py` — reproducibility check.
- Modify: `tests/parity/run.sh` — point at canonical clip; invoke capture if fixtures missing.
- Modify: `tests/parity/README.md` — append hook-points table + Phase-0 docs.
- Create: `tests/parity/fixtures/01_dialogue/raw_embeddings.npz` — pre-PLDA WeSpeaker outputs.
- Create: `tests/parity/fixtures/01_dialogue/plda_embeddings.npz` — `post_xvec` + `post_plda` (+ chunk/slot index for the train subset).
- Create: `tests/parity/fixtures/01_dialogue/ahc_init_labels.npy` — AHC init labels per train embedding.
- Create: `tests/parity/fixtures/01_dialogue/vbx_state.npz` — `qinit`, `q_final`, `sp_final`, `elbo_trajectory`.
- Create: `tests/parity/fixtures/01_dialogue/clustering.npz` — `soft_clusters`, `hard_clusters`, `centroids`.
- Create: `tests/parity/fixtures/01_dialogue/reference.rttm` — final RTTM (matches existing `reference.py` output).
- Create: `tests/parity/fixtures/01_dialogue/manifest.json` — sha256 + version metadata.
- Create: `models/plda/xvec_transform.npz` — copied from HF snapshot.
- Create: `models/plda/plda.npz` — copied from HF snapshot.
- Create: `models/plda/SOURCE.md` — provenance + CC-BY-4.0 attribution.

---

## Task 3: README hook-point table

**Files:** Modify `tests/parity/README.md`.

The hook research is already done (see "Pyannote 4.0.4 facts" above). This task transcribes it into the README so future maintainers don't need to re-derive when they re-snapshot.

- [ ] **Step 1: Append hook-points section to `tests/parity/README.md`**

Append (do not rewrite — preserve existing content):

```markdown
## Capture hook points (Phase 0, pyannote.audio 4.0.4)

`python/capture_intermediates.py` uses two complementary mechanisms to
record intermediate artifacts. If `pyannote.audio` is bumped past
4.0.4, the line numbers below shift and the script must be re-synced.

### Public `hook` callback (Pipeline.apply)

| Event | `speaker_diarization.py` | Artifact |
|-------|--------------------------|----------|
| `"segmentation"` | 594 | `(num_chunks, num_frames, local_num_speakers)` SlidingWindowFeature |
| `"speaker_counting"` | 614 | `(num_frames, 1)` int counts |
| `"embeddings"` | 637 | `(num_chunks, local_num_speakers, 256)` raw WeSpeaker embeddings |
| `"discrete_diarization"` | 693 | `(num_frames, num_speakers)` post-reconstruct labels |

### `CapturingVBxClustering` subclass

The script replaces `pipeline.clustering` with a subclass whose
`__call__` body is a verbatim copy of `pipelines/clustering.py:572-668`
(VBxClustering.__call__) with capture statements interleaved. This
gives access to:

| Artifact | Source line | Notes |
|----------|-------------|-------|
| `train_embeddings`, `train_chunk_idx`, `train_speaker_idx` | 584 | post-`filter_embeddings` |
| `ahc_clusters` | 602 | AHC init labels |
| `post_xvec`, `post_plda` | 608 (we invoke `_xvec_tf` + `_plda_tf` separately) | PLDA stages |
| `qinit` | replicated from `vbx.py:142-144` | smoothed one-hot of AHC init |
| `q_final`, `sp_final`, `elbo_trajectory` | invoke `VBx(..., return_model=True)` directly | final posteriors + ELBO curve per iteration |
| `soft_clusters` | 651 | input to Hungarian |
| `hard_clusters` | 660-662 | post-Hungarian |
| `centroids` | 618-619 (or KMeans branch) | per-cluster centroids |

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

- `plda/xvec_transform.npz` (134 KB) — keys `mean1`, `mean2`, `lda`.
- `plda/plda.npz` (134 KB) — keys `mu`, `tr`, `psi`.

License: CC-BY-4.0. The capture script copies both to
`models/plda/` (committed with attribution in `models/plda/SOURCE.md`).
```

- [ ] **Step 2: Verify the README change reads cleanly**

```bash
git diff tests/parity/README.md
```

- [ ] **Step 3: Commit**

```bash
git add tests/parity/README.md
git commit -m "parity: document pyannote 4.0.4 hook points and capture strategy"
```

No `Co-Authored-By` trailer.

---

## Task 4: Capture script

**Files:** Create `tests/parity/python/capture_intermediates.py`.

The capture script is one self-contained file. The `CapturingVBxClustering.__call__` body is a verbatim port of `pipelines/clustering.py:572-668` with capture lines interleaved — it MUST stay in sync with the source. The pyannote version pin in `pyproject.toml` is the safety net: if pyannote ships a behavior change, `verify_capture.py` will fail and force a deliberate re-sync.

- [ ] **Step 1: Write the capture script**

Create `tests/parity/python/capture_intermediates.py`:

```python
"""Capture pyannote/speaker-diarization-community-1 intermediate artifacts.

Outputs (under tests/parity/fixtures/<clip-stem>/):
  - raw_embeddings.npz    (num_chunks, num_slots, 256) pre-PLDA WeSpeaker
  - plda_embeddings.npz   post_xvec + post_plda (num_train, 128) + train indices
  - ahc_init_labels.npy   (num_train,) AHC init labels
  - vbx_state.npz         qinit, q_final, sp_final, elbo_trajectory
  - clustering.npz        soft_clusters, hard_clusters, centroids
  - reference.rttm        final RTTM
  - manifest.json         sha256 + pyannote/numpy versions

Strategy:
  - hook callback for raw embeddings + final discrete diarization (public API).
  - Replace pipeline.clustering with CapturingVBxClustering subclass whose
    __call__ body mirrors pyannote 4.0.4's VBxClustering.__call__ verbatim
    with capture statements interleaved.

Usage:
  uv run python capture_intermediates.py <clip.wav>
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyannote.audio
from einops import rearrange
from huggingface_hub import snapshot_download
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.clustering import VBxClustering
from pyannote.audio.utils.vbx import VBx
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import cdist
from scipy.special import softmax as scipy_softmax
from sklearn.cluster import KMeans

PIPELINE_NAME = "pyannote/speaker-diarization-community-1"
VBX_INIT_SMOOTHING = 7.0  # cluster_vbx default in pyannote.audio.utils.vbx


@dataclass
class CaptureBuffer:
    # via hook
    segmentation: np.ndarray | None = None
    speaker_counting: np.ndarray | None = None
    raw_embeddings: np.ndarray | None = None
    discrete_diarization: np.ndarray | None = None

    # via CapturingVBxClustering
    train_embeddings: np.ndarray | None = None
    train_chunk_idx: np.ndarray | None = None
    train_speaker_idx: np.ndarray | None = None
    post_xvec: np.ndarray | None = None
    post_plda: np.ndarray | None = None
    ahc_clusters: np.ndarray | None = None
    qinit: np.ndarray | None = None
    q_final: np.ndarray | None = None
    sp_final: np.ndarray | None = None
    elbo_trajectory: list[float] = field(default_factory=list)
    soft_clusters: np.ndarray | None = None
    hard_clusters: np.ndarray | None = None
    centroids: np.ndarray | None = None


class CapturingVBxClustering(VBxClustering):
    """Records every intermediate of VBxClustering.__call__ to `self._buf`.

    The body of __call__ is a verbatim copy of
    pyannote.audio.pipelines.clustering.VBxClustering.__call__ from
    pyannote.audio==4.0.4 (clustering.py:572-668), with capture
    statements interleaved. If the upstream version is bumped, this
    body must be re-synced against the new source.
    """

    def __init__(self, *args, capture_buf: CaptureBuffer, **kwargs):
        super().__init__(*args, **kwargs)
        self._buf = capture_buf

    def __call__(self, embeddings, segmentations=None, num_clusters=None,
                 min_clusters=None, max_clusters=None, **kwargs):
        buf = self._buf
        constrained_assignment = self.constrained_assignment

        train_embeddings, chunk_idx, speaker_idx = self.filter_embeddings(
            embeddings, segmentations=segmentations
        )
        buf.train_embeddings = train_embeddings.copy()
        buf.train_chunk_idx = np.asarray(chunk_idx).copy()
        buf.train_speaker_idx = np.asarray(speaker_idx).copy()

        if train_embeddings.shape[0] < 2:
            num_chunks, num_speakers, _ = embeddings.shape
            hard_clusters = np.zeros((num_chunks, num_speakers), dtype=np.int8)
            soft_clusters = np.ones((num_chunks, num_speakers, 1))
            centroids = np.mean(train_embeddings, axis=0, keepdims=True)
            buf.hard_clusters = hard_clusters.copy()
            buf.soft_clusters = soft_clusters.copy()
            buf.centroids = centroids.copy()
            return hard_clusters, soft_clusters, centroids

        # AHC (clustering.py:597-603)
        train_embeddings_normed = train_embeddings / np.linalg.norm(
            train_embeddings, axis=1, keepdims=True
        )
        dendrogram = linkage(
            train_embeddings_normed, method="centroid", metric="euclidean"
        )
        ahc_clusters = fcluster(dendrogram, self.threshold, criterion="distance") - 1
        _, ahc_clusters = np.unique(ahc_clusters, return_inverse=True)
        buf.ahc_clusters = ahc_clusters.copy()

        # PLDA — capture xvec/plda stages separately by invoking the lambdas
        # directly. self.plda(x) is _plda_tf(_xvec_tf(x), lda_dim=...).
        post_xvec = self.plda._xvec_tf(train_embeddings)
        buf.post_xvec = post_xvec.copy()
        fea = self.plda._plda_tf(post_xvec, lda_dim=self.plda.lda_dimension)
        buf.post_plda = fea.copy()

        # VBx — replicate cluster_vbx() inline so we can capture qinit and
        # the ELBO trajectory `Li` (cluster_vbx discards them).
        qinit = np.zeros((len(ahc_clusters), int(ahc_clusters.max()) + 1))
        qinit[range(len(ahc_clusters)), ahc_clusters.astype(int)] = 1.0
        qinit = scipy_softmax(qinit * VBX_INIT_SMOOTHING, axis=1)
        buf.qinit = qinit.copy()

        gamma, pi, Li, _, _ = VBx(
            fea, self.plda.phi,
            Fa=self.Fa, Fb=self.Fb,
            pi=qinit.shape[1], gamma=qinit, maxIters=20,
            return_model=True,
        )
        buf.q_final = gamma.copy()
        buf.sp_final = pi.copy()
        buf.elbo_trajectory = [float(np.asarray(li).item()) for li in Li]

        # Centroids (clustering.py:617-620)
        num_chunks, num_speakers, dimension = embeddings.shape
        W = gamma[:, pi > 1e-7]
        centroids = (
            W.T @ train_embeddings.reshape(-1, dimension)
        ) / W.sum(0, keepdims=True).T

        # KMeans branch (clustering.py:625-643)
        auto_num_clusters, _ = centroids.shape
        if min_clusters is not None and auto_num_clusters < min_clusters:
            num_clusters = min_clusters
        elif max_clusters is not None and auto_num_clusters > max_clusters:
            num_clusters = max_clusters
        if num_clusters and num_clusters != auto_num_clusters:
            constrained_assignment = False
            kmeans_clusters = KMeans(
                n_clusters=num_clusters, n_init=3, random_state=42, copy_x=False
            ).fit_predict(train_embeddings_normed)
            centroids = np.vstack([
                np.mean(train_embeddings[kmeans_clusters == k], axis=0)
                for k in range(num_clusters)
            ])

        # e2k distances (clustering.py:646-655)
        e2k_distance = rearrange(
            cdist(
                rearrange(embeddings, "c s d -> (c s) d"),
                centroids,
                metric=self.metric,
            ),
            "(c s) k -> c s k", c=num_chunks, s=num_speakers,
        )
        soft_clusters = 2 - e2k_distance

        # Constrained Hungarian (clustering.py:658-662)
        if constrained_assignment:
            const = soft_clusters.min() - 1.0
            soft_clusters[segmentations.data.sum(1) == 0] = const
            hard_clusters = self.constrained_argmax(soft_clusters)
        else:
            hard_clusters = np.argmax(soft_clusters, axis=2)

        hard_clusters = hard_clusters.reshape(num_chunks, num_speakers)
        buf.soft_clusters = soft_clusters.copy()
        buf.hard_clusters = hard_clusters.copy()
        buf.centroids = centroids.copy()
        return hard_clusters, soft_clusters, centroids


def make_hook(buf: CaptureBuffer):
    """pyannote `hook(name, artifact, file=None, total=None, completed=None, ...)`."""
    def hook(name, artifact, file=None, total=None, completed=None, **kw):
        # Skip progress callbacks (total + completed kwargs).
        if total is not None or completed is not None:
            return
        if name == "segmentation":
            buf.segmentation = np.asarray(artifact.data).copy()
        elif name == "speaker_counting":
            buf.speaker_counting = np.asarray(artifact.data).copy()
        elif name == "embeddings":
            buf.raw_embeddings = np.asarray(artifact).copy()
        elif name == "discrete_diarization":
            buf.discrete_diarization = np.asarray(artifact.data).copy()
    return hook


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _export_plda_weights(repo_root: Path) -> None:
    """Copy plda/xvec_transform.npz + plda/plda.npz from HF cache to models/plda/."""
    snap = Path(snapshot_download(PIPELINE_NAME))
    dst = repo_root / "models" / "plda"
    dst.mkdir(parents=True, exist_ok=True)
    for fname in ("xvec_transform.npz", "plda.npz"):
        src = snap / "plda" / fname
        if not src.exists():
            raise SystemExit(f"could not find {src} in HF snapshot")
        target = dst / fname
        target.write_bytes(src.read_bytes())
        print(f"[capture] exported {fname} -> {target.relative_to(repo_root)}")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: capture_intermediates.py <clip.wav>")
    clip = Path(sys.argv[1]).resolve()
    if not clip.exists():
        raise SystemExit(f"clip not found: {clip}")
    out_dir = clip.parent
    print(f"[capture] clip: {clip}")
    print(f"[capture] out:  {out_dir}")

    pipeline = Pipeline.from_pretrained(PIPELINE_NAME)
    buf = CaptureBuffer()

    # Replace pipeline.clustering with our capturing subclass for one run.
    original_clustering = pipeline.clustering
    cap = CapturingVBxClustering(
        plda=pipeline._plda,
        metric=original_clustering.metric,
        constrained_assignment=original_clustering.constrained_assignment,
        capture_buf=buf,
    )
    cap.threshold = original_clustering.threshold
    cap.Fa = original_clustering.Fa
    cap.Fb = original_clustering.Fb
    pipeline.clustering = cap
    try:
        result = pipeline(str(clip), hook=make_hook(buf))
    finally:
        pipeline.clustering = original_clustering

    diarization = (
        result.speaker_diarization
        if hasattr(result, "speaker_diarization")
        else result
    )

    # Persist artifacts
    np.savez_compressed(
        out_dir / "raw_embeddings.npz",
        embeddings=buf.raw_embeddings,
    )
    np.savez_compressed(
        out_dir / "plda_embeddings.npz",
        post_xvec=buf.post_xvec,
        post_plda=buf.post_plda,
        train_chunk_idx=buf.train_chunk_idx,
        train_speaker_idx=buf.train_speaker_idx,
    )
    np.save(out_dir / "ahc_init_labels.npy", buf.ahc_clusters)
    np.savez_compressed(
        out_dir / "vbx_state.npz",
        qinit=buf.qinit,
        q_final=buf.q_final,
        sp_final=buf.sp_final,
        elbo_trajectory=np.array(buf.elbo_trajectory, dtype=np.float64),
    )
    np.savez_compressed(
        out_dir / "clustering.npz",
        soft_clusters=buf.soft_clusters,
        hard_clusters=buf.hard_clusters,
        centroids=buf.centroids,
    )

    rttm_path = out_dir / "reference.rttm"
    with rttm_path.open("w") as f:
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            f.write(
                f"SPEAKER {clip.stem} 1 {turn.start:.3f} {turn.duration:.3f}"
                f" <NA> <NA> {speaker} <NA> <NA>\n"
            )

    artifact_files = [
        "raw_embeddings.npz",
        "plda_embeddings.npz",
        "ahc_init_labels.npy",
        "vbx_state.npz",
        "clustering.npz",
        "reference.rttm",
    ]
    manifest = {
        "pyannote_audio_version": pyannote.audio.__version__,
        "numpy_version": np.__version__,
        "clip_path": str(clip),
        "clip_sha256": _file_sha256(clip),
        "artifacts": {f: _file_sha256(out_dir / f) for f in artifact_files},
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    repo_root = Path(__file__).resolve().parents[3]
    _export_plda_weights(repo_root)

    # Summary
    print(f"[capture] raw_embeddings: {buf.raw_embeddings.shape}")
    print(f"[capture] post_xvec:     {buf.post_xvec.shape}")
    print(f"[capture] post_plda:     {buf.post_plda.shape}")
    print(f"[capture] ahc_clusters:  {buf.ahc_clusters.shape}, "
          f"unique={sorted(set(buf.ahc_clusters.tolist()))}")
    print(f"[capture] q_final:       {buf.q_final.shape}")
    print(f"[capture] sp_final:      {buf.sp_final}")
    print(f"[capture] elbo iters:    {len(buf.elbo_trajectory)}")
    print(f"[capture] hard_clusters: {buf.hard_clusters.shape}, "
          f"unique={sorted(set(buf.hard_clusters.flatten().tolist()))}")
    print("[capture] done")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script end-to-end**

```bash
cd /Users/user/Develop/findit-studio/dia/tests/parity/python
uv run python capture_intermediates.py \
  ../fixtures/01_dialogue/clip_16k.wav
```

Expected on first run: ~30–60 s of pyannote inference (segmentation + embedding ONNX). Then the capture summary prints sensible shapes:
- `raw_embeddings`: `(num_chunks, 3, 256)` — `num_chunks` depends on clip; for 227 s clip with default 5 s overlap, expect ~45.
- `post_xvec`, `post_plda`: `(num_train, 128)` — `num_train` ≤ `num_chunks * 3`, after filtering low-quality slots.
- `ahc_clusters`: same length as `num_train`, unique labels ≥ 2 for the 2-speaker clip.
- `q_final`: `(num_train, S)` where S is the AHC-determined upper bound.
- `elbo iters: 20` (or fewer if convergence hits early).
- `hard_clusters`: `(num_chunks, 3)`, unique labels in `{-2, 0, 1}` (-2 is the "inactive" throwaway).

- [ ] **Step 3: Verify output files exist and are well-formed**

```bash
cd /Users/user/Develop/findit-studio/dia/tests/parity/python
uv run python -c "
import numpy as np
from pathlib import Path
d = Path('../fixtures/01_dialogue')
for name in ['raw_embeddings.npz','plda_embeddings.npz','vbx_state.npz','clustering.npz']:
    arr = np.load(d / name)
    print(f'{name}: keys={list(arr.keys())}')
    for k in arr.keys():
        print(f'  {k}: shape={arr[k].shape} dtype={arr[k].dtype}')
print('ahc_init_labels.npy:', np.load(d/'ahc_init_labels.npy').shape)
"
```

Expected: every npz has the keys listed above with finite values; ahc_init_labels is a 1-D int array.

- [ ] **Step 4: Sanity-check `post_xvec` scaling**

`xvec_tf` (`utils/vbx.py:211-213`) is `sqrt(lda.shape[1]) * l2_norm(...)` — the inner unit vector is scaled by `sqrt(D_out)` where `D_out = 128` (LDA output dim). So `post_xvec` rows have norm `sqrt(128) ≈ 11.31`, not 1.0. `post_plda` is whitened and generally has larger, non-unit norms.

```bash
uv run python -c "
import numpy as np
d = np.load('../fixtures/01_dialogue/plda_embeddings.npz')
post_xvec = d['post_xvec']
norms = np.linalg.norm(post_xvec, axis=1)
expected = np.sqrt(post_xvec.shape[1])
assert max(abs(norms - expected)) < 1e-3, f'expected norm {expected}, got {norms.mean()}'
print(f'post_xvec norms ≈ sqrt({post_xvec.shape[1]}) = {expected:.4f} ✓')
"
```

- [ ] **Step 5: Verify PLDA weight files exported**

```bash
ls -lh /Users/user/Develop/findit-studio/dia/models/plda/
```

Expected: `xvec_transform.npz` and `plda.npz`, each ~134 KB.

- [ ] **Step 6: Commit**

```bash
cd /Users/user/Develop/findit-studio/dia
git add tests/parity/python/capture_intermediates.py
git commit -m "parity: capture pyannote intermediates via hook + VBxClustering subclass"
```

(The artifact files + PLDA weights are committed in Tasks 5–6.) No `Co-Authored-By` trailer.

---

## Task 5: Commit captured artifacts + PLDA weights

**Files:**
- Add: `tests/parity/fixtures/01_dialogue/{raw_embeddings.npz, plda_embeddings.npz, ahc_init_labels.npy, vbx_state.npz, clustering.npz, reference.rttm, manifest.json}`
- Add: `models/plda/xvec_transform.npz`, `models/plda/plda.npz`
- Create: `models/plda/SOURCE.md`

- [ ] **Step 1: Confirm artifact sizes**

```bash
cd /Users/user/Develop/findit-studio/dia
du -sh tests/parity/fixtures/01_dialogue/* models/plda/*.npz
```

Total expected: ~1–3 MB for the captured artifacts plus 268 KB for the PLDA weights — small enough to commit.

- [ ] **Step 2: Write `models/plda/SOURCE.md`**

Find the snapshot revision SHA:

```bash
ls /Users/user/.cache/huggingface/hub/models--pyannote--speaker-diarization-community-1/snapshots/
```

(The directory name is the revision SHA.)

Create `models/plda/SOURCE.md`:

```markdown
# PLDA weights — pyannote/speaker-diarization-community-1

`xvec_transform.npz` and `plda.npz` are copied from the HuggingFace
snapshot of [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1).

- **License:** CC-BY-4.0. Attribution: PLDA model trained by
  [BUT Speech@FIT](https://speech.fit.vut.cz/). Integration of VBx in
  pyannote.audio by Jiangyu Han and Petr Pálka.
- **Snapshot revision:** `<sha>` (fill in from HF cache directory name).
- **Original layout:** `plda/xvec_transform.npz`, `plda/plda.npz`
  inside the snapshot.

`xvec_transform.npz` keys: `mean1`, `mean2`, `lda` (256→128 LDA matrix).
`plda.npz` keys: `mu`, `tr`, `psi`.

These files drive `pyannote.audio.utils.vbx.vbx_setup` (sourced by
`pyannote.audio.core.plda.PLDA`) to produce `xvec_tf` (centering +
LDA + L2-norm) and `plda_tf` (centering + whitening) — see Phase 0
README. The Rust port (Phase 1+) reads these same files and must
reproduce the same transformation.

**Refresh:** rerun `tests/parity/python/capture_intermediates.py`.
```

- [ ] **Step 3: Commit artifacts and weights**

```bash
git add models/plda/xvec_transform.npz models/plda/plda.npz models/plda/SOURCE.md
git add tests/parity/fixtures/01_dialogue/{raw_embeddings.npz,plda_embeddings.npz,ahc_init_labels.npy,vbx_state.npz,clustering.npz,reference.rttm,manifest.json}
git rm tests/parity/fixtures/01_dialogue/.gitkeep
git rm models/plda/.gitkeep
git commit -m "parity: snapshot pyannote intermediates + PLDA weights for 01_dialogue"
```

(The `.gitkeep` files are no longer needed — the directories now contain real files.) No `Co-Authored-By` trailer.

---

## Task 6: Reproducibility-check script

**Files:** Create `tests/parity/python/verify_capture.py`.

- [ ] **Step 1: Write the verifier**

```python
"""Re-run capture_intermediates and assert byte-identical outputs.

A green run proves the snapshot is deterministic — same pyannote
version + same clip + same hardware should always produce the same
artifacts. Phase 1+ (Rust ports) will rely on that determinism: when
a Rust port produces output that doesn't match, the failure is the
port, not flaky pyannote.

Usage:
  uv run python verify_capture.py <clip.wav>
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: verify_capture.py <clip.wav>")
    clip = Path(sys.argv[1]).resolve()
    snapshot_dir = clip.parent
    manifest_path = snapshot_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(
            f"no manifest at {manifest_path}; run capture_intermediates.py first"
        )
    expected = json.loads(manifest_path.read_text())["artifacts"]

    backup = snapshot_dir.parent / f".{snapshot_dir.name}.backup"
    if backup.exists():
        shutil.rmtree(backup)
    shutil.copytree(snapshot_dir, backup)
    print(f"[verify] backup written to {backup}")

    print("[verify] re-running capture...")
    subprocess.run(
        [sys.executable,
         str(Path(__file__).parent / "capture_intermediates.py"),
         str(clip)],
        check=True,
    )

    mismatches: list[str] = []
    for name, expected_hash in expected.items():
        actual = _sha256(snapshot_dir / name)
        if actual != expected_hash:
            mismatches.append(f"  {name}: {actual} != {expected_hash}")

    if mismatches:
        print("[verify] MISMATCHES:")
        for m in mismatches:
            print(m)
        sys.exit(1)
    print("[verify] all artifacts match — snapshot is reproducible")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

```bash
cd /Users/user/Develop/findit-studio/dia/tests/parity/python
uv run python verify_capture.py ../fixtures/01_dialogue/clip_16k.wav
```

Expected: `[verify] all artifacts match — snapshot is reproducible`.

If artifacts mismatch:
- Most likely cause: an RNG seed wasn't fixed somewhere. The capture script does NOT set `np.random.seed` because pyannote's pipeline initialization is deterministic for this model (no `pi=int` branch is taken — `gamma` is always `qinit`-driven). If a mismatch appears, investigate which artifact differs first; the diff usually points at the source.
- Less likely: pyannote's underlying ONNX runtime exhibits non-determinism on some hardware. If reproducibility cannot be achieved, document the affected artifacts in this verifier (allow-list) and proceed.

- [ ] **Step 3: Clean up the backup**

```bash
rm -rf /Users/user/Develop/findit-studio/dia/tests/parity/fixtures/.01_dialogue.backup
```

- [ ] **Step 4: Commit**

```bash
git add tests/parity/python/verify_capture.py
git commit -m "parity: verify_capture.py — assert byte-identical re-runs"
```

---

## Task 7: Wire `run.sh` + extend README

**Files:**
- Modify: `tests/parity/run.sh`
- Modify: `tests/parity/README.md` (append usage section)

- [ ] **Step 1: Update `run.sh` to default to canonical clip + invoke capture if missing**

Replace `tests/parity/run.sh` with:

```bash
#!/usr/bin/env bash
# Pyannote parity harness.
#
# Requires:
# - models/segmentation-3.0.onnx and models/wespeaker_resnet34_lm.onnx
# - models/plda/xvec_transform.npz and models/plda/plda.npz (Phase 0)
# - uv
# - the clip path; defaults to the canonical 2-speaker fixture

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/../.."

DEFAULT_CLIP="$SCRIPT_DIR/fixtures/01_dialogue/clip_16k.wav"
CLIP="${1:-$DEFAULT_CLIP}"
ABS_CLIP="$(cd "$ROOT" && realpath "$CLIP")"
SNAPSHOT_DIR="$(dirname "$ABS_CLIP")"
MANIFEST="$SNAPSHOT_DIR/manifest.json"

cd "$SCRIPT_DIR/python"
if [ ! -d .venv ]; then
  uv venv
fi
uv pip install -e .

if [ ! -f "$MANIFEST" ]; then
  echo "[run.sh] no manifest at $MANIFEST; running Phase-0 capture..."
  uv run python capture_intermediates.py "$ABS_CLIP"
else
  echo "[run.sh] reusing existing snapshot at $SNAPSHOT_DIR"
fi

# Reuse the captured RTTM as the reference (saves rerunning pyannote).
REF_RTTM="$SNAPSHOT_DIR/reference.rttm"

cd "$ROOT"
cargo run --release --manifest-path tests/parity/Cargo.toml -- "$CLIP" \
  > "$SCRIPT_DIR/hyp.rttm"

cd "$SCRIPT_DIR/python"
uv run python score.py "$REF_RTTM" "$SCRIPT_DIR/hyp.rttm"
```

- [ ] **Step 2: Smoke-test the harness**

```bash
cd /Users/user/Develop/findit-studio/dia
bash tests/parity/run.sh
```

Expected: capture is skipped (manifest already present); dia runs; DER printed.

If the dia binary errors, ALSO acceptable for Phase 0 — the harness wiring is what's being verified, not the DER number itself. Phase 1+ Rust ports drive the DER down.

- [ ] **Step 3: Append usage section to `tests/parity/README.md`**

Append:

```markdown
## Phase 0: Pyannote intermediates snapshot

The canonical fixture lives at `tests/parity/fixtures/01_dialogue/`.
It is produced by `python/capture_intermediates.py` and is
**deterministic** — same pyannote version + same clip + same hardware
must produce byte-identical artifacts.

### Refreshing the snapshot

```bash
cd tests/parity/python
uv run python capture_intermediates.py \
  ../fixtures/01_dialogue/clip_16k.wav
```

This overwrites every artifact under `01_dialogue/` and re-exports
`models/plda/{xvec_transform,plda}.npz`.

### Verifying determinism

```bash
cd tests/parity/python
uv run python verify_capture.py \
  ../fixtures/01_dialogue/clip_16k.wav
```

A green run is required before opening a Phase-1+ Rust PR — every Rust
port parity-checks against the snapshot.

### Why we pin pyannote

`pyproject.toml` pins `pyannote.audio == 4.0.4`. If upstream pyannote
ships a behavior change, `verify_capture.py` will fail and force a
deliberate snapshot refresh + version bump rather than letting the
change leak silently into Rust-port reviews.
```

- [ ] **Step 4: Commit**

```bash
git add tests/parity/run.sh tests/parity/README.md
git commit -m "parity: wire capture into run.sh, document Phase-0 usage"
```

---

## Task 8: Push + open PR

**Files:** None (workflow task).

- [ ] **Step 1: Push the branch**

```bash
git push -u origin feat/parity
```

- [ ] **Step 2: Open the PR**

```bash
gh pr create --title "parity: Phase-0 pyannote intermediates capture" --body "$(cat <<'EOF'
## Summary
Implements Phase 0 of Option A pyannote-parity
(`docs/superpowers/plans/2026-04-28-dia-pyannote-parity-phase-0-capture.md`).

This PR adds NO Rust code. It produces a deterministic snapshot of
`pyannote/speaker-diarization-community-1` (pyannote.audio 4.0.4)
intermediate artifacts on the canonical 2-speaker clip so subsequent
Rust ports (Phase 1 PLDA → Phase 2 VBx → Phase 3 constrained Hungarian
→ Phase 4 centroid AHC) can be parity-validated commit-by-commit.

## What's captured

Via the public `hook` callback:
- raw WeSpeaker embeddings (pre-PLDA)
- segmentation, speaker counting, final discrete diarization

Via a `CapturingVBxClustering` subclass (a verbatim port of pyannote's
`VBxClustering.__call__` with capture statements interleaved):
- post-`xvec_tf` PLDA-projected embeddings (128-d, L2-normed)
- post-`plda_tf` whitened embeddings (128-d, input to VBx)
- AHC initialization labels
- VBx `qinit`, final `q/sp`, ELBO trajectory `Li`
- soft_clusters (input to Hungarian), hard_clusters, centroids

Per-iteration VBx posteriors are deliberately NOT captured (deferred);
the ELBO trajectory + initial + final posteriors give sufficient
parity evidence for a Rust port.

Plus `models/plda/{xvec_transform,plda}.npz` (CC-BY-4.0, with
attribution in `models/plda/SOURCE.md`).

## Verification
- `tests/parity/python/verify_capture.py` re-runs the capture and
  asserts byte-identical outputs.
- `tests/parity/run.sh` continues to work; it now defaults to the
  canonical fixture and reuses the captured RTTM as the pyannote
  reference (no need to rerun pyannote per parity check).

## Test plan
- [ ] `tests/parity/run.sh` runs end-to-end on a clean checkout
- [ ] `verify_capture.py` is green
- [ ] All artifact files under `tests/parity/fixtures/01_dialogue/`
  open in numpy with sensible shapes
- [ ] `manifest.json` lists every artifact with a sha256

## Out of scope
- Phases 1–4 (Rust ports). Each gets its own plan + PR.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 3: Note follow-on plans**

Phases 1–4 will each get their own plan documents written once Phase 0
lands. Each follow-on plan will (a) point at this snapshot for parity,
(b) define per-PR acceptance criteria using DER deltas measured against
this snapshot. Plan filenames will follow `YYYY-MM-DD-dia-<phase>.md`.

---

## Self-review

**1. Spec coverage.** Phase 0 of the analysis doc requires capturing pyannote intermediates so subsequent Rust ports can be parity-validated. Tasks 3–8 cover every artifact named in the analysis doc:
- raw embeddings → captured via `"embeddings"` hook (Task 4).
- post-`xvec_tf` → captured via `_xvec_tf(train_embeddings)` invocation (Task 4).
- post-`plda_tf` → captured via `_plda_tf(post_xvec, ...)` invocation (Task 4).
- AHC init → captured from `ahc_clusters` local variable (Task 4).
- VBx posteriors → final `q, sp` + ELBO trajectory captured by replicating `cluster_vbx` inline (Task 4). Per-iter intentionally skipped (out-of-scope, justified).
- Hungarian final labels → `hard_clusters` (Task 4).
- PLDA weight files → exported in Task 4 (`_export_plda_weights`), committed in Task 5.
- Reproducibility verifier → Task 6.

**2. Placeholder scan.** No `<placeholder>`, `TBD`, `EMBEDDING_HOOK_TARGET`, or "Replace these imports" strings remaining. The only `<sha>` placeholder in Task 5 Step 2 is for the actual snapshot revision SHA, which the implementer fills in from the local HF cache directory name — that's data extraction, not deferred design.

**3. Type consistency.** `CaptureBuffer` field names are referenced consistently in the script's persistence step and in the README's hook-points table. The script's class signature (`CapturingVBxClustering`) and its instantiation in `main()` match. PLDA file names (`xvec_transform.npz`, `plda.npz`) are consistent across the script, README, and `SOURCE.md`.

**4. Risk areas explicitly called out:**
- pyannote 4.0.4 internal API churn (Task 4 Step 1, Task 7 Step 3) — `==` pin makes drift fail loudly.
- `CapturingVBxClustering.__call__` body must stay in sync with `clustering.py:572-668` — explicit comment in the source warns of this.
- Reproducibility (Task 6 Step 2) — investigate first via the manifest mismatch list; allow-list as last resort.

---

**Plan complete and superseded the original placeholder-driven version.** Phases 1–4 will get their own plan docs once this lands.
