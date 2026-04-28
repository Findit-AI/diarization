# Pyannote Parity Capture — Phase 0 of Option A

> **For agentic workers:** REQUIRED SUB-SKILL — use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a frozen snapshot of `pyannote/speaker-diarization-community-1`'s intermediate artifacts on the canonical 2-speaker clip, so the four subsequent Rust phases (PLDA → VBx → constrained Hungarian → centroid AHC) can be parity-validated commit-by-commit against ground truth instead of flying blind.

**Architecture:** A new Python capture script monkey-patches the pyannote pipeline at well-defined seams to dump intermediate tensors to disk. The PLDA weight files (`xvec_transform.npz`, `plda.npz`) are copied out of the HuggingFace cache to a stable `models/plda/` location. A reproducibility-check script re-runs the capture and asserts byte-identical outputs so we know the snapshot is deterministic.

**Tech Stack:** Python 3.11 via `uv`, `pyannote.audio` (community-1 release), `numpy`, `ffmpeg` for resampling.

**Out of scope:**
- Any Rust code (Phases 1–4).
- DER computation against the captured labels (already exists in `tests/parity/python/score.py`).
- Replacing the existing high-level `reference.py` / `run.sh` flow — capture is an additive script, not a rewrite.

---

## Prerequisites & open questions

1. **Branch.** Recommendation: cut a feature branch `feat/parity` (or worktree) from `0.1.0`. Five weeks of Option-A churn shouldn't sit on the active release branch. Confirm before Task 1.
2. **License.** `pyannote/speaker-diarization-community-1` distributes PLDA weights under its own terms; verify before redistributing the `.npz` files inside the dia repo. If we cannot redistribute, store them in `models/plda/` (gitignored) and have the capture script fetch on first run.
3. **Pyannote version.** `tests/parity/python/pyproject.toml` currently pins `pyannote.audio >= 3.1`. The community-1 pipeline may require a specific 4.x release; Task 2 confirms.

## File Structure

- Create: `tests/parity/python/capture_intermediates.py` — main capture script (monkey-patches pyannote internals; dumps artifacts).
- Create: `tests/parity/python/verify_capture.py` — re-runs capture + asserts byte-identical outputs.
- Create: `tests/parity/python/inspect_pyannote.py` — small "what hooks does this version expose?" research utility.
- Modify: `tests/parity/python/pyproject.toml` — pin pyannote version + add numpy.
- Modify: `tests/parity/run.sh` — point at canonical clip; capture artifacts on first run if missing.
- Modify: `tests/parity/README.md` — document Phase-0 outputs and refresh procedure.
- Create: `tests/parity/fixtures/01_dialogue/clip_16k.wav` — resampled canonical clip (16 kHz mono); gitignored if size > 5 MB.
- Create: `tests/parity/fixtures/01_dialogue/raw_embeddings.npz` — pre-PLDA WeSpeaker outputs + chunk/slot indexing.
- Create: `tests/parity/fixtures/01_dialogue/plda_embeddings.npz` — post-`xvec_tf` and post-`plda_tf`.
- Create: `tests/parity/fixtures/01_dialogue/ahc_init_labels.npy` — AHC initialization labels per (chunk, slot).
- Create: `tests/parity/fixtures/01_dialogue/vbx_posteriors.npz` — per-iteration `q[t,k]`, `sp[k]`, `tr[k1,k2]`.
- Create: `tests/parity/fixtures/01_dialogue/final_labels.npy` — post-Hungarian per-(chunk,slot) labels.
- Create: `tests/parity/fixtures/01_dialogue/reference.rttm` — final RTTM (already produced by `reference.py`; copy here).
- Create: `tests/parity/fixtures/01_dialogue/manifest.json` — checksums + pyannote version + numpy version + clip duration.
- Create: `models/plda/xvec_transform.npz` — copied from HF cache.
- Create: `models/plda/plda.npz` — copied from HF cache.
- Create: `models/plda/SOURCE.md` — provenance (HF model, commit hash, license).
- Modify: `.gitignore` — exclude `tests/parity/fixtures/01_dialogue/clip_16k.wav` if oversized; include the rest.

---

## Task 1: Branch + canonical-clip resample

**Files:**
- Modify: `tests/parity/run.sh:12-13`
- Create: `tests/parity/fixtures/01_dialogue/clip_16k.wav`
- Modify: `.gitignore`

- [ ] **Step 1: Confirm branch with user; cut feature branch**

```bash
# From repo root, after user confirms.
git checkout 0.1.0
git pull origin 0.1.0
git checkout -b feat/parity
```

If the user prefers a worktree:

```bash
git worktree add ../dia-parity feat/parity
cd ../dia-parity
```

- [ ] **Step 2: Create the fixture directory**

```bash
mkdir -p tests/parity/fixtures/01_dialogue
mkdir -p models/plda
```

- [ ] **Step 3: Resample the canonical clip to 16 kHz mono**

```bash
SOURCE="/Users/user/Develop/findit-studio/indexer/assets/audios/01_人声_自录双人对话.wav"
ffmpeg -y -i "$SOURCE" -ac 1 -ar 16000 -c:a pcm_f32le \
  tests/parity/fixtures/01_dialogue/clip_16k.wav
```

- [ ] **Step 4: Verify the resampled clip**

```bash
ffprobe -v error -show_streams tests/parity/fixtures/01_dialogue/clip_16k.wav \
  2>&1 | grep -E "^(codec_name|sample_rate|channels|duration)="
```

Expected output:

```
codec_name=pcm_f32le
sample_rate=16000
channels=1
duration=226.96
```

- [ ] **Step 5: Decide gitignore policy**

The resampled WAV is roughly `227 s × 16000 Hz × 4 B ≈ 14 MB`. That's too big for a committed fixture. Add it to `.gitignore`:

```bash
cat >> .gitignore <<'EOF'

# Phase-0 parity capture: large local artifacts.
tests/parity/fixtures/*/clip_16k.wav
EOF
```

The capture script regenerates this from the source path on demand (Task 3 wires that up).

- [ ] **Step 6: Commit**

```bash
git add .gitignore tests/parity/fixtures/01_dialogue/.gitkeep models/plda/.gitkeep
touch tests/parity/fixtures/01_dialogue/.gitkeep models/plda/.gitkeep
git add .
git commit -m "parity: scaffold Phase-0 capture directories"
```

---

## Task 2: Pin pyannote version + verify pipeline loads

**Files:**
- Modify: `tests/parity/python/pyproject.toml:5-8`
- Create: `tests/parity/python/inspect_pyannote.py`

- [ ] **Step 1: Look up the pyannote.audio release that ships speaker-diarization-community-1**

```bash
# Quick check: does community-1 exist on the HF hub right now?
curl -fsSL https://huggingface.co/api/models/pyannote/speaker-diarization-community-1 \
  | python3 -c "import sys, json; m = json.load(sys.stdin); print('cardData:', m.get('cardData')); print('library_name:', m.get('library_name'))"
```

Read the model card to find the required pyannote.audio version. Record it for Step 2.

- [ ] **Step 2: Pin the version in `pyproject.toml`**

Modify `tests/parity/python/pyproject.toml` to a precise pin (replace `<X.Y.Z>` with the version found in Step 1):

```toml
[project]
name = "dia-parity-reference"
version = "0.0.0"
requires-python = ">=3.10"
dependencies = [
  "pyannote.audio == <X.Y.Z>",
  "pyannote.metrics >= 3.2",
  "numpy >= 1.26",
]
```

A pinned `==` (not `>=`) is intentional: the capture is a frozen snapshot. If pyannote ships a behavior change, we want the Phase-0 verification to fail loudly so we can re-snapshot deliberately.

- [ ] **Step 3: Refresh the venv**

```bash
cd tests/parity/python
rm -rf .venv
uv venv
uv pip install -e .
```

- [ ] **Step 4: Write the inspect utility**

Create `tests/parity/python/inspect_pyannote.py`:

```python
"""Print the structure of pyannote/speaker-diarization-community-1.

Used during Phase 0 to locate hook points for the capture script. Not
shipped as a runnable test — it exists to document what we monkey-patch.
"""

import inspect
from pathlib import Path

from pyannote.audio import Pipeline
from pyannote.audio.pipelines import speaker_diarization as sd_mod


def main() -> None:
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1"
    )
    print("Pipeline class:", type(pipeline).__name__)
    print("Pipeline module:", type(pipeline).__module__)
    print()
    print("Top-level attributes:")
    for name in sorted(vars(pipeline)):
        val = getattr(pipeline, name)
        print(f"  {name}: {type(val).__name__}")
    print()
    print("Methods (own only):")
    for name in sorted(vars(type(pipeline))):
        if name.startswith("_"):
            continue
        member = getattr(type(pipeline), name)
        if callable(member):
            try:
                sig = inspect.signature(member)
            except (TypeError, ValueError):
                sig = "(<unavailable>)"
            print(f"  {name}{sig}")
    print()
    print("Module file:", Path(inspect.getfile(sd_mod)).resolve())
    print("Clustering attribute:", getattr(pipeline, "clustering", None))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run it**

```bash
cd tests/parity/python
uv run python inspect_pyannote.py | tee /tmp/pyannote-shape.txt
```

Expected: a listing of pipeline attributes, methods, and the source file path. Save the output for reference in Task 4.

- [ ] **Step 6: Commit**

```bash
git add tests/parity/python/pyproject.toml tests/parity/python/inspect_pyannote.py
git commit -m "parity: pin pyannote version, add internal-shape inspector"
```

---

## Task 3: Locate pyannote hook points (research)

**Files:**
- Read-only: pyannote.audio source under the venv.
- Output: short notes appended to `tests/parity/README.md` documenting which methods the capture script monkey-patches.

This task produces no code — it is the prerequisite research that prevents Tasks 4–8 from being placeholder-driven. **Do not skip it.** The five subsequent tasks all assume the hook points identified here.

- [ ] **Step 1: Find the pipeline source file**

```bash
cd tests/parity/python
PA_FILE=$(uv run python -c "import pyannote.audio.pipelines.speaker_diarization as m; print(m.__file__)")
echo "$PA_FILE"
```

- [ ] **Step 2: Locate where raw WeSpeaker embeddings are produced**

```bash
grep -n "embedding\|WeSpeaker\|infer_speaker_embedding" "$PA_FILE" | head -20
```

Expected: a function or method in the pipeline that calls into the embedding ONNX model and returns a `(num_chunks, num_slots, embedding_dim)`-shaped tensor. Record the qualified name (e.g. `SpeakerDiarization.get_embeddings` or similar).

- [ ] **Step 3: Locate the PLDA / clustering entry point**

```bash
grep -rn "vbx\|plda\|xvec_transform\|AgglomerativeClustering" \
  "$(dirname "$PA_FILE")/.." | head -20
```

Look for: (a) the file that contains `xvec_tf` / `plda_tf` (per the analysis doc, `utils/vbx.py:181-217`), (b) the AHC entry point used for VBx initialization, (c) the VBx HMM EM loop, (d) the constrained Hungarian step.

- [ ] **Step 4: Locate the HF cache path for the PLDA weight files**

```bash
uv run python -c "
from huggingface_hub import HfFileSystem, snapshot_download
import os
path = snapshot_download('pyannote/speaker-diarization-community-1')
print('Snapshot at:', path)
for f in os.listdir(path):
    print(' ', f)
"
```

Expected: a path under `~/.cache/huggingface/hub/...` containing (among other things) `xvec_transform.npz` and `plda.npz` or equivalents.

- [ ] **Step 5: Document hook points**

Append a section to `tests/parity/README.md`:

```markdown
## Capture hook points (Phase 0)

The capture script (`python/capture_intermediates.py`) monkey-patches
`pyannote.audio` internals at the following seams. If pyannote moves
these in a future release, the parity-capture pin in `pyproject.toml`
must be bumped *deliberately* and the snapshot regenerated.

| Hook | Source | Captures |
|------|--------|----------|
| `<class>.<method>` | `<file>:<lineno>` | raw WeSpeaker `(num_chunks, num_slots, dim)` embeddings |
| `<class>.<method>` | `<file>:<lineno>` | PLDA-projected `(num_chunks, num_slots, plda_dim)` embeddings |
| `<class>.<method>` | `<file>:<lineno>` | AHC initialization labels |
| `<class>.<method>` | `<file>:<lineno>` | VBx posteriors `q[t,k]` per iteration |
| `<class>.<method>` | `<file>:<lineno>` | constrained Hungarian per-chunk-slot labels |

Fill the table in with what Steps 2–3 found.
```

- [ ] **Step 6: Commit**

```bash
git add tests/parity/README.md
git commit -m "parity: document pyannote hook points for capture script"
```

---

## Task 4: Capture script skeleton + raw embeddings

**Files:**
- Create: `tests/parity/python/capture_intermediates.py`

- [ ] **Step 1: Write the capture skeleton**

Create `tests/parity/python/capture_intermediates.py`:

```python
"""Capture pyannote/speaker-diarization-community-1 intermediates.

Outputs (all under tests/parity/fixtures/<clip-stem>/):
  - raw_embeddings.npz     : pre-PLDA WeSpeaker embeddings + chunk/slot index
  - plda_embeddings.npz    : post-PLDA (post-xvec_tf, post-plda_tf)
  - ahc_init_labels.npy    : AHC initialization labels
  - vbx_posteriors.npz     : per-iter q[t,k] tensors
  - final_labels.npy       : post-Hungarian per-(chunk,slot) labels
  - reference.rttm         : final RTTM
  - manifest.json          : checksums + versions

Usage:
  uv run python capture_intermediates.py <clip.wav>
"""

from __future__ import annotations

import hashlib
import json
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyannote.audio
from pyannote.audio import Pipeline


@dataclass
class CaptureBuffer:
    raw_embeddings: np.ndarray | None = None
    plda_embeddings_post_xvec: np.ndarray | None = None
    plda_embeddings_post_plda: np.ndarray | None = None
    ahc_init_labels: np.ndarray | None = None
    vbx_posteriors_per_iter: list[np.ndarray] = field(default_factory=list)
    vbx_sp_per_iter: list[np.ndarray] = field(default_factory=list)
    final_labels: np.ndarray | None = None


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@contextmanager
def patched_pipeline(buf: CaptureBuffer):
    """Install monkey-patches on pyannote internals for one run, then restore.

    Hook targets are filled in from the Task 3 research — replace each
    `<class>.<method>` placeholder with the qualified name that Task 3
    identified.
    """
    # TASK 3 RESEARCH FILL-IN: import the modules + classes pyannote
    # exposes for community-1. Example shape:
    #
    #   from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization
    #   from pyannote.audio.pipelines.utils.vbx import vbx_run, plda_tf, xvec_tf
    #
    # then capture original methods, replace with wrappers that record
    # to `buf`, yield, and restore in `finally`.
    raise NotImplementedError(
        "Fill in the patched_pipeline body using hooks identified in Task 3."
    )


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python capture_intermediates.py <clip.wav>")
    clip = Path(sys.argv[1]).resolve()
    if not clip.exists():
        raise SystemExit(f"clip not found: {clip}")

    out_dir = clip.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[capture] clip: {clip}")
    print(f"[capture] out:  {out_dir}")

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1"
    )
    buf = CaptureBuffer()
    with patched_pipeline(buf):
        result = pipeline(str(clip))
    diarization = (
        result.speaker_diarization
        if hasattr(result, "speaker_diarization")
        else result
    )

    # Persist artifacts.
    np.savez_compressed(
        out_dir / "raw_embeddings.npz",
        embeddings=buf.raw_embeddings,
    )
    print(f"[capture] raw_embeddings: {buf.raw_embeddings.shape}")

    # ... remaining artifacts populated by Tasks 5-8 ...

    rttm_path = out_dir / "reference.rttm"
    with rttm_path.open("w") as f:
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            f.write(
                f"SPEAKER {clip.stem} 1 {turn.start:.3f} {turn.duration:.3f}"
                f" <NA> <NA> {speaker} <NA> <NA>\n"
            )

    manifest = {
        "pyannote_audio_version": pyannote.audio.__version__,
        "numpy_version": np.__version__,
        "clip_path": str(clip),
        "clip_sha256": _file_sha256(clip),
        "artifacts": {},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("[capture] done")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it expecting the NotImplementedError**

```bash
cd tests/parity/python
uv run python capture_intermediates.py \
  ../fixtures/01_dialogue/clip_16k.wav
```

Expected: `NotImplementedError: Fill in the patched_pipeline body...` — confirms the script wires up correctly before we touch monkey-patches.

- [ ] **Step 3: Implement the raw-embedding monkey-patch**

Replace the `raise NotImplementedError(...)` with the actual patch. The exact target is what Task 3 Step 2 identified; the shape below uses a placeholder name `EMBEDDING_HOOK_TARGET` — substitute the real one.

```python
@contextmanager
def patched_pipeline(buf: CaptureBuffer):
    # Replace these imports with the actual classes Task 3 located.
    from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization

    target_cls = SpeakerDiarization
    # `EMBEDDING_HOOK_TARGET` is the method name Task 3 identified —
    # for example `get_embeddings` or `_get_embeddings`.
    method_name = "EMBEDDING_HOOK_TARGET"
    original = getattr(target_cls, method_name)

    def wrapper(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        # Record a copy so downstream mutation doesn't change the snapshot.
        buf.raw_embeddings = np.asarray(result).copy()
        return result

    setattr(target_cls, method_name, wrapper)
    try:
        yield
    finally:
        setattr(target_cls, method_name, original)
```

- [ ] **Step 4: Run + verify**

```bash
cd tests/parity/python
uv run python capture_intermediates.py \
  ../fixtures/01_dialogue/clip_16k.wav
```

Expected output line includes `[capture] raw_embeddings: (<num_chunks>, <num_slots>, <dim>)` with sensible shapes (e.g. `(45, 3, 256)` — exact numbers depend on the clip and the WeSpeaker dim).

- [ ] **Step 5: Verify the artifact file**

```bash
uv run python -c "
import numpy as np
d = np.load('../fixtures/01_dialogue/raw_embeddings.npz')
print('keys:', list(d.keys()))
print('shape:', d['embeddings'].shape)
print('dtype:', d['embeddings'].dtype)
print('finite:', np.isfinite(d['embeddings']).all())
"
```

Expected: shape printed, dtype `float32` or `float64`, all finite.

- [ ] **Step 6: Commit**

```bash
git add tests/parity/python/capture_intermediates.py
git commit -m "parity: capture raw WeSpeaker embeddings"
```

---

## Task 5: Capture PLDA-projected embeddings

**Files:**
- Modify: `tests/parity/python/capture_intermediates.py`

- [ ] **Step 1: Add the post-`xvec_tf` and post-`plda_tf` hooks**

Per the analysis doc, pyannote applies two stages:

```
xvec_tf(x) = l2norm(lda(l2norm(x - mean1)) - mean2)
plda_tf(x) = whiten(x - mu)
```

Both are pure functions in the VBx utility module. Wrap each to record its output:

```python
@contextmanager
def patched_pipeline(buf: CaptureBuffer):
    from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization
    # The VBx module path is what Task 3 Step 3 located. Replace as needed.
    from pyannote.audio.pipelines.utils import vbx as vbx_mod

    # ── existing raw-embedding patch (unchanged) ──

    original_xvec_tf = vbx_mod.xvec_tf
    original_plda_tf = vbx_mod.plda_tf

    def xvec_wrapper(*args, **kwargs):
        out = original_xvec_tf(*args, **kwargs)
        buf.plda_embeddings_post_xvec = np.asarray(out).copy()
        return out

    def plda_wrapper(*args, **kwargs):
        out = original_plda_tf(*args, **kwargs)
        buf.plda_embeddings_post_plda = np.asarray(out).copy()
        return out

    vbx_mod.xvec_tf = xvec_wrapper
    vbx_mod.plda_tf = plda_wrapper
    try:
        yield
    finally:
        # Restore in reverse order of installation.
        vbx_mod.plda_tf = original_plda_tf
        vbx_mod.xvec_tf = original_xvec_tf
        # ... plus restore the raw-embedding patch from Task 4 ...
```

If Task 3 found different names (e.g. these are inner methods of a class, not module-level functions), adapt — the principle is the same.

- [ ] **Step 2: Add the persistence step in `main()`**

After raw embeddings are written:

```python
np.savez_compressed(
    out_dir / "plda_embeddings.npz",
    post_xvec=buf.plda_embeddings_post_xvec,
    post_plda=buf.plda_embeddings_post_plda,
)
print(
    f"[capture] plda_embeddings post_xvec: "
    f"{buf.plda_embeddings_post_xvec.shape}, "
    f"post_plda: {buf.plda_embeddings_post_plda.shape}"
)
```

- [ ] **Step 3: Run + verify shapes**

```bash
cd tests/parity/python
uv run python capture_intermediates.py \
  ../fixtures/01_dialogue/clip_16k.wav
```

Expected: post_xvec shape ends in 128 (LDA reduces 256 → 128 per the analysis doc); post_plda shape ends in the PLDA dim (also 128).

- [ ] **Step 4: Sanity-check L2 norms**

After `xvec_tf`, vectors are L2-normalized; after `plda_tf`, they generally are not (whitening). Verify:

```bash
uv run python -c "
import numpy as np
d = np.load('../fixtures/01_dialogue/plda_embeddings.npz')
post_xvec = d['post_xvec']
norms = np.linalg.norm(post_xvec.reshape(-1, post_xvec.shape[-1]), axis=1)
assert np.allclose(norms, 1.0, atol=1e-4), f'post-xvec not L2-normed: {norms[:5]}'
print('post_xvec L2 norms ≈ 1.0 ✓')
"
```

- [ ] **Step 5: Commit**

```bash
git add tests/parity/python/capture_intermediates.py
git commit -m "parity: capture PLDA-projected embeddings"
```

---

## Task 6: Export PLDA weight files

**Files:**
- Create: `models/plda/xvec_transform.npz` (gitignored if license forbids redistribution)
- Create: `models/plda/plda.npz` (same)
- Create: `models/plda/SOURCE.md`
- Modify: `tests/parity/python/capture_intermediates.py`

- [ ] **Step 1: Confirm license**

Read the model card for `pyannote/speaker-diarization-community-1` (the page on huggingface.co). Record the license in `models/plda/SOURCE.md` (Step 4) and decide whether the weight files can be committed. If unsure, gitignore them.

- [ ] **Step 2: Add a copy step to the capture script**

Append to the bottom of `main()`:

```python
def _export_plda_weights(repo_root: Path) -> None:
    """Copy PLDA weight files out of the HF cache to models/plda/."""
    from huggingface_hub import snapshot_download

    snap = Path(snapshot_download("pyannote/speaker-diarization-community-1"))
    dst = repo_root / "models" / "plda"
    dst.mkdir(parents=True, exist_ok=True)

    # Filenames are what Task 3 Step 4 listed. Adjust if pyannote ships
    # them under different names in this revision.
    for fname in ("xvec_transform.npz", "plda.npz"):
        src = snap / fname
        if not src.exists():
            # Fall back to a recursive search if pyannote nests the
            # weight files under a sub-directory.
            matches = list(snap.rglob(fname))
            if not matches:
                raise SystemExit(f"could not find {fname} under {snap}")
            src = matches[0]
        target = dst / fname
        target.write_bytes(src.read_bytes())
        print(f"[capture] exported {fname} -> {target}")
```

Call it after the artifacts are written:

```python
repo_root = Path(__file__).resolve().parents[3]
_export_plda_weights(repo_root)
```

- [ ] **Step 3: Run + verify the files exist**

```bash
cd tests/parity/python
uv run python capture_intermediates.py \
  ../fixtures/01_dialogue/clip_16k.wav

ls -lh ../../../models/plda/
```

Expected: two `.npz` files, each in the few-hundred-KB range.

- [ ] **Step 4: Document provenance**

Create `models/plda/SOURCE.md`:

```markdown
# PLDA weights

These two `.npz` files are copied from the HuggingFace snapshot of
[`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1).

- `xvec_transform.npz`: stage-1 weights consumed by `xvec_tf`
  (mean1, LDA matrix, mean2). Reduces 256-d WeSpeaker embeddings to
  128-d.
- `plda.npz`: stage-2 weights consumed by `plda_tf`
  (mu, whitening matrix). Whitens 128-d vectors so within-speaker
  variance shrinks and between-speaker variance is preserved.

**License:** <fill in from the model card — verified at <date>>.

**Snapshot commit:** <hash from `huggingface_hub.snapshot_download`>.

**Refresh:** rerun `tests/parity/python/capture_intermediates.py` —
it re-fetches and overwrites these files.
```

- [ ] **Step 5: Either commit the .npz files or gitignore them**

If license permits redistribution:

```bash
git add models/plda/
git commit -m "parity: export PLDA weight files from pyannote community-1"
```

If not:

```bash
echo "models/plda/*.npz" >> .gitignore
git add .gitignore models/plda/SOURCE.md
git commit -m "parity: document PLDA weights provenance (gitignored)"
```

---

## Task 7: Capture AHC init + VBx posteriors + final labels

**Files:**
- Modify: `tests/parity/python/capture_intermediates.py`

- [ ] **Step 1: Add the AHC init hook**

Wrap the AHC entry point Task 3 located. Pseudo-code (substitute real symbols):

```python
# Inside patched_pipeline:
from pyannote.audio.pipelines.utils.vbx import ahc_init  # actual name from Task 3

original_ahc = ahc_init

def ahc_wrapper(*args, **kwargs):
    out = original_ahc(*args, **kwargs)
    buf.ahc_init_labels = np.asarray(out).copy()
    return out

# install + restore as in Task 4
```

- [ ] **Step 2: Add the VBx EM hook**

VBx has a per-iteration loop. The cleanest interception point is the iterator/loop body — wrap the function that returns `(q, sp, tr)` per iteration. Per the analysis doc, look for the function in `utils/vbx.py:140`. If pyannote exposes only the final result, fall back to wrapping the per-iteration update function:

```python
# Pseudo: depends on Task 3 findings.
original_vbx_iter = vbx_mod.vbx_iteration  # name from Task 3

def vbx_iter_wrapper(*args, **kwargs):
    q, sp, tr = original_vbx_iter(*args, **kwargs)
    buf.vbx_posteriors_per_iter.append(np.asarray(q).copy())
    buf.vbx_sp_per_iter.append(np.asarray(sp).copy())
    return q, sp, tr
```

- [ ] **Step 3: Add the final-labels hook**

The constrained Hungarian step produces per-(chunk, slot) labels. Hook the function or method that returns them.

- [ ] **Step 4: Persist the new artifacts**

In `main()`, after PLDA artifacts are written:

```python
np.save(out_dir / "ahc_init_labels.npy", buf.ahc_init_labels)
np.savez_compressed(
    out_dir / "vbx_posteriors.npz",
    q_per_iter=np.stack(buf.vbx_posteriors_per_iter, axis=0),
    sp_per_iter=np.stack(buf.vbx_sp_per_iter, axis=0),
)
np.save(out_dir / "final_labels.npy", buf.final_labels)
print(
    f"[capture] ahc_init_labels: {buf.ahc_init_labels.shape}, "
    f"vbx_iters: {len(buf.vbx_posteriors_per_iter)}, "
    f"final_labels: {buf.final_labels.shape}"
)
```

- [ ] **Step 5: Run + verify**

```bash
cd tests/parity/python
uv run python capture_intermediates.py \
  ../fixtures/01_dialogue/clip_16k.wav
```

Expected: `vbx_iters: 20` (matches the analysis doc's "20 iterations"). `ahc_init_labels` is a 1-D integer array; `final_labels` is the same length.

- [ ] **Step 6: Sanity-check label cardinality**

```bash
uv run python -c "
import numpy as np
ahc = np.load('../fixtures/01_dialogue/ahc_init_labels.npy')
fin = np.load('../fixtures/01_dialogue/final_labels.npy')
print('AHC unique:', sorted(set(ahc.flatten().tolist())))
print('Final unique:', sorted(set(fin.flatten().tolist())))
assert ahc.shape == fin.shape, (ahc.shape, fin.shape)
"
```

Expected: 2 unique labels for our 2-speaker clip (or 3 if pyannote opened a phantom cluster — that itself is a useful diagnostic).

- [ ] **Step 7: Commit**

```bash
git add tests/parity/python/capture_intermediates.py
git commit -m "parity: capture AHC init, VBx posteriors, final labels"
```

---

## Task 8: Manifest with checksums

**Files:**
- Modify: `tests/parity/python/capture_intermediates.py`

- [ ] **Step 1: Compute checksums for every artifact**

Replace the placeholder `manifest` build with:

```python
artifact_files = [
    "raw_embeddings.npz",
    "plda_embeddings.npz",
    "ahc_init_labels.npy",
    "vbx_posteriors.npz",
    "final_labels.npy",
    "reference.rttm",
]
manifest = {
    "pyannote_audio_version": pyannote.audio.__version__,
    "numpy_version": np.__version__,
    "clip_path": str(clip),
    "clip_sha256": _file_sha256(clip),
    "artifacts": {
        f: _file_sha256(out_dir / f) for f in artifact_files
    },
}
(out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
```

- [ ] **Step 2: Run + inspect the manifest**

```bash
cd tests/parity/python
uv run python capture_intermediates.py \
  ../fixtures/01_dialogue/clip_16k.wav
cat ../fixtures/01_dialogue/manifest.json
```

Expected: JSON with one sha256 per artifact and the pyannote/numpy versions. Every checksum is a 64-hex-char string.

- [ ] **Step 3: Commit the manifest + artifacts**

If the artifacts are small enough to commit (a few MB total):

```bash
git add tests/parity/fixtures/01_dialogue/{raw_embeddings.npz,plda_embeddings.npz,ahc_init_labels.npy,vbx_posteriors.npz,final_labels.npy,reference.rttm,manifest.json}
git commit -m "parity: snapshot pyannote intermediates for 01_dialogue"
```

If the `.npz` files are too large for git, gitignore them and commit only `manifest.json` + a regen script note in the README.

---

## Task 9: Reproducibility-check script

**Files:**
- Create: `tests/parity/python/verify_capture.py`

- [ ] **Step 1: Write the verifier**

Create `tests/parity/python/verify_capture.py`:

```python
"""Re-run capture_intermediates and assert byte-identical outputs.

A green run proves the snapshot is deterministic — same pyannote
version + same clip + same hardware should always produce the same
artifacts. Phase-1 onward will rely on that determinism: when a Rust
port produces output that doesn't match, the failure is the port, not
flaky pyannote.

Usage:
  uv run python verify_capture.py <clip.wav>
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python verify_capture.py <clip.wav>")
    clip = Path(sys.argv[1]).resolve()
    snapshot_dir = clip.parent
    manifest_path = snapshot_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(
            f"no manifest at {manifest_path}; run capture_intermediates.py first"
        )
    expected = json.loads(manifest_path.read_text())["artifacts"]

    # Stage the existing artifacts somewhere safe so a failed re-run
    # doesn't destroy our snapshot.
    backup = snapshot_dir.parent / f".{snapshot_dir.name}.backup"
    if backup.exists():
        shutil.rmtree(backup)
    shutil.copytree(snapshot_dir, backup)

    print(f"[verify] backup written to {backup}")
    print("[verify] re-running capture...")
    subprocess.run(
        [sys.executable, str(Path(__file__).parent / "capture_intermediates.py"), str(clip)],
        check=True,
    )

    mismatches: list[str] = []
    for name, expected_hash in expected.items():
        actual = _sha256(snapshot_dir / name)
        if actual != expected_hash:
            mismatches.append(f"{name}: {actual} != {expected_hash}")
    if mismatches:
        print("[verify] MISMATCHES:")
        for m in mismatches:
            print(f"  - {m}")
        sys.exit(1)
    print("[verify] all artifacts match — snapshot is reproducible")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

```bash
cd tests/parity/python
uv run python verify_capture.py \
  ../fixtures/01_dialogue/clip_16k.wav
```

Expected: `[verify] all artifacts match — snapshot is reproducible`. If any artifact mismatches, we have nondeterminism — investigate before proceeding (likely a non-fixed RNG seed somewhere; pyannote takes a `seed` kwarg).

- [ ] **Step 3: Commit**

```bash
git add tests/parity/python/verify_capture.py
git commit -m "parity: verify_capture.py — assert byte-identical re-runs"
```

---

## Task 10: Wire the canonical-clip flow into `run.sh`; document

**Files:**
- Modify: `tests/parity/run.sh`
- Modify: `tests/parity/README.md`

- [ ] **Step 1: Update `run.sh` to invoke capture if fixtures missing**

Replace the body of `tests/parity/run.sh` with:

```bash
#!/usr/bin/env bash
# Pyannote parity harness.
#
# Requires:
# - models/segmentation-3.0.onnx and models/wespeaker_resnet34_lm.onnx
# - models/plda/xvec_transform.npz and models/plda/plda.npz (Phase 0)
# - uv
# - the clip path; defaults to the canonical 2-speaker fixture
#
# Behavior:
# - If <fixture-dir>/manifest.json is missing, runs capture first.
# - Then runs dia and pyannote, computes DER.

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

- [ ] **Step 2: Smoke-run the updated harness**

```bash
cd /Users/user/Develop/findit-studio/dia
bash tests/parity/run.sh
```

Expected: capture is skipped (fixtures already present from earlier tasks); dia runs; DER printed.

- [ ] **Step 3: Update `tests/parity/README.md`**

Append:

```markdown
## Phase 0: pyannote intermediates snapshot

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

This overwrites every artifact under `01_dialogue/`. The PLDA weight
files (`models/plda/xvec_transform.npz` + `plda.npz`) are also
re-exported.

### Verifying determinism

```bash
cd tests/parity/python
uv run python verify_capture.py \
  ../fixtures/01_dialogue/clip_16k.wav
```

A green run is required before opening a Phase-1+ Rust PR — every Rust
port parity-checks against the snapshot, so the snapshot must be
trustable.

### Why we pin pyannote

`pyproject.toml` pins `pyannote.audio` to an exact version. If
upstream pyannote ships a behavior change, `verify_capture.py` will
fail and force a deliberate snapshot refresh + version bump rather
than letting the change leak silently into Rust-port reviews.
```

- [ ] **Step 4: Commit**

```bash
git add tests/parity/run.sh tests/parity/README.md
git commit -m "parity: wire capture into run.sh, document Phase-0 flow"
```

---

## Task 11: PR + handoff to Phase 1

**Files:**
- None (workflow task).

- [ ] **Step 1: Push the branch**

```bash
git push -u origin feat/parity
```

- [ ] **Step 2: Open a PR**

Title: `parity: Phase-0 pyannote intermediates capture`

Body (paste into PR description):

```markdown
## Summary
Implements Phase 0 of the Option A pyannote-parity plan
(`docs/superpowers/plans/2026-04-28-dia-pyannote-parity-phase-0-capture.md`).

This PR adds no Rust code. It produces a deterministic snapshot of
`pyannote/speaker-diarization-community-1`'s intermediate artifacts on
the canonical 2-speaker clip so subsequent Rust ports (PLDA → VBx →
constrained Hungarian → centroid AHC) can be parity-validated commit
by commit.

## What's captured
- raw WeSpeaker embeddings (pre-PLDA)
- PLDA-projected embeddings (post-`xvec_tf`, post-`plda_tf`)
- AHC initialization labels
- VBx posteriors per iteration (q, sp)
- post-Hungarian final per-(chunk, slot) labels
- final RTTM

Plus the PLDA weight files (`xvec_transform.npz`, `plda.npz`) exported
from the HF cache to `models/plda/`.

## Verification
- `tests/parity/python/verify_capture.py` re-runs the capture and
  asserts byte-identical outputs (deterministic).
- `tests/parity/run.sh` continues to work as before; it now defaults
  to the canonical fixture and reuses the captured RTTM as the
  pyannote reference (no need to rerun pyannote per parity check).

## Test plan
- [ ] `tests/parity/run.sh` runs end-to-end on a clean checkout
- [ ] `verify_capture.py` is green
- [ ] All `.npz` files under `tests/parity/fixtures/01_dialogue/`
  open in numpy with sensible shapes
- [ ] `manifest.json` lists every artifact with a sha256

## Out of scope
- Phases 1–4 (Rust ports). Each gets its own plan + PR.
```

- [ ] **Step 3: Note follow-on plans**

In the PR body or a follow-up issue, list the four phases with their plan-document filenames (TBD when each plan is written):

- Phase 1: `2026-XX-XX-dia-plda-rust-port.md`
- Phase 2: `2026-XX-XX-dia-vbx-rust-port.md`
- Phase 3: `2026-XX-XX-dia-constrained-hungarian.md`
- Phase 4: `2026-XX-XX-dia-centroid-ahc.md`

Each follow-on plan will (a) point at the Phase-0 snapshot for parity, (b) define per-PR acceptance criteria using DER deltas measured against the snapshot.

---

## Self-review

**1. Spec coverage.** Phase 0 of the analysis doc says: capture pyannote intermediates so subsequent Rust ports can be parity-validated. Tasks 1–10 cover every named artifact (raw embeddings, post-xvec, post-plda, AHC init, VBx posteriors, final labels, PLDA weight files). Task 9 covers reproducibility. Task 11 is the merge handoff. Nothing in the analysis doc's Phase-0 description is uncovered.

**2. Placeholder scan.** Several tasks (3, 4, 5, 7) say "the exact symbol Task 3 located" or "substitute the real one." That's an explicit, scoped research dependency, not a placeholder for unknown logic — Task 3 produces the symbol names and the README table that Tasks 4–7 reference. The principle (monkey-patch a function, record its output, restore in `finally`) is fully specified; only the binding to a particular pyannote version is research-dependent.

**3. Type consistency.** `CaptureBuffer` defines `raw_embeddings`, `plda_embeddings_post_xvec`, `plda_embeddings_post_plda`, `ahc_init_labels`, `vbx_posteriors_per_iter`, `vbx_sp_per_iter`, `final_labels`. Every persistence step in Tasks 4–8 reads from one of these names, and the verifier in Task 9 reads the manifest written in Task 8. Consistent.

**4. Risk areas explicitly called out:**
- License (Task 6 Step 1) — could block redistribution; documented gitignore fallback.
- Snapshot size (Task 1 Step 5, Task 8 Step 3) — `clip_16k.wav` is gitignored; `.npz` files committed if license + size allow, otherwise gitignored with a regen note.
- Pyannote nondeterminism (Task 9 Step 2) — verifier failure is handled with explicit "investigate seed" guidance.
- Pyannote internal API churn (Task 2 Step 2, Task 10 Step 3) — the `==` pin makes drift fail loudly.

---

## Plan complete

Subsequent phases (PLDA Rust port, VBx Rust port, constrained Hungarian, centroid AHC) will each get their own plan documents authored once Phase 0 lands and the captured artifacts' actual shapes / dtypes / numerical ranges are known. Writing those plans now would force the same kind of placeholder content this plan deliberately avoids.
