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
