#!/usr/bin/env bash
# Pyannote parity harness.
#
# Requires:
# - models/segmentation-3.0.onnx and models/wespeaker_resnet34_lm.onnx
# - models/plda/xvec_transform.npz and models/plda/plda.npz (Phase 0)
# - uv (https://docs.astral.sh/uv/)
# - the clip path; defaults to the canonical 2-speaker fixture
#
# Behavior:
# - If <fixture-dir>/manifest.json is missing, runs Phase-0 capture first.
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
uv pip install -e . > /dev/null

if [ ! -f "$MANIFEST" ]; then
  echo "[run.sh] no manifest at $MANIFEST; running Phase-0 capture..."
  uv run python capture_intermediates.py "$ABS_CLIP"
else
  echo "[run.sh] reusing existing snapshot at $SNAPSHOT_DIR"
fi

# Reuse the captured RTTM as the reference (no need to rerun pyannote).
REF_RTTM="$SNAPSHOT_DIR/reference.rttm"

cd "$ROOT"
cargo run --release --manifest-path tests/parity/Cargo.toml -- "$CLIP" \
  > "$SCRIPT_DIR/hyp.rttm"

cd "$SCRIPT_DIR/python"
uv run python score.py "$REF_RTTM" "$SCRIPT_DIR/hyp.rttm"
