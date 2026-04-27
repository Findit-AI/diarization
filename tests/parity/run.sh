#!/usr/bin/env bash
# Pyannote parity harness: runs both reference + dia, computes DER.
# Requires:
# - models/segmentation-3.0.onnx and models/wespeaker_resnet34_lm.onnx
# - uv (or python3 + venv)
# - the clip path provided as $1, relative to dia crate root

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/../.."

CLIP="${1:?usage: run.sh <clip.wav>}"
ABS_CLIP="$(cd "$ROOT" && realpath "$CLIP")"

# Python reference.
cd "$SCRIPT_DIR/python"
if [ ! -d .venv ]; then
  uv venv
fi
uv pip install -e .
uv run python reference.py "$ABS_CLIP" > "$SCRIPT_DIR/ref.rttm"

# Rust dia.
cd "$ROOT"
cargo run --release --manifest-path tests/parity/Cargo.toml -- "$CLIP" \
  > "$SCRIPT_DIR/hyp.rttm"

# Score.
cd "$SCRIPT_DIR/python"
uv run python score.py "$SCRIPT_DIR/ref.rttm" "$SCRIPT_DIR/hyp.rttm"
