#!/usr/bin/env bash
# Download the WeSpeaker ResNet34-LM ONNX model used by `dia::embed`.
#
# Spec §3 deferred items: dia v0.1.0 does NOT bundle model files. Run
# this script (or set DIA_EMBED_MODEL_PATH) before invoking the gated
# integration tests:
#
#   ./scripts/download-embed-model.sh
#   cargo test --features ort -- --ignored
#
# Source: onnx-community/wespeaker-voxceleb-resnet34-LM on Hugging Face.
# Variant: model.onnx (FP32, 26.5 MB). The FP16 / quantized variants
# diverge from the pyannote reference and are deferred to v0.2.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/../models"
mkdir -p "$MODELS_DIR"

URL="https://huggingface.co/onnx-community/wespeaker-voxceleb-resnet34-LM/resolve/main/onnx/model.onnx"
DEST="$MODELS_DIR/wespeaker_resnet34_lm.onnx"

# SHA-256 of the FP32 model.onnx as of 2026-04-27. Update if the upstream
# repo re-publishes the model — a mismatch indicates a content drift that
# could silently invalidate the byte-determinism / pyannote-parity gates.
EXPECTED_SHA256="3955447b0499dc9e0a4541a895df08b03c69098eba4e56c02b5603e9f7f4fcbb"

if [ -f "$DEST" ]; then
  ACTUAL_SHA256="$(shasum -a 256 "$DEST" | awk '{print $1}')"
  if [ "$ACTUAL_SHA256" = "$EXPECTED_SHA256" ]; then
    echo "Model already present at $DEST (sha256 verified)."
    exit 0
  fi
  echo "Warning: existing $DEST does not match expected sha256."
  echo "  expected: $EXPECTED_SHA256"
  echo "  actual:   $ACTUAL_SHA256"
  echo "Re-downloading..."
fi

echo "Downloading WeSpeaker ResNet34-LM (26.5 MB) from $URL ..."
curl --fail --show-error --location --output "$DEST" --progress-bar "$URL"

ACTUAL_SHA256="$(shasum -a 256 "$DEST" | awk '{print $1}')"
if [ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]; then
  echo "Error: downloaded file sha256 mismatch."
  echo "  expected: $EXPECTED_SHA256"
  echo "  actual:   $ACTUAL_SHA256"
  echo "The Hugging Face repo may have re-published; verify upstream and"
  echo "update EXPECTED_SHA256 in this script."
  exit 1
fi

echo "Saved to $DEST (sha256 verified)."
