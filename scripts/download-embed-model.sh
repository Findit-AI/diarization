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
# Source: FinDIT-Studio/dia-models on Hugging Face — the canonical
# bundle of all dia model artifacts (segmentation, WeSpeaker embedding
# in three forms, PLDA weights, with attribution preserved).
#
# Variant fetched: `wespeaker_resnet34_lm.onnx` (FP32, single-file
# packed form, ~25.5 MiB). All weights are inlined; no `.onnx.data`
# sidecar is needed. This is the form that loads cleanly on every
# ORT execution provider — including CoreML, whose optimizer fails
# to relocate external initializers in the alternative external-data
# layout. FP16 / quantized variants are deferred (they perturb the
# pyannote-parity numerics and need separate validation).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/../models"
mkdir -p "$MODELS_DIR"

# Pin a specific HF commit so the download is reproducible. The
# README quickstart pins the same revision + SHA-256 inline; keep
# both in sync when bumping.
REV="8ecadc1f8bc31dac506e46d1f5aa7c25e00277ec"
URL="https://huggingface.co/FinDIT-Studio/dia-models/resolve/$REV/wespeaker_resnet34_lm.onnx"
DEST="$MODELS_DIR/wespeaker_resnet34_lm.onnx"

# SHA-256 of the canonical packed FP32 model (single-file, no
# external data) at the pinned `$REV`. Update both if the upstream
# HF repo re-publishes — a mismatch indicates content drift that
# could silently invalidate byte-determinism / pyannote-parity gates.
EXPECTED_SHA256="4c15c6be4235318d092c9d347e00c68ba476136d6172f675f76ad6b0c2661f01"

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
