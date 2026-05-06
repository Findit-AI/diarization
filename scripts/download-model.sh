#!/usr/bin/env bash
# Download the pyannote/segmentation-3.0 ONNX model into `models/`.
# Mirror sources are listed in priority order; the script tries each until
# one succeeds. Re-run is idempotent.

set -euo pipefail

DEST_DIR="$(cd "$(dirname "$0")/.." && pwd)/models"
DEST_FILE="$DEST_DIR/segmentation-3.0.onnx"

mkdir -p "$DEST_DIR"

if [[ -f "$DEST_FILE" ]]; then
  echo "model already present: $DEST_FILE"
  exit 0
fi

URLS=(
  "https://huggingface.co/pyannote/segmentation-3.0/resolve/main/pytorch_model.onnx"
  "https://huggingface.co/onnx-community/pyannote-segmentation-3.0/resolve/main/onnx/model.onnx"
)

for url in "${URLS[@]}"; do
  echo "trying $url"
  if curl -fL --retry 3 --retry-delay 2 -o "$DEST_FILE.tmp" "$url"; then
    mv "$DEST_FILE.tmp" "$DEST_FILE"
    echo "downloaded to $DEST_FILE"
    exit 0
  fi
done

echo "failed to download model from any mirror" >&2
rm -f "$DEST_FILE.tmp"
exit 1
