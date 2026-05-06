#!/usr/bin/env bash
# Generate / fetch test fixtures for `dia::Diarizer` integration tests.
#
# Currently produces a synthetic 30-second tone wav. Replace with a
# real multi-speaker clip (and update the SHA below) for meaningful
# diarization-quality tests.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIX_DIR="$SCRIPT_DIR/../tests/fixtures"
mkdir -p "$FIX_DIR"

DEST="$FIX_DIR/diarize_test_30s.wav"
if [ -f "$DEST" ]; then
  echo "Fixture already present at $DEST."
  exit 0
fi

if ! command -v ffmpeg > /dev/null; then
  echo "Error: ffmpeg required to generate the synthetic fixture." >&2
  echo "Install via 'brew install ffmpeg' (macOS) or your distro's package manager." >&2
  exit 1
fi

echo "Generating synthetic 30 s tone wav at $DEST ..."
ffmpeg -loglevel error \
  -f lavfi -i "sine=frequency=440:duration=10:sample_rate=16000" \
  -f lavfi -i "sine=frequency=660:duration=10:sample_rate=16000" \
  -f lavfi -i "sine=frequency=880:duration=10:sample_rate=16000" \
  -filter_complex "[0:a][1:a][2:a]concat=n=3:v=0:a=1[out]" \
  -map "[out]" -ac 1 -ar 16000 -sample_fmt s16 "$DEST" -y

echo "Saved to $DEST."
