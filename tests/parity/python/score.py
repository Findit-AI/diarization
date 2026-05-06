"""Compute Diarization Error Rate (DER) between two RTTM files.

Usage: uv run python score.py <ref.rttm> <hyp.rttm>

Exit code 0 iff DER <= 0.10 (rev-8 T3-I relaxed threshold).
"""

import sys

from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate


def load_rttm(path: str) -> Annotation:
    annotation = Annotation()
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0] != "SPEAKER":
                continue
            start = float(parts[3])
            duration = float(parts[4])
            speaker = parts[7]
            annotation[Segment(start, start + duration)] = speaker
    return annotation


if len(sys.argv) != 3:
    raise SystemExit("usage: python score.py <ref.rttm> <hyp.rttm>")

ref = load_rttm(sys.argv[1])
hyp = load_rttm(sys.argv[2])

metric = DiarizationErrorRate(collar=0.5, skip_overlap=False)
der = metric(ref, hyp)
print(f"DER = {der:.4f}")
sys.exit(0 if der <= 0.10 else 1)
