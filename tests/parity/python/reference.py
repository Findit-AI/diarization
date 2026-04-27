"""Run pyannote.audio.SpeakerDiarization on a clip; dump RTTM to stdout.

Usage: uv run python reference.py <clip.wav>
"""

import sys
from pathlib import Path

from pyannote.audio import Pipeline

if len(sys.argv) != 2:
    raise SystemExit("usage: python reference.py <clip.wav>")
clip = sys.argv[1]
uri = Path(clip).stem

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1")
output = pipeline(clip)
diarization = (
    output.speaker_diarization if hasattr(output, "speaker_diarization") else output
)

for turn, _, speaker in diarization.itertracks(yield_label=True):
    start = turn.start
    duration = turn.duration
    print(
        f"SPEAKER {uri} 1 {start:.3f} {duration:.3f} <NA> <NA> {speaker} <NA> <NA>"
    )
