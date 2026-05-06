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
