"""Re-run capture_intermediates and assert byte-identical outputs.

A green run proves the snapshot is deterministic: same pyannote
version + same clip + same hardware should always produce the same
artifacts. Phase 1+ (Rust ports) relies on that determinism — when a
Rust port produces output that doesn't match the snapshot, the failure
is the port, not flaky pyannote.

Usage:
  uv run python verify_capture.py <clip.wav>
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: verify_capture.py <clip.wav>")
    clip = Path(sys.argv[1]).resolve()
    snapshot_dir = clip.parent
    manifest_path = snapshot_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(
            f"no manifest at {manifest_path}; run capture_intermediates.py first"
        )
    expected = json.loads(manifest_path.read_text())["artifacts"]

    # Stage existing artifacts to a sibling backup so a failed re-run
    # doesn't destroy the snapshot.
    backup = snapshot_dir.parent / f".{snapshot_dir.name}.backup"
    if backup.exists():
        shutil.rmtree(backup)
    shutil.copytree(snapshot_dir, backup)
    print(f"[verify] backup written to {backup}")

    print("[verify] re-running capture...")
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parent / "capture_intermediates.py"),
            str(clip),
        ],
        check=True,
    )

    mismatches: list[str] = []
    for name, expected_hash in expected.items():
        actual = _sha256(snapshot_dir / name)
        if actual != expected_hash:
            mismatches.append(f"  {name}: {actual} != {expected_hash}")

    if mismatches:
        print("[verify] MISMATCHES:")
        for m in mismatches:
            print(m)
        print(f"[verify] backup preserved at {backup}")
        sys.exit(1)

    # Clean up backup on success.
    shutil.rmtree(backup)
    print("[verify] all artifacts match — snapshot is reproducible")


if __name__ == "__main__":
    main()
