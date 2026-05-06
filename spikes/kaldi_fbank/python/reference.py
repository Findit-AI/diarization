"""Compute 80-mel kaldi fbank via torchaudio on the same clip; dump CSV.

Counterpart to `src/main.rs`. Both sides MUST share identical fbank
options (defaults inherited from torchaudio, with `num_mel_bins=80`,
`dither=0.0`, `window_type="hamming"` overridden on each side) so the
only remaining difference is the implementation under test.
"""
import csv
import sys

import soundfile as sf
import torch
import torchaudio


def main() -> None:
    waveform, sr = sf.read("../test_clip.wav", dtype="float32")
    assert sr == 16_000, f"expected 16 kHz, got {sr}"
    assert waveform.ndim == 1, f"expected mono, got shape {waveform.shape}"

    # soundfile's dtype="float32" returns samples normalized to [-1.0, 1.0).
    # torchaudio.compliance.kaldi.fbank expects amplitude in the int16 range
    # (Kaldi convention), so undo the normalization.
    wf = torch.from_numpy(waveform).unsqueeze(0) * 32_768.0  # (1, num_samples)

    features = torchaudio.compliance.kaldi.fbank(
        wf,
        num_mel_bins=80,
        frame_length=25.0,
        frame_shift=10.0,
        dither=0.0,
        window_type="hamming",
        sample_frequency=16_000,
    )

    w = csv.writer(sys.stdout)
    w.writerow(["frame"] + [f"mel{i}" for i in range(80)])
    for i, row in enumerate(features.numpy()):
        w.writerow([i] + [f"{x}" for x in row])


if __name__ == "__main__":
    main()
