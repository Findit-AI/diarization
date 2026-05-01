# Bundled model files

`dia` ships two pyannote model artifacts compiled into the binary via
`include_bytes!`. Downstream redistributors must reproduce the
attributions in `NOTICE` (CC-BY-4.0 for PLDA, MIT for segmentation).

## `segmentation-3.0.onnx`

The 16 kHz 7-class powerset speaker-segmentation network from
`pyannote/segmentation-3.0`. Embedded by
`SegmentModel::bundled()` when the `bundled-segmentation` cargo
feature is enabled (default-on). Off-switch: callers who BYO a
fine-tuned variant turn off `default-features` and use
`SegmentModel::from_file` / `from_memory`.

- **License:** MIT (CNRS / Hervé Bredin)
- **Source:** <https://huggingface.co/pyannote/segmentation-3.0>
- **Original layout:** `pytorch_model.onnx` in the HF repo (renamed
  on download).
- **SHA-256:** `057ee564753071c0b09b5b611648b50ac188d50846bff5f01e9f7bbf1591ea25`
- **Size:** 5 986 908 bytes (~5.99 MiB), gzip ~5.28 MiB.

`scripts/download-model.sh` mirrors the upstream snapshot for callers
who disable bundling. Refreshing the bundled file: re-run the script
into `models/segmentation-3.0.onnx`, update the SHA-256 above, and
re-run `cargo test`.

## `plda/`

PLDA whitening weights from
`pyannote/speaker-diarization-community-1`. Embedded by
`crate::plda::loader`. See `models/plda/SOURCE.md` for the full
provenance + refresh procedure.

- **License:** CC-BY-4.0 (BUT Speech@FIT; pyannote integration by
  Jiangyu Han and Petr Pálka)

## NOT bundled — `wespeaker_resnet34_lm.onnx` (+ `.onnx.data`)

The 27 MB WeSpeaker ResNet34-LM export exceeds the crates.io 10 MB
crate-tarball limit (the float32 weights are mostly incompressible —
gzip recovers ~7 %). Callers fetch it via
`scripts/download-embed-model.sh` (or set `DIA_EMBED_MODEL_PATH`).
The expected SHA-256 lives in that script.

The `.pt` TorchScript variant is a separate dev-only file used by the
optional `tch` feature and is also out-of-tree.
