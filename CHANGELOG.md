# 0.1.0 (unreleased)

FEATURES

- `dia::segment` module: Sans-I/O speaker segmentation backed by
  pyannote/segmentation-3.0 ONNX. Two layers: a pure state-machine `Segmenter`
  with no `ort` dependency, plus a default-on `ort` feature exposing a
  `SegmentModel` ONNX wrapper and `process_samples` / `finish_stream`
  streaming helpers.
