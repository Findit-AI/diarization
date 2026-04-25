//! ONNX Runtime wrapper for pyannote/segmentation-3.0 plus Layer-2
//! streaming convenience methods on [`Segmenter`].

use std::path::Path;

use ort::{session::Session as OrtSession, value::TensorRef};

use crate::segment::{
  error::Error,
  options::{FRAMES_PER_WINDOW, POWERSET_CLASSES, WINDOW_SAMPLES},
  segmenter::Segmenter,
  types::{Action, Event},
};

/// Thin ort wrapper for one segmentation model session.
///
/// Owns one `ort::Session` plus reusable input/output scratch. `Send` but
/// not `Sync` — use one per worker thread. Mirrors `silero::Session`.
pub struct SegmentModel {
  inner: OrtSession,
  input_scratch: Vec<f32>,
}

impl SegmentModel {
  /// Load the model from disk.
  pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
    let path = path.as_ref();
    let session = OrtSession::builder()?
      .commit_from_file(path)
      .map_err(|source| Error::LoadModel {
        path: path.to_path_buf(),
        source,
      })?;
    Ok(Self {
      inner: session,
      input_scratch: Vec::with_capacity(WINDOW_SAMPLES as usize),
    })
  }

  /// Load the model from an in-memory ONNX byte buffer.
  pub fn from_memory(bytes: &[u8]) -> Result<Self, Error> {
    let session = OrtSession::builder()?.commit_from_memory(bytes)?;
    Ok(Self {
      inner: session,
      input_scratch: Vec::with_capacity(WINDOW_SAMPLES as usize),
    })
  }

  /// Run inference on one 160 000-sample window. Returns the flattened
  /// `[FRAMES_PER_WINDOW * POWERSET_CLASSES] = [4123]` logits.
  ///
  /// Exposed for advanced callers who want to combine Layer 1's state
  /// machine with their own batching or scheduling.
  pub fn infer(&mut self, samples: &[f32]) -> Result<Vec<f32>, Error> {
    debug_assert_eq!(samples.len(), WINDOW_SAMPLES as usize);

    self.input_scratch.clear();
    self.input_scratch.extend_from_slice(samples);

    // pyannote/segmentation-3.0 expects input shape [batch, channels, samples]
    // = [1, 1, 160000] f32.
    let input_name = self
      .inner
      .inputs()
      .first()
      .map(|i| i.name().to_string())
      .ok_or(Error::InvalidOptions("model has no inputs"))?;
    let output_name = self
      .inner
      .outputs()
      .first()
      .map(|o| o.name().to_string())
      .ok_or(Error::InvalidOptions("model has no outputs"))?;

    let outputs = self.inner.run(ort::inputs![
      input_name.as_str() => TensorRef::from_array_view(
        ([1usize, 1usize, WINDOW_SAMPLES as usize], self.input_scratch.as_slice()),
      )?,
    ])?;

    let (_shape, data) = outputs[output_name.as_str()].try_extract_tensor::<f32>()?;
    let expected = FRAMES_PER_WINDOW * POWERSET_CLASSES;
    if data.len() != expected {
      return Err(Error::InferenceShapeMismatch {
        expected,
        got: data.len(),
      });
    }
    Ok(data.to_vec())
  }
}

impl Segmenter {
  /// Push samples and drive the state machine to a quiescent state by
  /// fulfilling each `NeedsInference` via `model.infer`. `emit` is called
  /// for every emitted [`Event`].
  ///
  /// This is the streaming entry point that mirrors
  /// `silero::SpeechSegmenter::process_samples`.
  #[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
  pub fn process_samples<F>(
    &mut self,
    model: &mut SegmentModel,
    samples: &[f32],
    mut emit: F,
  ) -> Result<(), Error>
  where
    F: FnMut(Event),
  {
    self.push_samples(samples);
    self.drain(model, &mut emit)
  }

  /// Equivalent to `finish` followed by draining all remaining actions
  /// (running inference for any unprocessed window).
  #[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
  pub fn finish_stream<F>(&mut self, model: &mut SegmentModel, mut emit: F) -> Result<(), Error>
  where
    F: FnMut(Event),
  {
    self.finish();
    self.drain(model, &mut emit)
  }

  fn drain<F>(&mut self, model: &mut SegmentModel, emit: &mut F) -> Result<(), Error>
  where
    F: FnMut(Event),
  {
    while let Some(action) = self.poll() {
      match action {
        Action::NeedsInference { id, samples } => {
          let scores = model.infer(&samples)?;
          self.push_inference(id, &scores)?;
        }
        Action::Activity(a) => emit(Event::Activity(a)),
        Action::VoiceSpan(r) => emit(Event::VoiceSpan(r)),
      }
    }
    Ok(())
  }
}
