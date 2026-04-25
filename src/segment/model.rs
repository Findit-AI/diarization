//! ONNX Runtime wrapper for pyannote/segmentation-3.0 plus Layer-2
//! streaming convenience methods on [`Segmenter`].

use std::path::Path;

use ort::{
  execution_providers::ExecutionProviderDispatch,
  session::{
    Session as OrtSession,
    builder::{GraphOptimizationLevel, SessionBuilder},
  },
  value::TensorRef,
};

use crate::segment::{
  error::Error,
  options::{FRAMES_PER_WINDOW, POWERSET_CLASSES, WINDOW_SAMPLES},
  segmenter::Segmenter,
  types::{Action, Event},
};

/// Builder for [`SegmentModel`] runtime configuration.
///
/// Default: ort defaults for optimization level and threading, no execution
/// providers configured beyond ort's default search. Mutate via the `with_*`
/// builders.
#[derive(Default)]
pub struct SegmentModelOptions {
  optimization_level: Option<GraphOptimizationLevel>,
  providers: Vec<ExecutionProviderDispatch>,
  intra_op_num_threads: Option<usize>,
  inter_op_num_threads: Option<usize>,
}

impl SegmentModelOptions {
  /// Construct with all-default options.
  pub fn new() -> Self {
    Self::default()
  }

  /// Override the graph optimization level.
  pub fn with_optimization_level(mut self, level: GraphOptimizationLevel) -> Self {
    self.optimization_level = Some(level);
    self
  }

  /// Configure execution providers in priority order. Default: ort's
  /// default execution-provider selection (typically CPU).
  ///
  /// **Caveat:** CoreML on macOS is known to degrade pyannote/segmentation-3.0
  /// numerics (see the design spec). Do not enable without measuring.
  pub fn with_providers(mut self, providers: Vec<ExecutionProviderDispatch>) -> Self {
    self.providers = providers;
    self
  }

  /// Override `intra_op_num_threads`. Set to `1` for bit-exact
  /// reproducibility across runs (parallel reductions are not deterministic).
  pub fn with_intra_op_num_threads(mut self, n: usize) -> Self {
    self.intra_op_num_threads = Some(n);
    self
  }

  /// Override `inter_op_num_threads`.
  pub fn with_inter_op_num_threads(mut self, n: usize) -> Self {
    self.inter_op_num_threads = Some(n);
    self
  }

  /// Apply the option set to a `SessionBuilder`.
  fn apply(self, mut builder: SessionBuilder) -> Result<SessionBuilder, Error> {
    if let Some(level) = self.optimization_level {
      builder = builder
        .with_optimization_level(level)
        .map_err(ort::Error::from)?;
    }
    if let Some(n) = self.intra_op_num_threads {
      builder = builder.with_intra_threads(n).map_err(ort::Error::from)?;
    }
    if let Some(n) = self.inter_op_num_threads {
      builder = builder.with_inter_threads(n).map_err(ort::Error::from)?;
    }
    if !self.providers.is_empty() {
      builder = builder
        .with_execution_providers(self.providers)
        .map_err(ort::Error::from)?;
    }
    Ok(builder)
  }
}

/// Thin ort wrapper for one segmentation model session.
///
/// Owns one `ort::Session` plus reusable input scratch. Auto-derives
/// `Send`; does NOT auto-derive `Sync` because `ort::Session` is `!Sync`.
/// Use one per worker thread. Matches `silero::Session` exactly
/// (silero/src/session.rs line 61: "Send but not Sync").
///
/// **Shape validation:** v0.1.0 validates the model's output shape on first
/// inference (returns [`Error::InferenceShapeMismatch`] if `[589, 7]` is
/// violated). Load-time dimension verification (`Error::IncompatibleModel`)
/// is reserved for a future revision once a stable ort metadata API is
/// available.
pub struct SegmentModel {
  inner: OrtSession,
  input_scratch: Vec<f32>,
}

impl SegmentModel {
  /// Load the model from disk with default options.
  pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
    Self::from_file_with_options(path, SegmentModelOptions::default())
  }

  /// Load the model from disk with custom options.
  pub fn from_file_with_options<P: AsRef<Path>>(
    path: P,
    opts: SegmentModelOptions,
  ) -> Result<Self, Error> {
    let path = path.as_ref();
    let mut builder = opts.apply(OrtSession::builder()?)?;
    let session = builder
      .commit_from_file(path)
      .map_err(|source| Error::LoadModel {
        path: path.to_path_buf(),
        source,
      })?;
    Ok(Self::new_from_session(session))
  }

  /// Load the model from an in-memory ONNX byte buffer with default options.
  ///
  /// `bytes` is **copied** into ort's session; the buffer can be dropped
  /// immediately after this call returns.
  pub fn from_memory(bytes: &[u8]) -> Result<Self, Error> {
    Self::from_memory_with_options(bytes, SegmentModelOptions::default())
  }

  /// Load the model from an in-memory ONNX byte buffer with custom options.
  pub fn from_memory_with_options(bytes: &[u8], opts: SegmentModelOptions) -> Result<Self, Error> {
    let mut builder = opts.apply(OrtSession::builder()?)?;
    let session = builder.commit_from_memory(bytes)?;
    Ok(Self::new_from_session(session))
  }

  fn new_from_session(session: OrtSession) -> Self {
    Self {
      inner: session,
      input_scratch: Vec::with_capacity(WINDOW_SAMPLES as usize),
    }
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

    // Use the first input and first output by position. pyannote/segmentation-3.0
    // is a single-input, single-output model; this avoids needing to know the
    // name and is robust to exporter-version naming differences.
    let outputs = self.inner.run(ort::inputs![TensorRef::from_array_view((
      [1usize, 1usize, WINDOW_SAMPLES as usize],
      self.input_scratch.as_slice()
    ),)?,])?;

    let (_shape, data) = outputs[0].try_extract_tensor::<f32>()?;
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
  /// `silero::Session::process_stream`.
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
