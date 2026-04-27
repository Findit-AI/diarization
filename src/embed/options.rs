//! Constants for `dia::embed`. All values match spec §4.2 / §5.

/// 2 s @ 16 kHz; the WeSpeaker model's fixed input length.
///
/// Named with the `EMBED_` prefix to avoid collision with
/// `dia::segment::WINDOW_SAMPLES` (160 000 = 10 s at the same rate).
pub const EMBED_WINDOW_SAMPLES: u32 = 32_000;

/// 1 s @ 16 kHz; sliding-window hop for the long-clip path (§5.1).
/// 50 % overlap with `EMBED_WINDOW_SAMPLES`.
pub const HOP_SAMPLES: u32 = 16_000;

/// ~25 ms @ 16 kHz; one kaldi window. Below this, `embed` returns
/// [`Error::InvalidClip`](crate::embed::Error::InvalidClip).
pub const MIN_CLIP_SAMPLES: u32 = 400;

/// Number of mel bins in the kaldi fbank features (spec §4.2).
pub const FBANK_NUM_MELS: usize = 80;

/// Number of fbank frames per `EMBED_WINDOW_SAMPLES` of audio
/// (25 ms frame length, 10 ms shift → 200 frames per 2 s).
pub const FBANK_FRAMES: usize = 200;

/// Output dimensionality of the WeSpeaker ResNet34 embedding.
pub const EMBEDDING_DIM: usize = 256;

/// Numerical floor used in L2-normalization to avoid divide-by-zero.
/// Matches `findit-speaker-embedding`'s `1e-12` (verified at
/// `embedder.py:85`); diverging would lose Python parity in edge cases.
pub const NORM_EPSILON: f32 = 1e-12;

/// 16 kHz mono — the WeSpeaker ResNet34 expected sample rate.
/// Matches [`dia::segment::SAMPLE_RATE_HZ`](crate::segment::SAMPLE_RATE_HZ).
pub const SAMPLE_RATE_HZ: u32 = 16_000;

// ── EmbedModelOptions ─────────────────────────────────────────────────────

#[cfg(feature = "ort")]
use ort::execution_providers::ExecutionProviderDispatch;
#[cfg(feature = "ort")]
use ort::session::builder::{GraphOptimizationLevel, SessionBuilder};

/// Builder for [`EmbedModel`](crate::embed::EmbedModel) runtime configuration.
///
/// Mirrors [`SegmentModelOptions`](crate::segment::SegmentModelOptions): the
/// same four ort knobs (graph optimization level, execution providers,
/// intra/inter-op thread counts), with both consuming `with_*` and
/// in-place `set_*` builders.
///
/// Default: ort defaults for optimization level and threading, no
/// execution providers configured beyond ort's default search.
#[cfg(feature = "ort")]
#[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
#[derive(Default)]
pub struct EmbedModelOptions {
  optimization_level: Option<GraphOptimizationLevel>,
  providers: Vec<ExecutionProviderDispatch>,
  intra_op_num_threads: Option<usize>,
  inter_op_num_threads: Option<usize>,
}

#[cfg(feature = "ort")]
impl EmbedModelOptions {
  /// Construct with all-default options.
  pub fn new() -> Self {
    Self::default()
  }

  // ── Builder (consuming with_*) ───────────────────────────────────────

  /// Override the graph optimization level.
  pub fn with_optimization_level(mut self, level: GraphOptimizationLevel) -> Self {
    self.optimization_level = Some(level);
    self
  }

  /// Configure execution providers in priority order. Default: ort's
  /// default execution-provider selection (typically CPU).
  ///
  /// **Caveat:** non-CPU providers may degrade WeSpeaker ResNet34 numerics
  /// and break the byte-determinism guarantees in spec §11.9. Do not enable
  /// without measuring against the pyannote parity harness (Task 46).
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

  // ── Mutators (in-place set_*) ────────────────────────────────────────

  /// Set the graph optimization level (in-place).
  pub fn set_optimization_level(&mut self, level: GraphOptimizationLevel) -> &mut Self {
    self.optimization_level = Some(level);
    self
  }

  /// Set the execution providers (in-place).
  pub fn set_providers(&mut self, providers: Vec<ExecutionProviderDispatch>) -> &mut Self {
    self.providers = providers;
    self
  }

  /// Set `intra_op_num_threads` (in-place).
  pub fn set_intra_op_num_threads(&mut self, n: usize) -> &mut Self {
    self.intra_op_num_threads = Some(n);
    self
  }

  /// Set `inter_op_num_threads` (in-place).
  pub fn set_inter_op_num_threads(&mut self, n: usize) -> &mut Self {
    self.inter_op_num_threads = Some(n);
    self
  }

  // ── Internal apply ───────────────────────────────────────────────────

  /// Apply the option set to a `SessionBuilder`. Used internally by
  /// [`EmbedModel`](crate::embed::EmbedModel).
  #[allow(dead_code)] // wired into EmbedModel::from_file_with_options in Task 25
  pub(crate) fn apply(
    self,
    mut builder: SessionBuilder,
  ) -> Result<SessionBuilder, crate::embed::Error> {
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
