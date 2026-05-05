//! ONNX Runtime execution providers — opt-in hardware acceleration.
//!
//! Each `*` cargo feature in `Cargo.toml` toggles the matching
//! ORT execution provider (EP). When the feature is on, the EP type
//! is re-exported here so callers can construct a provider and pass
//! it to [`crate::segment::SegmentModelOptions::with_providers`] or
//! [`crate::embed::EmbedModelOptions::with_providers`] without taking
//! a direct `ort` dependency.
//!
//! ## Example: register a single provider
//!
//! ```ignore
//! # // ignored: requires the `coreml` cargo feature + Apple host.
//! use diarization::{
//!   embed::{EmbedModel, EmbedModelOptions},
//!   ep::CoreMLExecutionProvider,
//!   segment::{SegmentModel, SegmentModelOptions},
//! };
//!
//! let seg_opts = SegmentModelOptions::default()
//!   .with_providers(vec![CoreMLExecutionProvider::default().build()]);
//! let mut seg = SegmentModel::bundled_with_options(seg_opts)?;
//!
//! let emb_opts = EmbedModelOptions::default()
//!   .with_providers(vec![CoreMLExecutionProvider::default().build()]);
//! let mut emb = EmbedModel::from_file_with_options(
//!   "models/wespeaker_resnet34_lm.onnx",
//!   emb_opts,
//! )?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Example: ship a single binary that auto-picks GPU
//!
//! Build with the `gpu` meta-feature (`--features gpu`); the helper
//! returns whichever EPs were compiled in, in a priority order. ORT
//! registers each as `MayUse` — the first one whose ops match runs
//! and the rest stay dormant on CPU fallback.
//!
//! ```ignore
//! # // ignored: depends on which `*` features are compiled in.
//! use diarization::ep::auto_providers;
//! let seg_opts = diarization::segment::SegmentModelOptions::default()
//!   .with_providers(auto_providers());
//! ```
//!
//! ## Runtime requirements
//!
//! The `*` cargo features only enable the *bindings*. Each EP
//! still needs the matching native library on the host:
//!
//! - `coreml` — Apple Silicon / macOS, no extra install (ships in
//!   the system).
//! - `cuda` — NVIDIA CUDA toolkit + cuDNN.
//! - `tensorrt` — NVIDIA TensorRT (also pulls CUDA).
//! - `directml` — Windows 10+ with DirectX 12.
//! - `rocm` / `migraphx` — AMD ROCm runtime.
//! - `openvino` — Intel OpenVINO toolkit.
//! - `webgpu` — a WebGPU-capable native runtime (Dawn / wgpu).
//! - `xnnpack` — ARM/x86 SIMD CPU EP, no extra install.
//! - others (`onednn`, `cann`, `acl`, `qnn`, `nnapi`, `tvm`, `azure`)
//!   follow vendor-specific install paths.
//!
//! The `ort` crate's default `download-binaries` feature ships a
//! CPU-only build; vendor EPs typically require either a vendor build
//! of onnxruntime or `LD_LIBRARY_PATH` / `DYLD_LIBRARY_PATH` pointing
//! at the vendor libs. See
//! <https://ort.pyke.io/perf/execution-providers> for setup details.
//!
//! ## EP determinism
//!
//! Different EPs can produce slightly different f32/f16 outputs from
//! the same model — vendor kernels round differently, fuse ops
//! differently, and use different math libraries. The dia parity
//! tests assert against pyannote's CPU reference; switching EPs may
//! perturb DER by a small amount but should not regress the partition
//! shape on realistic inputs. The CoreML EP, for instance, has been
//! observed to flip a few span boundaries on short fixtures while
//! preserving overall speaker counts.

pub use ort::execution_providers::ExecutionProviderDispatch;

#[cfg(feature = "coreml")]
#[cfg_attr(docsrs, doc(cfg(feature = "coreml")))]
pub use ort::execution_providers::CoreMLExecutionProvider;

#[cfg(feature = "cuda")]
#[cfg_attr(docsrs, doc(cfg(feature = "cuda")))]
pub use ort::execution_providers::CUDAExecutionProvider;

#[cfg(feature = "tensorrt")]
#[cfg_attr(docsrs, doc(cfg(feature = "tensorrt")))]
pub use ort::execution_providers::TensorRTExecutionProvider;

#[cfg(feature = "directml")]
#[cfg_attr(docsrs, doc(cfg(feature = "directml")))]
pub use ort::execution_providers::DirectMLExecutionProvider;

#[cfg(feature = "rocm")]
#[cfg_attr(docsrs, doc(cfg(feature = "rocm")))]
pub use ort::execution_providers::ROCmExecutionProvider;

#[cfg(feature = "migraphx")]
#[cfg_attr(docsrs, doc(cfg(feature = "migraphx")))]
pub use ort::execution_providers::MIGraphXExecutionProvider;

#[cfg(feature = "openvino")]
#[cfg_attr(docsrs, doc(cfg(feature = "openvino")))]
pub use ort::execution_providers::OpenVINOExecutionProvider;

#[cfg(feature = "webgpu")]
#[cfg_attr(docsrs, doc(cfg(feature = "webgpu")))]
pub use ort::execution_providers::WebGPUExecutionProvider;

#[cfg(feature = "xnnpack")]
#[cfg_attr(docsrs, doc(cfg(feature = "xnnpack")))]
pub use ort::execution_providers::XNNPACKExecutionProvider;

#[cfg(feature = "onednn")]
#[cfg_attr(docsrs, doc(cfg(feature = "onednn")))]
pub use ort::execution_providers::OneDNNExecutionProvider;

#[cfg(feature = "cann")]
#[cfg_attr(docsrs, doc(cfg(feature = "cann")))]
pub use ort::execution_providers::CANNExecutionProvider;

#[cfg(feature = "acl")]
#[cfg_attr(docsrs, doc(cfg(feature = "acl")))]
pub use ort::execution_providers::ACLExecutionProvider;

#[cfg(feature = "qnn")]
#[cfg_attr(docsrs, doc(cfg(feature = "qnn")))]
pub use ort::execution_providers::QNNExecutionProvider;

#[cfg(feature = "nnapi")]
#[cfg_attr(docsrs, doc(cfg(feature = "nnapi")))]
pub use ort::execution_providers::NNAPIExecutionProvider;

#[cfg(feature = "tvm")]
#[cfg_attr(docsrs, doc(cfg(feature = "tvm")))]
pub use ort::execution_providers::TVMExecutionProvider;

#[cfg(feature = "azure")]
#[cfg_attr(docsrs, doc(cfg(feature = "azure")))]
pub use ort::execution_providers::AzureExecutionProvider;

/// Build a provider list from whichever `*` features are compiled in.
///
/// Order is "most-likely-to-accelerate first":
/// `TensorRT → CUDA → CoreML → DirectML → ROCm → MIGraphX →
/// OpenVINO → WebGPU → OneDNN → XNNPACK → CANN → QNN → ACL →
/// NNAPI → TVM → Azure`. ORT registers each as `MayUse`,
/// so the first whose ops match accelerates and the rest stay
/// dormant on CPU fallback.
///
/// Returns an empty `Vec` if no `*` features are enabled, in which
/// case ORT runs on its default CPU dispatch.
///
/// # Example
///
/// ```ignore
/// # // ignored: depends on which `*` features are compiled in.
/// use diarization::ep::auto_providers;
///
/// let seg_opts = diarization::segment::SegmentModelOptions::default()
///   .with_providers(auto_providers());
/// ```
#[must_use]
pub fn auto_providers() -> Vec<ExecutionProviderDispatch> {
  #[allow(unused_mut)]
  let mut out: Vec<ExecutionProviderDispatch> = Vec::new();
  #[cfg(feature = "tensorrt")]
  out.push(TensorRTExecutionProvider::default().build());
  #[cfg(feature = "cuda")]
  out.push(CUDAExecutionProvider::default().build());
  #[cfg(feature = "coreml")]
  out.push(CoreMLExecutionProvider::default().build());
  #[cfg(feature = "directml")]
  out.push(DirectMLExecutionProvider::default().build());
  #[cfg(feature = "rocm")]
  out.push(ROCmExecutionProvider::default().build());
  #[cfg(feature = "migraphx")]
  out.push(MIGraphXExecutionProvider::default().build());
  #[cfg(feature = "openvino")]
  out.push(OpenVINOExecutionProvider::default().build());
  #[cfg(feature = "webgpu")]
  out.push(WebGPUExecutionProvider::default().build());
  #[cfg(feature = "onednn")]
  out.push(OneDNNExecutionProvider::default().build());
  #[cfg(feature = "xnnpack")]
  out.push(XNNPACKExecutionProvider::default().build());
  #[cfg(feature = "cann")]
  out.push(CANNExecutionProvider::default().build());
  #[cfg(feature = "qnn")]
  out.push(QNNExecutionProvider::default().build());
  #[cfg(feature = "acl")]
  out.push(ACLExecutionProvider::default().build());
  #[cfg(feature = "nnapi")]
  out.push(NNAPIExecutionProvider::default().build());
  #[cfg(feature = "tvm")]
  out.push(TVMExecutionProvider::default().build());
  #[cfg(feature = "azure")]
  out.push(AzureExecutionProvider::default().build());
  out
}
