//! Heap-or-mmap spill buffer for size-known-upfront allocations.
//!
//! Several `Result`-returning public APIs in `dia` allocate flat
//! `Vec<T>` scratch buffers proportional to input size: AHC pdist
//! (`n*(n-1)/2` f64 cells), reconstruct grids, count-tensor
//! aggregates. Past a few hundred MB these can OOM-abort the
//! process from a `Result`-returning API.
//!
//! [`SpillVec<T>`] picks at construction time between two backends:
//!
//! - **Heap** (`Vec<T>`) when the requested allocation fits under
//!   [`SpillOptions::threshold_bytes`].
//! - **File-backed mmap** (over an unlinked tempfile) above the
//!   threshold. Pages are evicted to disk by the kernel page cache
//!   under memory pressure, keeping resident RAM bounded.
//!
//! The spill backend deliberately is NOT anonymous mmap
//! (`MAP_ANONYMOUS`): anonymous mmap stores dirty pages in
//! RAM + swap and consumes physical memory identically to `Vec`.
//! File-backed mmap is what actually trades RAM for disk.
//!
//! ## Transparent Huge Pages (Linux)
//!
//! On Linux, mmap'd buffers are advised with `MADV_HUGEPAGE`
//! (`Advice::HugePage`), which lets the kernel back the mapping with
//! 2 MiB pages where the THP policy permits. Reduces TLB pressure
//! on the dominant access patterns (sequential read of pdist /
//! aggregate buffers). The advise is opportunistic — silently
//! degrades to regular 4 KiB pages on kernels with THP disabled,
//! older kernels, or filesystems that don't support it.
//!
//! `memmapix` also exposes `MmapOptions::huge(Some(N))` which sets
//! `MAP_HUGETLB` on the resulting mapping, but only for `map_anon`:
//! the `map_mut` codepath ignores the `huge` field. We use
//! `map_mut(&tempfile)` (file-backed; spills dirty pages to disk)
//! rather than `map_anon` (anonymous; dirty pages stay in
//! RAM + swap), so `huge()` is unreachable for our backend.
//! Reaching `MAP_HUGETLB` over a tempfile would also require
//! mounting the file's parent on `hugetlbfs` plus a preconfigured
//! kernel hugepage pool — the wrong tradeoff for an opportunistic
//! perf hint that should fail soft. `MADV_HUGEPAGE` covers the
//! same TLB-win territory without those constraints.
//!
//! ## Configuration
//!
//! A single process-global [`SpillOptions`] held in
//! [`spill_options`] / [`set_spill_options`]. Every [`SpillVec::zeros`]
//! call reads the current value and picks heap vs. mmap accordingly.
//!
//! Each top-level Options struct in `dia`
//! (`OwnedPipelineOptions`, `OfflineInput`, `AssignEmbeddingsInput`,
//! `ReconstructInput`, `StreamingOfflineOptions`) carries a
//! [`SpillOptions`] field defaulting to [`SpillOptions::default`]. The
//! corresponding entry function calls [`set_spill_options`] at the
//! top of its body so the field's value takes effect for every
//! transitive `SpillVec::zeros` call. Concurrent multi-threaded
//! invocations with differing `spill_options` will race on this
//! global; multi-threaded callers should either set a stable global
//! once at startup and leave the field at its default, or
//! externally synchronize calls that customize spill behavior.
//!
//! ```no_run
//! use diarization::ops::spill::{SpillOptions, set_spill_options};
//!
//! // Process-global, e.g. set once at startup:
//! set_spill_options(
//!     SpillOptions::new()
//!         .with_threshold_bytes(128 * 1024 * 1024)
//!         .with_spill_dir(Some("/var/tmp/dia".into())),
//! );
//! ```
//!
//! ```no_run
//! use diarization::offline::OwnedPipelineOptions;
//! use diarization::ops::spill::SpillOptions;
//!
//! // Per-call setting via the entry-point Options struct: the entry
//! // function installs this on the global at the top of its body.
//! let opts = OwnedPipelineOptions::new().with_spill_options(
//!     SpillOptions::new().with_threshold_bytes(64 * 1024 * 1024),
//! );
//! ```
//!
//! Default: 256 MiB threshold, [`std::env::temp_dir`] for the spill
//! file. Production deployments where `/tmp` is `tmpfs` (Docker
//! default) **must** override `spill_dir` to a real-disk path,
//! otherwise "spill to disk" is a misnomer and the OOM concern
//! still applies.
//!
//! ## Type contract
//!
//! [`SpillVec<T>`] requires `T: bytemuck::Pod` — the type must be
//! plain-old-data (no padding, no destructors, every byte pattern
//! valid). `f64`, `f32`, `u8`, `u16`, `u32`, `u64`, `usize`, signed
//! variants all qualify; `bool` does NOT (only `0u8` and `1u8` are
//! valid). Mask buffers that previously stored `Vec<bool>` migrate
//! to `Vec<u8>` (0/1) when wrapped in `SpillVec`.
//!
//! ## Limitations
//!
//! - Sized once at construction. No `push`/`grow`. Every call site
//!   in `dia` knows the buffer length upfront, so this is fine.
//! - `Send` but not `Sync`. The `Vec` and `MmapMut` backends both
//!   have unique-ownership semantics; sharing a mutable handle
//!   across threads would race on the `as_mut_slice` borrow.

// Internal call sites currently use `as_mut_slice` exclusively;
// the read-only / inspection accessors and the configuration
// setters are part of the public API for downstream consumers and
// tests, but Rust flags them as "never used" inside the crate.
#![allow(dead_code)]

use core::marker::PhantomData;
use std::{
  path::{Path, PathBuf},
  sync::{OnceLock, RwLock},
};

use bytemuck::Pod;
#[cfg(target_os = "linux")]
use memmapix::Advice;
use memmapix::{MmapMut, MmapOptions};

/// Errors returned by [`SpillVec`] allocation.
#[derive(Debug, thiserror::Error)]
pub enum SpillError {
  /// `n.checked_mul(size_of::<T>())` overflowed `usize`. The caller
  /// asked for an allocation past `usize::MAX` bytes.
  #[error("spill: requested element count {n} times size_of::<T>={element_size} overflows usize")]
  SizeOverflow {
    /// Requested element count.
    n: usize,
    /// Per-element size (`size_of::<T>()`).
    element_size: usize,
  },
  /// Failed to create the unlinked tempfile that backs the mmap.
  /// Realistic causes: `ENOSPC`, `EACCES`, `EROFS`, missing
  /// `spill_dir` permissions.
  #[error("spill: failed to create tempfile in {dir:?}: {source}")]
  TempfileCreation {
    /// Directory the tempfile was created in (`None` =
    /// [`std::env::temp_dir`]).
    dir: Option<PathBuf>,
    /// Underlying I/O error.
    #[source]
    source: std::io::Error,
  },
  /// Failed to grow the tempfile to the requested size via
  /// `set_len`. Typically `ENOSPC`.
  #[error("spill: failed to grow tempfile to {bytes} bytes: {source}")]
  TempfileGrow {
    /// Requested file length in bytes.
    bytes: u64,
    /// Underlying I/O error.
    #[source]
    source: std::io::Error,
  },
  /// `mmap()` failed. Realistic causes on Linux: `EAGAIN` (locked
  /// memory limit), `ENFILE`/`EMFILE` (fd limit), `ENOMEM`
  /// (kernel-side address-space exhaustion).
  #[error("spill: mmap failed for {bytes} bytes: {source}")]
  MmapFailed {
    /// Requested mapping length in bytes.
    bytes: usize,
    /// Underlying I/O error from the mmap syscall.
    #[source]
    source: std::io::Error,
  },
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_threshold_bytes() -> usize {
  SpillOptions::DEFAULT_THRESHOLD_BYTES
}

/// Configuration for the spill backend. All fields are private;
/// access via the getters and modify via the `with_*` / `set_*`
/// builders.
///
/// Construct via [`SpillOptions::new`] (`const fn`) or [`Default`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SpillOptions {
  threshold_bytes: usize,
  #[cfg_attr(
    feature = "serde",
    serde(skip_serializing_if = "Option::is_none", default)
  )]
  spill_dir: Option<PathBuf>,
}

impl SpillOptions {
  /// Default threshold: 256 MiB. Allocations smaller than this stay
  /// on the heap; larger ones spill to file-backed mmap.
  ///
  /// 256 MiB is the OOM-cliff floor for typical 1–2 GiB container
  /// memory budgets: most production deployments can absorb a
  /// 256 MiB scratch buffer on the heap, but a 512 MiB or larger
  /// allocation hits container limits or fragments the heap arena.
  pub const DEFAULT_THRESHOLD_BYTES: usize = 256 * 1024 * 1024;

  /// Construct with default values: 256 MiB threshold,
  /// [`std::env::temp_dir`] for the spill directory.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      threshold_bytes: default_threshold_bytes(),
      spill_dir: None,
    }
  }

  /// Threshold (bytes) above which an allocation spills to mmap.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn threshold_bytes(&self) -> usize {
    self.threshold_bytes
  }

  /// Spill directory. `None` ⇒ [`std::env::temp_dir`]. Override to a
  /// real-disk path on deployments where `/tmp` is `tmpfs` (Docker
  /// default) — otherwise spilled pages live in RAM-backed `tmpfs`
  /// and the OOM concern is unaddressed.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn spill_dir(&self) -> Option<&Path> {
    self.spill_dir.as_deref()
  }

  /// Builder: set the spill threshold.
  #[must_use]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_threshold_bytes(mut self, threshold_bytes: usize) -> Self {
    self.set_threshold_bytes(threshold_bytes);
    self
  }

  /// Builder: set the spill directory. `None` resets to
  /// [`std::env::temp_dir`].
  #[must_use]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_spill_dir(mut self, spill_dir: Option<PathBuf>) -> Self {
    self.set_spill_dir(spill_dir);
    self
  }

  /// Mutating: set the spill threshold.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_threshold_bytes(&mut self, threshold_bytes: usize) -> &mut Self {
    self.threshold_bytes = threshold_bytes;
    self
  }

  /// Mutating: set the spill directory.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_spill_dir(&mut self, spill_dir: Option<PathBuf>) -> &mut Self {
    self.spill_dir = spill_dir;
    self
  }
}

impl Default for SpillOptions {
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn default() -> Self {
    Self::new()
  }
}

// ── Config: single process-global ─────────────────────────────────

/// `OnceLock<RwLock<SpillOptions>>` because:
/// - `OnceLock` defers default construction until first access (the
///   `Default::default()` call needs heap allocation for `PathBuf`,
///   which can't run at static-init time).
/// - `RwLock` (rather than `Mutex`) because reads vastly outnumber
///   writes — every spill allocation reads the config.
static SPILL_OPTIONS: OnceLock<RwLock<SpillOptions>> = OnceLock::new();

fn options_lock() -> &'static RwLock<SpillOptions> {
  SPILL_OPTIONS.get_or_init(|| RwLock::new(SpillOptions::new()))
}

/// Returns a clone of the current process-global [`SpillOptions`].
///
/// Cloning is cheap (a `usize` plus an `Option<PathBuf>` — at most a
/// few words). Returning a reference would require a guard type
/// that holds the read lock for the caller's lifetime, complicating
/// the API for no measurable win.
pub fn spill_options() -> SpillOptions {
  options_lock()
    .read()
    .expect("spill options lock poisoned")
    .clone()
}

/// Replace the process-global [`SpillOptions`]. Subsequent
/// [`SpillVec`] allocations use the new values; in-flight `SpillVec`
/// instances are NOT affected — once constructed, a buffer is
/// committed to its backend.
///
/// Multi-threaded callers must externally synchronize concurrent
/// writes that pick different values; the writes themselves are
/// race-free under the lock, but a `SpillVec::zeros` call observes
/// whichever value is current at its read.
pub fn set_spill_options(opts: SpillOptions) {
  *options_lock().write().expect("spill options lock poisoned") = opts;
}

/// A fixed-size flat buffer that picks heap-or-mmap at construction
/// time based on the process-global [`SpillOptions`].
///
/// Behaves like `Vec<T>` for read/write access via `as_slice` /
/// `as_mut_slice`. Length is set once at construction; the buffer
/// does not grow.
///
/// `T: Pod` so the byte buffer underlying the mmap can be
/// reinterpreted as `&[T]` / `&mut [T]` without UB. `bool` is NOT
/// `Pod` (only `0u8` and `1u8` are valid byte patterns); use
/// `Vec<u8>`-as-mask wrapped in `SpillVec<u8>` for boolean masks.
pub struct SpillVec<T> {
  inner: SpillInner,
  len: usize,
  _phantom: PhantomData<T>,
}

enum SpillInner {
  Heap(Vec<u8>),
  /// `_file` owns the tempfile so its lifetime ≥ the mmap's. The
  /// `tempfile::tempfile()` call returns an unlinked file; closing
  /// it (on drop) reclaims disk space.
  Mmap {
    map: MmapMut,
    _file: tempfile::NamedTempFile,
  },
}

impl<T> SpillVec<T> {
  /// Allocate `n` zero-initialized cells of `T`.
  ///
  /// Picks heap if `n * size_of::<T>() ≤ threshold_bytes`, else
  /// file-backed mmap in [`SpillOptions::spill_dir`]. Both backends
  /// return zero-initialized cells (`Vec<u8>::resize(_, 0)` for
  /// heap, `set_len` plus `mmap` for the file backend).
  pub fn zeros(n: usize) -> Result<Self, SpillError> {
    let element_size = core::mem::size_of::<T>();
    let bytes = n
      .checked_mul(element_size)
      .ok_or(SpillError::SizeOverflow { n, element_size })?;

    // Special case: `n == 0` always returns a heap-empty buffer.
    // mmap of length 0 is undefined / EINVAL on most platforms.
    if bytes == 0 {
      return Ok(Self {
        inner: SpillInner::Heap(Vec::new()),
        len: 0,
        _phantom: PhantomData,
      });
    }

    let opts = spill_options();
    if bytes <= opts.threshold_bytes() {
      // Heap path. Zeroed via `vec![0u8; bytes]`.
      Ok(Self {
        inner: SpillInner::Heap(vec![0u8; bytes]),
        len: n,
        _phantom: PhantomData,
      })
    } else {
      // mmap path.
      Self::new_mmap(n, bytes, opts.spill_dir())
    }
  }

  fn new_mmap(n: usize, bytes: usize, spill_dir: Option<&Path>) -> Result<Self, SpillError> {
    let file = match spill_dir {
      Some(dir) => tempfile::NamedTempFile::new_in(dir),
      None => tempfile::NamedTempFile::new(),
    }
    .map_err(|source| SpillError::TempfileCreation {
      dir: spill_dir.map(|d| d.to_path_buf()),
      source,
    })?;
    file
      .as_file()
      .set_len(bytes as u64)
      .map_err(|source| SpillError::TempfileGrow {
        bytes: bytes as u64,
        source,
      })?;
    // SAFETY: `file` is freshly created and exclusively owned by
    // this `SpillVec`. No other process or thread can mutate the
    // file or the resulting mapping.
    let map = unsafe {
      MmapOptions::new()
        .len(bytes)
        .map_mut(file.as_file())
        .map_err(|source| SpillError::MmapFailed { bytes, source })?
    };
    // Linux: hint the kernel to back the mapping with Transparent
    // Huge Pages where possible. Reduces TLB pressure for the
    // sequential read patterns in pdist/reconstruct (256 MB+
    // mappings touch ~64k regular pages but only ~128 huge pages).
    //
    // This is a HINT — `MADV_HUGEPAGE` is silently a no-op on
    // kernels where THP is disabled (`echo never >
    // /sys/kernel/mm/transparent_hugepage/enabled`), embedded
    // builds without THP, or filesystems that don't support it.
    // We deliberately do NOT use `MAP_HUGETLB`: it requires the
    // file to live on `hugetlbfs` and hard-fails if the kernel
    // hugepage pool is empty — wrong tradeoff for an opportunistic
    // optimization.
    //
    // We ignore the error result: a failed `madvise` on a freshly
    // created mapping is benign (the mapping is still valid),
    // and we don't want a system policy decision to fail an
    // otherwise-successful allocation.
    #[cfg(target_os = "linux")]
    let _ = map.advise(Advice::HugePage);
    Ok(Self {
      inner: SpillInner::Mmap { map, _file: file },
      len: n,
      _phantom: PhantomData,
    })
  }

  /// Number of `T` cells in the buffer.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn len(&self) -> usize {
    self.len
  }

  /// `true` if the buffer is empty.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn is_empty(&self) -> bool {
    self.len == 0
  }

  /// Borrow the buffer as `&[T]`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn as_slice(&self) -> &[T]
  where
    T: Pod,
  {
    let bytes: &[u8] = match &self.inner {
      SpillInner::Heap(v) => v.as_slice(),
      SpillInner::Mmap { map, .. } => &map[..],
    };
    if bytes.is_empty() {
      return &[];
    }
    bytemuck::cast_slice(bytes)
  }

  /// Borrow the buffer as `&mut [T]`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn as_mut_slice(&mut self) -> &mut [T]
  where
    T: Pod,
  {
    let bytes: &mut [u8] = match &mut self.inner {
      SpillInner::Heap(v) => v.as_mut_slice(),
      SpillInner::Mmap { map, .. } => &mut map[..],
    };
    if bytes.is_empty() {
      return &mut [];
    }
    bytemuck::cast_slice_mut(bytes)
  }

  /// Returns `true` if this buffer is backed by an mmap'd tempfile.
  /// `false` if it is heap-backed.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn is_mmapped(&self) -> bool {
    matches!(self.inner, SpillInner::Mmap { .. })
  }
}

// SAFETY: a `SpillVec<T>` owns its backing storage uniquely (heap
// `Vec` or per-instance `MmapMut + NamedTempFile`). Sending the
// owned handle across threads is safe; both `Vec<u8>` and
// `MmapMut` are `Send`. We do NOT impl `Sync` because
// `as_mut_slice` exposes a `&mut [T]` whose aliasing semantics
// require unique access.
unsafe impl<T: Pod + Send> Send for SpillVec<T> {}

#[cfg(test)]
mod tests {
  use super::*;
  use std::sync::Mutex;

  /// Tests in this module mutate the process-global `SPILL_OPTIONS`.
  /// `cargo test` runs tests in parallel by default, so without a
  /// shared mutex two tests writing different values would race —
  /// e.g. `small_allocation_uses_heap` would observe the
  /// `threshold = 0` write from `read_write_roundtrip_both_backends`
  /// and see a mmap-backed buffer where it expects heap.
  ///
  /// Each test that touches the global acquires this lock for its
  /// full body and resets to a known state at entry.
  static TEST_LOCK: Mutex<()> = Mutex::new(());

  fn lock() -> std::sync::MutexGuard<'static, ()> {
    TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
  }

  /// `SpillOptions::new()` is `const fn` and produces the documented
  /// default values.
  #[test]
  fn default_options_const_fn() {
    const OPTS: SpillOptions = SpillOptions::new();
    assert_eq!(
      OPTS.threshold_bytes(),
      SpillOptions::DEFAULT_THRESHOLD_BYTES
    );
    assert_eq!(SpillOptions::DEFAULT_THRESHOLD_BYTES, 256 * 1024 * 1024);
  }

  /// `with_threshold_bytes` is `const fn`; constructing a tuned
  /// `SpillOptions` at compile time is supported.
  #[test]
  fn const_fn_builder() {
    const OPTS: SpillOptions = SpillOptions::new().with_threshold_bytes(1024);
    assert_eq!(OPTS.threshold_bytes(), 1024);
    assert!(OPTS.spill_dir().is_none());
  }

  #[test]
  fn set_threshold_bytes_chains() {
    let mut opts = SpillOptions::new();
    opts
      .set_threshold_bytes(42)
      .set_spill_dir(Some("/tmp/dia".into()));
    assert_eq!(opts.threshold_bytes(), 42);
    assert_eq!(opts.spill_dir(), Some(Path::new("/tmp/dia")));
  }

  /// `SpillVec::zeros(0)` returns an empty heap buffer, never
  /// touching mmap (mmap of length 0 is `EINVAL` on most platforms).
  #[test]
  fn zeros_zero_returns_heap_empty() {
    let _guard = lock();
    set_spill_options(SpillOptions::default());
    let v: SpillVec<f64> = SpillVec::zeros(0).expect("zero-length must succeed");
    assert_eq!(v.len(), 0);
    assert!(v.is_empty());
    assert_eq!(v.as_slice().len(), 0);
    assert!(!v.is_mmapped());
  }

  /// Below threshold: heap-backed.
  #[test]
  fn small_allocation_uses_heap() {
    let _guard = lock();
    // Default threshold is 256 MiB; a 1 KiB f64 buffer is well under.
    set_spill_options(SpillOptions::default());
    let v: SpillVec<f64> = SpillVec::zeros(128).expect("alloc");
    assert_eq!(v.len(), 128);
    assert!(!v.is_mmapped());
    assert!(v.as_slice().iter().all(|&x| x == 0.0));
  }

  /// Reads and writes round-trip through both backends. We force
  /// the mmap path by overriding the threshold to 0.
  #[test]
  fn read_write_roundtrip_both_backends() {
    let _guard = lock();
    // Force mmap path by setting threshold to 0.
    set_spill_options(SpillOptions::default().with_threshold_bytes(0));
    let mut v: SpillVec<f64> = SpillVec::zeros(64).expect("mmap alloc");
    assert!(v.is_mmapped(), "should be mmap-backed at threshold=0");
    for (i, slot) in v.as_mut_slice().iter_mut().enumerate() {
      *slot = i as f64 * 1.5;
    }
    for (i, &x) in v.as_slice().iter().enumerate() {
      assert_eq!(x, i as f64 * 1.5);
    }
    drop(v);
    // Now force heap path by setting threshold huge.
    set_spill_options(SpillOptions::default().with_threshold_bytes(usize::MAX));
    let mut v: SpillVec<f64> = SpillVec::zeros(64).expect("heap alloc");
    assert!(
      !v.is_mmapped(),
      "should be heap-backed at threshold=usize::MAX"
    );
    for (i, slot) in v.as_mut_slice().iter_mut().enumerate() {
      *slot = i as f64 * 1.5;
    }
    for (i, &x) in v.as_slice().iter().enumerate() {
      assert_eq!(x, i as f64 * 1.5);
    }
    set_spill_options(SpillOptions::default());
  }

  /// Differential test: heap and mmap backends must produce
  /// bit-identical contents for the same write sequence.
  #[test]
  fn heap_mmap_differential_bit_equal() {
    let _guard = lock();
    fn fill_and_collect<F: FnOnce(&mut SpillVec<f64>)>(threshold: usize, fill: F) -> Vec<f64> {
      set_spill_options(SpillOptions::new().with_threshold_bytes(threshold));
      let mut v: SpillVec<f64> = SpillVec::zeros(1024).expect("alloc");
      fill(&mut v);
      v.as_slice().to_vec()
    }
    let fill_pattern = |v: &mut SpillVec<f64>| {
      for (i, slot) in v.as_mut_slice().iter_mut().enumerate() {
        *slot = (i as f64).sqrt() + 0.001 * (i as f64);
      }
    };
    let heap = fill_and_collect(usize::MAX, fill_pattern);
    let mmap = fill_and_collect(0, fill_pattern);
    assert_eq!(
      heap.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
      mmap.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
      "heap and mmap backends must produce bit-equal contents"
    );
    set_spill_options(SpillOptions::default());
  }

  /// Size-overflow surfaces a typed error instead of panicking.
  #[test]
  fn size_overflow_returns_typed_error() {
    // `usize::MAX / 4` * `size_of::<f64>() = 8` overflows usize.
    // Use `.err()` rather than `assert!(matches!)` because `SpillVec`
    // is not `Debug` (mmap state isn't usefully printable).
    let r: Result<SpillVec<f64>, _> = SpillVec::zeros(usize::MAX / 4);
    let err = r.err().expect("must error on overflow");
    assert!(
      matches!(err, SpillError::SizeOverflow { .. }),
      "got {err:?}"
    );
  }

  /// `Vec<u8>`-as-mask works for the boolean-mask migration. `bool`
  /// is not `Pod` so the masks switch to `u8` (0/1).
  #[test]
  fn u8_mask_roundtrip() {
    let _guard = lock();
    set_spill_options(SpillOptions::default());
    let mut v: SpillVec<u8> = SpillVec::zeros(16).expect("alloc");
    for (i, slot) in v.as_mut_slice().iter_mut().enumerate() {
      *slot = if i.is_multiple_of(2) { 1 } else { 0 };
    }
    let s = v.as_slice();
    for i in 0..16 {
      assert_eq!(s[i], if i.is_multiple_of(2) { 1 } else { 0 });
    }
  }

  /// `f32` masks (round-26's reconstruct cells are f32). Confirm
  /// the type works.
  #[test]
  fn f32_roundtrip() {
    let _guard = lock();
    set_spill_options(SpillOptions::default());
    let mut v: SpillVec<f32> = SpillVec::zeros(8).expect("alloc");
    let target: [f32; 8] = [
      0.0,
      1.0,
      0.5,
      -0.25,
      1e10,
      -1e10,
      f32::EPSILON,
      -f32::EPSILON,
    ];
    v.as_mut_slice().copy_from_slice(&target);
    assert_eq!(v.as_slice(), &target);
  }

  /// `set_spill_options` round-trips and is observed by subsequent
  /// `SpillVec::zeros` calls.
  #[test]
  fn set_spill_options_takes_effect() {
    let _guard = lock();
    set_spill_options(SpillOptions::new().with_threshold_bytes(0));
    let v: SpillVec<f64> = SpillVec::zeros(64).expect("alloc");
    assert!(v.is_mmapped());
    drop(v);

    set_spill_options(SpillOptions::new().with_threshold_bytes(usize::MAX));
    let v: SpillVec<f64> = SpillVec::zeros(64).expect("alloc");
    assert!(!v.is_mmapped());

    set_spill_options(SpillOptions::default());
  }
}
