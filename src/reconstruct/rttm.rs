//! Convert per-frame discrete diarization grid → RTTM-style spans.

use crate::reconstruct::{algo::SlidingWindow, error::ShapeError};

/// One contiguous turn from the discrete diarization grid.
#[derive(Debug, Clone, PartialEq)]
pub struct RttmSpan {
  cluster: usize,
  start: f64,
  duration: f64,
}

impl RttmSpan {
  /// Construct a span. `start` and `duration` in seconds; `cluster`
  /// is the 0-indexed cluster id mapped to `SPEAKER_{cluster:02}` in
  /// [`spans_to_rttm_lines`].
  pub const fn new(cluster: usize, start: f64, duration: f64) -> Self {
    Self {
      cluster,
      start,
      duration,
    }
  }

  /// Cluster id (0-indexed).
  pub const fn cluster(&self) -> usize {
    self.cluster
  }

  /// Span start time in seconds.
  pub const fn start(&self) -> f64 {
    self.start
  }

  /// Span duration in seconds.
  pub const fn duration(&self) -> f64 {
    self.duration
  }

  /// Span end time in seconds (`start + duration`).
  pub fn end(&self) -> f64 {
    self.start + self.duration
  }
}

/// Walk a `(num_frames * num_clusters)` flat binary grid and emit one
/// [`RttmSpan`] per contiguous high-region per cluster column.
///
/// Time mapping: span `[t_start, t_end]` covers grid frames
/// `[i_start, i_end)`. Pyannote's `Binarize` uses *frame centers* as
/// span boundaries (`pyannote.audio.utils.signal.Binarize.__call__`
/// reads `timestamps = [frames[i].middle for i in range(num_frames)]`),
/// so:
///
/// ```text
/// start    = frames_sw.start + i_start * frames_sw.step + frames_sw.duration / 2
/// duration = (i_end - i_start) * frames_sw.step
/// ```
///
/// `min_duration_off` (if `> 0.0`) merges adjacent same-cluster spans
/// separated by a gap `≤ min_duration_off` (matches pyannote's
/// `Annotation.support(collar=...)`).
///
/// Spans across clusters are sorted by `(start, cluster)` for RTTM
/// canonical order.
///
/// # Panics
///
/// Panics if `grid.len() != num_frames * num_clusters` or if
/// `num_frames * num_clusters` overflows `usize`. Use
/// [`try_discrete_to_spans`] to surface the precondition as
/// `Result<_, ShapeError>` instead.
pub fn discrete_to_spans(
  grid: &[f32],
  num_frames: usize,
  num_clusters: usize,
  frames_sw: SlidingWindow,
  min_duration_off: f64,
) -> Vec<RttmSpan> {
  try_discrete_to_spans(grid, num_frames, num_clusters, frames_sw, min_duration_off)
    .expect("discrete_to_spans: shape precondition violated; use try_discrete_to_spans to handle")
}

/// Fallible variant of [`discrete_to_spans`]. Validates the grid
/// shape with checked arithmetic so an adversarial dimension product
/// (which would otherwise wrap silently in release and trivially match
/// a small grid) surfaces as a typed `ShapeError` instead of a
/// process panic.
///
/// # Errors
///
/// - [`ShapeError::GridSizeOverflow`] if `num_frames * num_clusters`
///   overflows `usize`.
/// - [`ShapeError::GridLenMismatch`] if `grid.len() != num_frames *
///   num_clusters`.
pub fn try_discrete_to_spans(
  grid: &[f32],
  num_frames: usize,
  num_clusters: usize,
  frames_sw: SlidingWindow,
  min_duration_off: f64,
) -> Result<Vec<RttmSpan>, ShapeError> {
  let expected = num_frames
    .checked_mul(num_clusters)
    .ok_or(ShapeError::GridSizeOverflow)?;
  if grid.len() != expected {
    return Err(ShapeError::GridLenMismatch);
  }
  let center_offset = frames_sw.duration() / 2.0;
  let frame_start = frames_sw.start();
  let frame_step = frames_sw.step();
  let mut spans: Vec<RttmSpan> = Vec::new();
  for k in 0..num_clusters {
    let mut per_cluster: Vec<(f64, f64)> = Vec::new(); // (start, end)
    let mut active_start: Option<usize> = None;
    for t in 0..num_frames {
      let v = grid[t * num_clusters + k] != 0.0;
      match (v, active_start) {
        (true, None) => active_start = Some(t),
        (false, Some(s)) => {
          let start = frame_start + s as f64 * frame_step + center_offset;
          let end = frame_start + t as f64 * frame_step + center_offset;
          per_cluster.push((start, end));
          active_start = None;
        }
        _ => {}
      }
    }
    // Span still active at end-of-grid: pyannote's `Binarize.__call__`
    // closes the trailing region with `t = timestamps[-1]` =
    // `timestamps[num_frames - 1]`, not `timestamps[num_frames]`.
    // Closing one step past the last frame would over-extend
    // end-of-file speakers by `frames_sw.step` and convert a single
    // final-frame run into a non-empty span where pyannote emits
    // none.
    if let Some(s) = active_start {
      let start = frame_start + s as f64 * frame_step + center_offset;
      let end = frame_start + (num_frames - 1) as f64 * frame_step + center_offset;
      if end > start {
        per_cluster.push((start, end));
      }
    }
    // min_duration_off: merge adjacent spans whose gap is `≤ collar`.
    if min_duration_off > 0.0 && per_cluster.len() > 1 {
      let mut merged: Vec<(f64, f64)> = Vec::with_capacity(per_cluster.len());
      let mut cur = per_cluster[0];
      for &(s, e) in per_cluster.iter().skip(1) {
        let gap = s - cur.1;
        if gap <= min_duration_off {
          cur.1 = e;
        } else {
          merged.push(cur);
          cur = (s, e);
        }
      }
      merged.push(cur);
      per_cluster = merged;
    }
    for (s, e) in per_cluster {
      spans.push(RttmSpan::new(k, s, e - s));
    }
  }
  spans.sort_by(|a, b| {
    a.start()
      .partial_cmp(&b.start())
      .unwrap_or(std::cmp::Ordering::Equal)
      .then(a.cluster().cmp(&b.cluster()))
  });
  Ok(spans)
}

/// Format spans as RTTM lines. Output is one line per span:
///
/// ```text
/// SPEAKER <uri> 1 <start> <duration> <NA> <NA> SPEAKER_<NN> <NA> <NA>
/// ```
///
/// Times are formatted to 3 decimal places (millisecond resolution),
/// matching pyannote's `Annotation.write_rttm` default.
///
/// Cluster ids are remapped to `SPEAKER_NN` matching pyannote's
/// `Annotation.labels()` = `sorted(_labels, key=str)`
/// (`pyannote.core.annotation.Annotation:920-932`). The smallest
/// label by decimal-string lex-order becomes `SPEAKER_00`, the
/// next `SPEAKER_01`, etc. For ids below 10 this agrees with
/// numeric order; above 10 they diverge (e.g. `["10", "2"]`
/// lex-sorts to `["10", "2"]`). Real workloads with long
/// recordings or large meetings can produce 10+ alive clusters, so
/// using numeric sort would silently mislabel speakers vs the
/// pyannote reference.
///
/// Implementation: [`cmp_cluster_id_str`] is the canonical
/// pyannote-equivalent comparator. It renders both ids into stack-
/// allocated `itoa::Buffer`s and compares the resulting `&str`
/// slices — zero heap allocation.
pub fn spans_to_rttm_lines(spans: &[RttmSpan], uri: &str) -> Vec<String> {
  use std::collections::HashMap;
  let mut unique_ids: Vec<usize> = spans.iter().map(|s| s.cluster()).collect();
  unique_ids.sort_unstable_by(|a, b| cmp_cluster_id_str(*a, *b));
  unique_ids.dedup();
  let id_to_label: HashMap<usize, usize> = unique_ids
    .into_iter()
    .enumerate()
    .map(|(i, id)| (id, i))
    .collect();
  spans
    .iter()
    .map(|s| {
      let label = id_to_label[&s.cluster()];
      format!(
        "SPEAKER {uri} 1 {:.3} {:.3} <NA> <NA> SPEAKER_{:02} <NA> <NA>",
        s.start(),
        s.duration(),
        label
      )
    })
    .collect()
}

/// Lexicographically compare two cluster ids by their decimal string
/// representation. Mirrors Python's `sorted([a, b], key=str)` ordering
/// used by `pyannote.core.Annotation.labels()`.
///
/// Allocation-free: `itoa::Buffer` is a stack-allocated `[u8; 40]`
/// (sized for any 64-bit integer). Two buffers per compare = ~80
/// bytes stack — sort_unstable_by drives this O(n log n) times for
/// `n` distinct cluster ids, all stack work.
pub fn cmp_cluster_id_str(a: usize, b: usize) -> std::cmp::Ordering {
  let mut buf_a = itoa::Buffer::new();
  let mut buf_b = itoa::Buffer::new();
  buf_a.format(a).cmp(buf_b.format(b))
}
