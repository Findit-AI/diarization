//! Convert per-frame discrete diarization grid → RTTM-style spans.

use crate::reconstruct_pyannote::algo::SlidingWindow;

/// One contiguous turn from the discrete diarization grid.
#[derive(Debug, Clone, PartialEq)]
pub struct RttmSpan {
  /// Cluster index (0-indexed). Mapped to `SPEAKER_{cluster:02}` in
  /// [`spans_to_rttm_lines`].
  pub cluster: usize,
  /// Start time in seconds.
  pub start: f64,
  /// Duration in seconds.
  pub duration: f64,
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
pub fn discrete_to_spans(
  grid: &[f32],
  num_frames: usize,
  num_clusters: usize,
  frames_sw: SlidingWindow,
  min_duration_off: f64,
) -> Vec<RttmSpan> {
  assert_eq!(grid.len(), num_frames * num_clusters);
  let center_offset = frames_sw.duration / 2.0;
  let mut spans: Vec<RttmSpan> = Vec::new();
  for k in 0..num_clusters {
    let mut per_cluster: Vec<(f64, f64)> = Vec::new(); // (start, end)
    let mut active_start: Option<usize> = None;
    for t in 0..num_frames {
      let v = grid[t * num_clusters + k] != 0.0;
      match (v, active_start) {
        (true, None) => active_start = Some(t),
        (false, Some(s)) => {
          let start = frames_sw.start + s as f64 * frames_sw.step + center_offset;
          let end = frames_sw.start + t as f64 * frames_sw.step + center_offset;
          per_cluster.push((start, end));
          active_start = None;
        }
        _ => {}
      }
    }
    if let Some(s) = active_start {
      let start = frames_sw.start + s as f64 * frames_sw.step + center_offset;
      let end = frames_sw.start + num_frames as f64 * frames_sw.step + center_offset;
      per_cluster.push((start, end));
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
      spans.push(RttmSpan {
        cluster: k,
        start: s,
        duration: e - s,
      });
    }
  }
  spans.sort_by(|a, b| {
    a.start
      .partial_cmp(&b.start)
      .unwrap_or(std::cmp::Ordering::Equal)
      .then(a.cluster.cmp(&b.cluster))
  });
  spans
}

/// Format spans as RTTM lines. Output is one line per span:
///
/// ```text
/// SPEAKER <uri> 1 <start> <duration> <NA> <NA> SPEAKER_<NN> <NA> <NA>
/// ```
///
/// Times are formatted to 3 decimal places (millisecond resolution),
/// matching pyannote's `Annotation.write_rttm` default.
pub fn spans_to_rttm_lines(spans: &[RttmSpan], uri: &str) -> Vec<String> {
  spans
    .iter()
    .map(|s| {
      format!(
        "SPEAKER {uri} 1 {:.3} {:.3} <NA> <NA> SPEAKER_{:02} <NA> <NA>",
        s.start, s.duration, s.cluster
      )
    })
    .collect()
}
