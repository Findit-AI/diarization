//! Reconstruction math: clustered_segmentations + overlap-add aggregate
//! + top-K binarize.

use crate::{hungarian::UNMATCHED, reconstruct_pyannote::error::Error};

/// Hard upper bound on the cluster-id range accepted in `hard_clusters`.
/// Pyannote's diarization pipeline emits ids bounded by the alive
/// cluster count after VBx (typically 1–4). `1024` is ~256× any
/// realistic speaker count; it stops a corrupt or malicious caller
/// from driving the `num_clusters * num_chunks * num_frames_per_chunk`
/// allocation into the multi-GB range. Codex review MEDIUM round 4
/// of Phase 5.
pub const MAX_CLUSTER_ID: i32 = 1023;

/// Hard upper bound on `count[t]` (instantaneous active speaker count
/// per output frame). Pyannote derives `count` from
/// `aggregate(sum(binarized_seg, axis=-1))`, so the theoretical max is
/// `overlap_factor * num_speakers` ≈ 30 for the community-1 config
/// (10s chunk, 1s step, 3 speakers). Real fixtures observe max=2.
/// Capping at `64` allows comfortable headroom over realistic values
/// while catching `u8::MAX = 255`-style sentinel corruption that would
/// drive `num_clusters` and the top-K binarize past the actual
/// speaker space. Codex review HIGH round 5 of Phase 5.
pub const MAX_COUNT_PER_FRAME: u8 = 64;

/// Pyannote `SlidingWindow` (start, duration, step), all in seconds.
#[derive(Debug, Clone, Copy)]
pub struct SlidingWindow {
  pub start: f64,
  pub duration: f64,
  pub step: f64,
}

impl SlidingWindow {
  /// `pyannote.core.SlidingWindow.closest_frame(t)` — round to the
  /// nearest frame index whose center is at `t`. Frame `i`'s center
  /// is at `start + duration / 2 + i * step`.
  fn closest_frame(&self, t: f64) -> i64 {
    ((t - self.start - self.duration / 2.0) / self.step).round() as i64
  }
}

/// Inputs to [`reconstruct`].
#[derive(Debug, Clone)]
pub struct ReconstructInput<'a> {
  /// Per-chunk per-frame per-speaker segmentation activity, flattened
  /// `[c][f][s]` to length `num_chunks * num_frames_per_chunk *
  /// num_speakers`.
  pub segmentations: &'a [f64],
  pub num_chunks: usize,
  pub num_frames_per_chunk: usize,
  pub num_speakers: usize,
  /// Per-chunk hard cluster assignment (output of `dia::pipeline`).
  /// Length `num_chunks`; each inner vector has length `num_speakers`
  /// with `-2` indicating an unmatched speaker.
  pub hard_clusters: &'a [Vec<i32>],
  /// Per-output-frame instantaneous speaker count (from pyannote's
  /// segmentation binarization). Length `num_output_frames`.
  pub count: &'a [u8],
  pub num_output_frames: usize,
  /// Outer (chunk-level) sliding window — defines chunk start times.
  pub chunks_sw: SlidingWindow,
  /// Inner (frame-level) sliding window — defines per-output-frame
  /// timing. Used to compute `closest_frame(chunk_time)`.
  pub frames_sw: SlidingWindow,
}

/// Run pyannote's reconstruction.
///
/// Returns a binary `(num_output_frames * num_clusters)` flat vector
/// where row `t` has `1.0` at the top-`count[t]` cluster indices by
/// aggregated activation, `0.0` elsewhere.
///
/// `num_clusters` is derived as `max(hard_clusters) + 1`. If all
/// clusters are `UNMATCHED` (`-2`), returns an all-zero grid (no
/// clusters to assign).
///
/// # Errors
///
/// - [`Error::Shape`] for any dimension mismatch.
/// - [`Error::NonFinite`] if `segmentations` contains a non-finite
///   value (NaN handling is supported via [`Inference::aggregate`]'s
///   mask path; arbitrary `±inf` is rejected).
/// - [`Error::Timing`] for non-finite or non-positive sliding-window
///   parameters.
pub fn reconstruct(input: &ReconstructInput<'_>) -> Result<Vec<f32>, Error> {
  let &ReconstructInput {
    segmentations,
    num_chunks,
    num_frames_per_chunk,
    num_speakers,
    hard_clusters,
    count,
    num_output_frames,
    chunks_sw,
    frames_sw,
  } = input;

  // ── Boundary checks ────────────────────────────────────────────
  if num_chunks == 0 {
    return Err(Error::Shape("num_chunks must be at least 1"));
  }
  if num_frames_per_chunk == 0 {
    return Err(Error::Shape("num_frames_per_chunk must be at least 1"));
  }
  if num_speakers == 0 {
    return Err(Error::Shape("num_speakers must be at least 1"));
  }
  if segmentations.len() != num_chunks * num_frames_per_chunk * num_speakers {
    return Err(Error::Shape(
      "segmentations.len() != num_chunks * num_frames_per_chunk * num_speakers",
    ));
  }
  if hard_clusters.len() != num_chunks {
    return Err(Error::Shape("hard_clusters.len() != num_chunks"));
  }
  for row in hard_clusters {
    if row.len() != num_speakers {
      return Err(Error::Shape(
        "each hard_clusters[c] must have length num_speakers",
      ));
    }
  }
  if count.len() != num_output_frames {
    return Err(Error::Shape("count.len() != num_output_frames"));
  }
  // count[t] = instantaneous active speaker count at output frame t.
  // Pyannote derives this from `aggregate(sum(binarized_seg, axis=-1))`
  // which sums per-chunk active counts over overlapping chunks. Real
  // fixtures observe max=2; theoretical max for community-1 is
  // overlap_factor * num_speakers ≈ 30. `MAX_COUNT_PER_FRAME = 64`
  // allows headroom while catching u8::MAX=255 sentinel corruption that
  // would expand `num_clusters` past the actual speaker space and
  // fabricate dummy speakers in the top-K binarize. Codex review HIGH
  // round 5 of Phase 5.
  for &c in count {
    if c > MAX_COUNT_PER_FRAME {
      return Err(Error::Shape(
        "count entry exceeds MAX_COUNT_PER_FRAME (64)",
      ));
    }
  }
  for w in [chunks_sw, frames_sw] {
    if !w.duration.is_finite() || !w.step.is_finite() || !w.start.is_finite() {
      return Err(Error::Timing("non-finite sliding-window parameter"));
    }
    if w.duration <= 0.0 || w.step <= 0.0 {
      return Err(Error::Timing("non-positive duration or step"));
    }
  }
  // Reject all non-finite segmentation values (NaN and ±inf). Pyannote's
  // `Inference.aggregate` does `np.nan_to_num(score, nan=0.0)` and tracks
  // missingness via a parallel mask, but the realistic source of NaN is
  // upstream model corruption (torch nan-prop), and a silent fallback
  // here lets a degraded inference dependency produce plausible-but-
  // wrong RTTM output. Surfacing it as a clear typed error matches the
  // Phase 3 round-2 decision for `dia::hungarian` (±inf rejection at the
  // solver boundary). Codex review MEDIUM round 2 of Phase 5.
  for &v in segmentations {
    if !v.is_finite() {
      return Err(Error::NonFinite("segmentations"));
    }
  }

  // Validate cluster ids: `UNMATCHED` (-2) is allowed; non-negative
  // values must be in `[0, MAX_CLUSTER_ID]`. Codex review MEDIUM
  // round 4: a stray negative id (e.g. -1) silently dropped active
  // speech under the previous code (skipped by the speakers_in_k
  // filter), and a corrupt large positive id could drive the
  // num_clusters allocation into multi-GB range.
  for row in hard_clusters {
    for &k in row {
      if k == UNMATCHED {
        continue;
      }
      if k < 0 {
        return Err(Error::Shape(
          "hard_clusters contains a negative id other than UNMATCHED",
        ));
      }
      if k > MAX_CLUSTER_ID {
        return Err(Error::Shape(
          "hard_clusters id exceeds MAX_CLUSTER_ID (1023)",
        ));
      }
    }
  }

  // Determine num_clusters from hard_clusters.
  let mut max_cluster = -1i32;
  for row in hard_clusters {
    for &k in row {
      if k > max_cluster {
        max_cluster = k;
      }
    }
  }
  if max_cluster < 0 {
    // No assigned clusters anywhere — return all-zero grid.
    return Ok(vec![0.0; num_output_frames]);
  }
  let num_clusters_from_hard = (max_cluster + 1) as usize;

  // Pyannote pads num_clusters up to `max(count)` if needed (so the
  // top-K binarization can pull at least `count[t]` cluster slots).
  let max_count = count.iter().copied().max().unwrap_or(0) as usize;
  let num_clusters = num_clusters_from_hard.max(max_count.max(1));

  // ── Stage 1: clustered_segmentations ────────────────────────────
  // Initialized to NaN sentinel. We track NaN-ness via a parallel
  // bool mask to avoid f64::is_nan overhead in the aggregation loop.
  // Per-chunk: for each cluster k present in hard_clusters[c],
  // clustered[c, f, k] = max over speakers s where hard_clusters[c, s] == k
  //                       of segmentations[c, f, s].
  let cs_size = num_chunks * num_frames_per_chunk * num_clusters;
  let mut clustered = vec![0.0f64; cs_size];
  let mut clustered_mask = vec![false; cs_size]; // true = valid (not NaN)

  for c in 0..num_chunks {
    for k_iter in 0..num_clusters_from_hard {
      let k = k_iter as i32;
      // Find speakers in this chunk assigned to cluster k.
      let speakers_in_k: Vec<usize> = hard_clusters[c]
        .iter()
        .enumerate()
        .filter_map(|(s, &kk)| (kk == k).then_some(s))
        .collect();
      if speakers_in_k.is_empty() {
        continue;
      }
      for f in 0..num_frames_per_chunk {
        let mut max_act = f64::NEG_INFINITY;
        for &s in &speakers_in_k {
          let v = segmentations[(c * num_frames_per_chunk + f) * num_speakers + s];
          if v > max_act {
            max_act = v;
          }
        }
        let cs_idx = (c * num_frames_per_chunk + f) * num_clusters + k_iter;
        clustered[cs_idx] = max_act;
        clustered_mask[cs_idx] = true;
      }
    }
  }
  // UNMATCHED speakers (k == -2) skipped — clustered_mask stays false
  // for those (cluster, frame) cells and aggregate treats them as NaN
  // (skipped contribution).

  // ── Stage 2: aggregate(skip_average=True) ──────────────────────
  // Pyannote's overlap-add: for each chunk c, find start_frame =
  // closest_frame(chunk_start_time + 0.5 * frame_duration), then
  //   aggregated[start_frame .. start_frame + npc, k] += clustered * mask
  // hamming + warm_up are all-ones in cluster_vbx's call path.
  let mut aggregated = vec![0.0f32; num_output_frames * num_clusters];
  let mut agg_mask = vec![false; num_output_frames * num_clusters];

  for c in 0..num_chunks {
    let chunk_start_time = chunks_sw.start + (c as f64) * chunks_sw.step;
    let center_offset = 0.5 * frames_sw.duration;
    let start_frame = frames_sw.closest_frame(chunk_start_time + center_offset);
    if start_frame < 0 {
      // Pyannote produces frames at non-negative indices; if a chunk
      // starts before the first output frame, clip its leading
      // frames out. start_frame_clamp = 0; clip leading
      // (-start_frame) of the chunk.
    }
    for f in 0..num_frames_per_chunk {
      let out_f = start_frame + f as i64;
      if out_f < 0 || out_f as usize >= num_output_frames {
        continue;
      }
      let out_f = out_f as usize;
      for k in 0..num_clusters_from_hard {
        let cs_idx = (c * num_frames_per_chunk + f) * num_clusters + k;
        if !clustered_mask[cs_idx] {
          continue;
        }
        let v = clustered[cs_idx] as f32;
        let agg_idx = out_f * num_clusters + k;
        aggregated[agg_idx] += v;
        agg_mask[agg_idx] = true;
      }
    }
  }
  // Cells that never received a contribution → leave as 0.0
  // (pyannote uses `missing=0.0` for to_diarization).
  for (i, &m) in agg_mask.iter().enumerate() {
    if !m {
      aggregated[i] = 0.0;
    }
  }

  // ── Stage 3: top-`count[t]` binarize per output frame ──────────
  let mut out = vec![0.0f32; num_output_frames * num_clusters];
  for (t, &c_byte) in count.iter().enumerate().take(num_output_frames) {
    let c_count = c_byte as usize;
    if c_count == 0 {
      continue;
    }
    // Sort cluster indices by descending activation at frame t.
    let row_start = t * num_clusters;
    let mut sorted: Vec<usize> = (0..num_clusters).collect();
    sorted.sort_by(|&a, &b| {
      let va = aggregated[row_start + a];
      let vb = aggregated[row_start + b];
      // Descending: vb.partial_cmp(va). Stable tie-break by index
      // (sort_by is stable since 1.20).
      vb.partial_cmp(&va).unwrap_or(std::cmp::Ordering::Equal)
    });
    for &k in sorted.iter().take(c_count) {
      out[row_start + k] = 1.0;
    }
  }

  // Reference UNMATCHED so the import isn't dead code.
  let _ = UNMATCHED;
  Ok(out)
}
