//! Streaming VAD-gated diarization pipeline implementation.

use std::collections::VecDeque;

use crate::{
  embed::{EMBEDDING_DIM, EmbedModel, Embedding},
  offline::{OfflineOutput, OwnedDiarizationPipeline, OwnedPipelineConfig},
  plda::PldaTransform,
  segment::SegmentModel,
};
use silero::{
  Session as VadSession, SpeechOptions, SpeechSegment, SpeechSegmenter, StreamState as VadStream,
};

/// Span emitted by [`StreamingDiarizationPipeline`] for a closed
/// voice range.
#[derive(Debug, Clone)]
pub struct StreamingDiarizedSpan {
  /// Start sample (absolute, relative to the start of the audio
  /// stream).
  pub start_sample: u64,
  /// End sample (absolute).
  pub end_sample: u64,
  /// Globally-tracked speaker id (matched across voice ranges via
  /// the centroid bank).
  pub speaker_id: u32,
  /// `true` iff this is the first time `speaker_id` has been emitted
  /// in the current session (post `new`).
  pub is_new_speaker: bool,
  /// Speech-segment id this span belongs to (monotonic per session).
  pub speech_segment_id: u32,
}

/// Configuration for [`StreamingDiarizationPipeline`].
#[derive(Debug, Clone)]
pub struct StreamingPipelineConfig {
  /// Silero VAD options (default = silero v5 16 kHz defaults).
  pub vad: SpeechOptions,
  /// dia offline pipeline options applied per voice range.
  pub diarization: OwnedPipelineConfig,
  /// Speaker-bank cosine threshold for cross-range identity match.
  /// Speakers with cosine similarity `>= threshold` to a bank
  /// centroid are considered the same speaker. Default `0.45` is
  /// tuned for VAD-gated diarization where each voice range's
  /// per-cluster centroid is a noisy estimate (gathered from a
  /// single voice range, possibly only a few seconds of speech).
  /// Tighter thresholds (0.6+) over-fragment speakers across ranges;
  /// looser thresholds (0.3) merge distinct speakers. Tune up if
  /// you see false speaker merges, down if you see speaker
  /// fragmentation.
  pub speaker_match_threshold: f32,
}

impl Default for StreamingPipelineConfig {
  fn default() -> Self {
    // Silero defaults are tuned for utterance-level VAD
    // (`min_silence_duration = 100 ms`), which over-segments a
    // multi-speaker conversation into single utterances. Each
    // utterance is too short for the 10-second segmentation window
    // pyannote requires for meaningful diarization. Override to
    // coalesce utterances into multi-speaker conversation blocks:
    // 1.5 s of silence required to close a voice range, with a
    // hard 60 s cap so very long ranges still get diarized in
    // bounded latency.
    let vad = SpeechOptions::default()
      .with_min_silence_duration(std::time::Duration::from_millis(1500))
      .with_min_speech_duration(std::time::Duration::from_millis(250))
      .with_max_speech_duration(std::time::Duration::from_secs(60));
    Self {
      vad,
      diarization: OwnedPipelineConfig::default(),
      // Conservative default; per-range centroids are noisy
      // estimates from short speech segments. Tune empirically per
      // deployment audio.
      speaker_match_threshold: 0.6,
    }
  }
}

/// Errors from the streaming pipeline.
#[derive(Debug, thiserror::Error)]
pub enum StreamingError {
  #[error("streaming: vad: {0}")]
  Vad(#[from] silero::Error),
  #[error("streaming: offline: {0}")]
  Offline(#[from] crate::offline::Error),
  #[error("streaming: shape: {0}")]
  Shape(&'static str),
}

/// One entry in the session-wide speaker centroid bank.
#[derive(Debug, Clone)]
struct SpeakerBankEntry {
  speaker_id: u32,
  /// Running L2-normalized centroid (EMA-style).
  centroid: [f32; EMBEDDING_DIM],
  /// Number of voice ranges this speaker has been observed in;
  /// used to weight the EMA update (more samples → slower drift).
  observation_count: u32,
}

/// Streaming VAD-gated diarization pipeline. See
/// [`crate::streaming`] module docs for the full description.
pub struct StreamingDiarizationPipeline {
  config: StreamingPipelineConfig,
  vad_session: VadSession,
  vad_stream: VadStream,
  vad_segmenter: SpeechSegmenter,
  diarizer: OwnedDiarizationPipeline,
  /// Rolling sample buffer indexed by absolute sample position.
  /// Must contain at least every sample from
  /// `(audio_base..audio_base + audio_buffer.len())`. Trim policy:
  /// drop everything before the most recent VAD-emitted segment's
  /// start. Voice ranges are bounded so this stays small.
  audio_buffer: VecDeque<f32>,
  /// Absolute sample index of `audio_buffer[0]`.
  audio_base: u64,
  /// Total samples ever pushed via [`push_audio`](Self::push_audio).
  total_samples_pushed: u64,
  /// Session-wide speaker centroid bank.
  speaker_bank: Vec<SpeakerBankEntry>,
  next_speaker_id: u32,
  next_speech_segment_id: u32,
}

impl StreamingDiarizationPipeline {
  /// Construct with default config.
  pub fn new(vad_session: VadSession) -> Self {
    Self::with_config(vad_session, StreamingPipelineConfig::default())
  }

  /// Construct with explicit config.
  pub fn with_config(vad_session: VadSession, config: StreamingPipelineConfig) -> Self {
    let vad_stream = VadStream::new(config.vad.sample_rate());
    let vad_segmenter = SpeechSegmenter::new(config.vad.clone());
    Self {
      diarizer: OwnedDiarizationPipeline::with_config(config.diarization),
      config,
      vad_session,
      vad_stream,
      vad_segmenter,
      audio_buffer: VecDeque::new(),
      audio_base: 0,
      total_samples_pushed: 0,
      speaker_bank: Vec::new(),
      next_speaker_id: 0,
      next_speech_segment_id: 0,
    }
  }

  /// Borrow the configuration.
  pub fn config(&self) -> &StreamingPipelineConfig {
    &self.config
  }

  /// Number of distinct speakers tracked so far.
  pub fn num_speakers(&self) -> usize {
    self.speaker_bank.len()
  }

  /// Push raw 16 kHz mono samples into the pipeline. The silero VAD
  /// processes them in fixed-size chunks; when a voice range closes,
  /// it's diarized end-to-end via the offline pipeline and one
  /// [`StreamingDiarizedSpan`] is emitted per detected speaker turn
  /// within the range.
  pub fn push_audio<F>(
    &mut self,
    seg_model: &mut SegmentModel,
    embed_model: &mut EmbedModel,
    plda: &PldaTransform,
    samples: &[f32],
    mut emit: F,
  ) -> Result<(), StreamingError>
  where
    F: FnMut(StreamingDiarizedSpan),
  {
    self.audio_buffer.extend(samples.iter().copied());
    self.total_samples_pushed += samples.len() as u64;

    // Collect voice segments emitted by silero this batch. Cannot
    // process them inline because the segmenter callback does not
    // own access to seg_model/embed_model.
    let mut emitted_segments: Vec<SpeechSegment> = Vec::new();
    self.vad_segmenter.process_samples(
      &mut self.vad_session,
      &mut self.vad_stream,
      samples,
      |s| {
        emitted_segments.push(s);
      },
    )?;

    for seg in &emitted_segments {
      self.diarize_segment(seg_model, embed_model, plda, seg, &mut emit)?;
    }

    // Trim the audio buffer up to the latest emitted segment's end.
    // After this, the buffer only retains samples that may still
    // belong to an unclosed in-progress voice range.
    if let Some(last) = emitted_segments.last() {
      self.trim_audio_to(last.end_sample());
    }

    Ok(())
  }

  /// Flush the silero stream and finalize any trailing voice range.
  pub fn finish<F>(
    &mut self,
    seg_model: &mut SegmentModel,
    embed_model: &mut EmbedModel,
    plda: &PldaTransform,
    mut emit: F,
  ) -> Result<(), StreamingError>
  where
    F: FnMut(StreamingDiarizedSpan),
  {
    let mut trailing: Vec<SpeechSegment> = Vec::new();
    self
      .vad_segmenter
      .finish_stream(&mut self.vad_session, &mut self.vad_stream, |s| {
        trailing.push(s);
      })?;
    for seg in &trailing {
      self.diarize_segment(seg_model, embed_model, plda, seg, &mut emit)?;
    }
    Ok(())
  }

  fn diarize_segment<F>(
    &mut self,
    seg_model: &mut SegmentModel,
    embed_model: &mut EmbedModel,
    plda: &PldaTransform,
    seg: &SpeechSegment,
    emit: &mut F,
  ) -> Result<(), StreamingError>
  where
    F: FnMut(StreamingDiarizedSpan),
  {
    let start = seg.start_sample();
    let end = seg.end_sample();
    if end <= start {
      return Ok(());
    }
    // Slice the buffered audio for this voice range.
    let slice = self.slice_audio(start, end)?;

    let speech_segment_id = self.next_speech_segment_id;
    self.next_speech_segment_id = self.next_speech_segment_id.wrapping_add(1);

    // Run offline diarization on the voice range.
    let out = match self.diarizer.run(seg_model, embed_model, plda, &slice) {
      Ok(o) => o,
      Err(crate::offline::Error::Shape(_)) => {
        // Range too short / shape mismatch (e.g. < 1 chunk worth of
        // audio reached the segmentation model). Treat as a single-
        // speaker span using a degenerate centroid: carry the whole
        // range as one span, speaker matched against the bank.
        return self.emit_short_range_as_single_span(start, end, speech_segment_id, emit);
      }
      Err(e) => return Err(e.into()),
    };

    // For each cluster in the range, compute its centroid from the
    // offline pipeline's `discrete_diarization` (which marks per-
    // frame per-cluster activity). Match against the session-wide
    // speaker bank; assign global speaker_id.
    self.emit_offline_output_with_global_ids(
      &out,
      &slice,
      start,
      embed_model,
      speech_segment_id,
      emit,
    )?;
    Ok(())
  }

  fn slice_audio(&self, start: u64, end: u64) -> Result<Vec<f32>, StreamingError> {
    if start < self.audio_base {
      return Err(StreamingError::Shape(
        "voice segment starts before audio buffer base — trim policy violation",
      ));
    }
    let offset = (start - self.audio_base) as usize;
    let want_len = (end - start) as usize;
    // Silero's `speech_pad` extends segment END by ~30 ms past the
    // last consumed sample, so `end - audio_base` can momentarily
    // exceed `audio_buffer.len()`. Treat this as a benign overrun:
    // clamp to what's actually buffered. The trailing silence
    // padding doesn't carry diarizable signal anyway.
    let avail = self.audio_buffer.len().saturating_sub(offset);
    let len = want_len.min(avail);
    Ok(
      self
        .audio_buffer
        .range(offset..offset + len)
        .copied()
        .collect(),
    )
  }

  fn trim_audio_to(&mut self, sample: u64) {
    if sample <= self.audio_base {
      return;
    }
    let drop_n = (sample - self.audio_base) as usize;
    if drop_n >= self.audio_buffer.len() {
      self.audio_buffer.clear();
      self.audio_base = sample;
    } else {
      self.audio_buffer.drain(..drop_n);
      self.audio_base = sample;
    }
  }

  fn emit_short_range_as_single_span<F>(
    &mut self,
    start: u64,
    end: u64,
    speech_segment_id: u32,
    emit: &mut F,
  ) -> Result<(), StreamingError>
  where
    F: FnMut(StreamingDiarizedSpan),
  {
    // No embedding to match — give it the most recently observed
    // speaker (or speaker 0 if none). This is a degenerate
    // fallback for sub-chunk-length voice ranges.
    let speaker_id = self.speaker_bank.last().map_or(0, |e| e.speaker_id);
    let is_new_speaker = self.speaker_bank.is_empty();
    if is_new_speaker {
      self.speaker_bank.push(SpeakerBankEntry {
        speaker_id: 0,
        centroid: [0.0; EMBEDDING_DIM],
        observation_count: 1,
      });
      self.next_speaker_id = 1;
    }
    emit(StreamingDiarizedSpan {
      start_sample: start,
      end_sample: end,
      speaker_id,
      is_new_speaker,
      speech_segment_id,
    });
    Ok(())
  }

  fn emit_offline_output_with_global_ids<F>(
    &mut self,
    out: &OfflineOutput,
    range_samples: &[f32],
    range_start: u64,
    embed_model: &mut EmbedModel,
    speech_segment_id: u32,
    emit: &mut F,
  ) -> Result<(), StreamingError>
  where
    F: FnMut(StreamingDiarizedSpan),
  {
    // Group spans by cluster; each cluster gets one centroid.
    use std::collections::HashMap;
    let mut spans_by_cluster: HashMap<usize, Vec<&crate::reconstruct::RttmSpan>> = HashMap::new();
    for span in &out.spans {
      spans_by_cluster.entry(span.cluster).or_default().push(span);
    }

    // Compute one centroid per cluster by gathering per-cluster
    // sample ranges and running `embed` once per cluster.
    let sr = silero::SampleRate::Rate16k.hz() as f64;
    let mut cluster_to_global: HashMap<usize, (u32, bool)> = HashMap::new();
    for (cluster, spans) in &spans_by_cluster {
      // Build a keep-mask covering all this cluster's spans.
      let mut keep = vec![false; range_samples.len()];
      for span in spans {
        let s_lo = (span.start * sr).max(0.0) as usize;
        let s_hi = ((span.start + span.duration) * sr) as usize;
        let lo = s_lo.min(range_samples.len());
        let hi = s_hi.min(range_samples.len());
        for v in &mut keep[lo..hi] {
          *v = true;
        }
      }
      // Try to embed; if too little gathered audio, fall back to
      // using the last bank entry's centroid (or a zero centroid).
      let embedding = match embed_model.embed_masked(range_samples, &keep) {
        Ok(r) => Some(*r.embedding()),
        Err(_) => None,
      };
      let (speaker_id, is_new_speaker) = match embedding {
        Some(e) => self.match_or_insert(&e),
        None => {
          // No embedding → use last speaker if any.
          let last = self.speaker_bank.last().map(|x| x.speaker_id).unwrap_or(0);
          (last, self.speaker_bank.is_empty())
        }
      };
      cluster_to_global.insert(*cluster, (speaker_id, is_new_speaker));
    }

    // Emit one span per (cluster, span), in start-time order.
    let mut sorted: Vec<&crate::reconstruct::RttmSpan> = out.spans.iter().collect();
    sorted.sort_by(|a, b| {
      a.start
        .partial_cmp(&b.start)
        .unwrap_or(std::cmp::Ordering::Equal)
    });
    for span in sorted {
      let &(speaker_id, is_new_speaker) = cluster_to_global
        .get(&span.cluster)
        .ok_or(StreamingError::Shape("cluster missing from global map"))?;
      let span_start_abs = range_start + (span.start * sr) as u64;
      let span_end_abs = range_start + ((span.start + span.duration) * sr) as u64;
      emit(StreamingDiarizedSpan {
        start_sample: span_start_abs,
        end_sample: span_end_abs,
        speaker_id,
        is_new_speaker,
        speech_segment_id,
      });
    }
    Ok(())
  }

  fn match_or_insert(&mut self, embedding: &Embedding) -> (u32, bool) {
    let arr = embedding.as_array();
    // Find best cosine similarity against the bank (centroids are
    // L2-normalized, embedding is L2-normalized, so dot = cos).
    let mut best_idx: Option<usize> = None;
    let mut best_sim = self.config.speaker_match_threshold;
    for (i, e) in self.speaker_bank.iter().enumerate() {
      let sim: f32 = arr.iter().zip(e.centroid.iter()).map(|(a, b)| a * b).sum();
      if sim > best_sim {
        best_sim = sim;
        best_idx = Some(i);
      }
    }
    if let Some(idx) = best_idx {
      // Update centroid via EMA: w = 1 / (n + 1).
      let n = self.speaker_bank[idx].observation_count.max(1) as f32;
      let alpha = 1.0 / (n + 1.0);
      let centroid = &mut self.speaker_bank[idx].centroid;
      for (c, &v) in centroid.iter_mut().zip(arr.iter()) {
        *c = (1.0 - alpha) * *c + alpha * v;
      }
      // Re-normalize.
      let norm: f32 = centroid.iter().map(|v| v * v).sum::<f32>().sqrt();
      if norm > 1e-6 {
        for c in centroid.iter_mut() {
          *c /= norm;
        }
      }
      self.speaker_bank[idx].observation_count =
        self.speaker_bank[idx].observation_count.saturating_add(1);
      (self.speaker_bank[idx].speaker_id, false)
    } else {
      let speaker_id = self.next_speaker_id;
      self.next_speaker_id = self.next_speaker_id.wrapping_add(1);
      let mut centroid = [0.0_f32; EMBEDDING_DIM];
      centroid.copy_from_slice(arr);
      self.speaker_bank.push(SpeakerBankEntry {
        speaker_id,
        centroid,
        observation_count: 1,
      });
      (speaker_id, true)
    }
  }
}
