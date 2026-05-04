# Design: mmap-spill for large clustering / reconstruction allocations

**Status:** draft
**Date:** 2026-05-04
**Related issues / commits:** rounds 26-27 of adversarial review (`MAX_AHC_TRAIN`, `MAX_RECONSTRUCT_GRID_CELLS`).

## Problem

Several `Result`-returning public APIs in `dia` allocate scratch buffers proportional to input size. Past a threshold the allocations either OOM-abort the process or blow the `Vec<f64>` capacity-overflow cliff (`n * 8 > isize::MAX`). Rounds 21-27 of adversarial review added typed `*TooLarge` errors that *reject* inputs past the threshold; this doc plans the next step — *spill* the worst offenders onto disk via `mmap` so the same inputs succeed at the cost of paging instead of failing outright.

## Audit: where memory grows

The table below lists every allocation in the production diarization path that scales with input size, together with the input quantity it scales with, the realistic size at the documented production scale (1 hour of conversational audio, ~10 000 active pairs after `filter_embeddings`), and the theoretical worst case under the current caps.

| # | Site | Shape | Cells / bytes at 1 h scale | At current cap |
|---|------|-------|----------------------------|----------------|
| 1 | `cluster/ahc/algo.rs:l2_normalize_to_row_major` (`Vec::with_capacity(n*d)`) | `num_train × embed_dim` f64 | 10 000 × 256 = 2.56M cells (~20 MB) | 16 000 × 256 = 4.1M cells (~33 MB) |
| 2 | `ops::pdist_euclidean` condensed output | `num_train * (num_train-1) / 2` f64 | ~50M cells (~400 MB) | ~128M cells (~1 GB) |
| 3 | `kodama::linkage` working memory | implementation-defined `O(N²)` | comparable to (2) | comparable to (2) |
| 4 | `pipeline::algo::build_qinit` (`q`) | `num_train × num_init` f64 | 10 000 × 15 = 150k cells (~1 MB) | 5M cells (~40 MB) |
| 5 | `pipeline::algo` `train_embeddings` matrix | `num_train × embed_dim` f64 | same as (1) | same as (1) |
| 6 | `cluster::vbx::vbx_iterate` `gamma`/`new_gamma`/`rho` | `num_train × plda_dim` f64 | 10 000 × 128 = 1.28M cells × 3 (~30 MB) | bounded by `MAX_QINIT_CELLS` |
| 7 | `cluster::vbx::vbx_iterate` `inv_l`/`alpha` (per-state) | `num_alive × plda_dim` f64 | 4 × 128 = tiny | tiny (`num_alive ≤ ~15`) |
| 8 | `pipeline::algo` `soft` (per-chunk dense matrices) | `num_chunks × num_speakers × num_alive` f64 | 3600 × 3 × 15 = 162k cells (~1.3 MB) | bounded by `MAX_AHC_TRAIN` |
| 9 | `reconstruct::algo` `clustered`/`clustered_mask` | `num_chunks × num_frames_per_chunk × num_clusters` f64 + bool | bounded by `MAX_RECONSTRUCT_GRID_CELLS = 1e8` (~800 MB f64) | same |
| 10 | `reconstruct::algo` `aggregated`/`agg_mask` | `num_output_frames × num_clusters` f32 + bool | bounded by `MAX_RECONSTRUCT_GRID_CELLS = 1e8` (~400 MB f32) | same |
| 11 | `aggregate::count` `out` (hamming) | `num_output_frames` f64 | bounded by `MAX_OUTPUT_FRAMES = 1e8` (~800 MB) | same |
| 12 | `aggregate::count` `aggregated` + `overlapping_count` (count_pyannote) | `num_output_frames` f64 × 2 | same as (11) ×2 | ~1.6 GB at cap |
| 13 | `cluster::spectral::build_affinity` `a` | `n × n` f64 (in `cluster_offline`) | bounded by `MAX_OFFLINE_INPUT = 1000` (8 MB) | same |
| 14 | `cluster::spectral` `sorted_vecs` | `n × n` f64 | same as (13) | same |
| 15 | `cluster::agglomerative` `d` (full `Vec<Vec<f32>>`) | `n × n` f32 (in `cluster_offline`) | bounded by `MAX_OFFLINE_INPUT = 1000` (4 MB) | same |
| 16 | `offline::algo` `embeddings` matrix | `num_chunks × num_speakers × EMBEDDING_DIM` f64 | 3600 × 3 × 256 = 2.76M cells (~22 MB) | bounded by `MAX_AHC_TRAIN` |
| 17 | `offline::algo` `post_plda` matrix | `num_train × plda_dim` f64 | same as (1) but `plda_dim = 128` (~10 MB) | bounded |

### Rankings

- **High risk** (>500 MB at production scale or under current cap): (2), (3), (12), and (9)+(10).
- **Medium risk** (100–500 MB): (1)+(5), (6), and (11) at peak.
- **Low risk / bounded** (<100 MB at cap): (4), (7), (8), (13)–(17).

## Design: spill-above-512 MB policy

For each high-risk allocation, replace the unconditional `Vec<f64>` with a small `enum` that picks one of two backends at construction time:

```rust
pub(crate) enum SpillBuf<T: Copy + Default> {
    Heap(Vec<T>),
    Mmap {
        // Owns the file (`tempfile::tempfile()`) so it auto-unlinks on
        // drop. Memory mapping is read-write, anonymous-on-disk.
        _file: std::fs::File,
        map: memmap2::MmapMut,
        len: usize,
        _phantom: PhantomData<T>,
    },
}

impl<T: Copy + Default> SpillBuf<T> {
    /// Allocate `n` cells. If `n * size_of::<T>() > MMAP_SPILL_THRESHOLD_BYTES`,
    /// back the buffer with an mmap'd tempfile; otherwise heap.
    pub fn new(n: usize) -> std::io::Result<Self> {
        let bytes = n.checked_mul(size_of::<T>()).expect("size overflow");
        if bytes > MMAP_SPILL_THRESHOLD_BYTES {
            let file = tempfile::tempfile()?;
            file.set_len(bytes as u64)?;
            // SAFETY: we own the file, no other process maps it.
            let map = unsafe { memmap2::MmapMut::map_mut(&file)? };
            Ok(Self::Mmap { _file: file, map, len: n, _phantom: PhantomData })
        } else {
            Ok(Self::Heap(vec![T::default(); n]))
        }
    }

    pub fn as_slice(&self) -> &[T] { /* … */ }
    pub fn as_mut_slice(&mut self) -> &mut [T] { /* … */ }
}
```

### Threshold

```rust
/// Allocations larger than this are mmap-backed onto an unlinked
/// tempfile rather than the process heap. Below this threshold the
/// heap path is faster and avoids a syscall, so we keep it.
///
/// 512 MB matches the OOM-cliff math: `Vec` capacity overflow hits
/// at `isize::MAX` (~9 EB on 64-bit), but real-world OOM-abort hits
/// the kernel's overcommit threshold somewhere between 1 GB and the
/// total RAM. Spilling at 512 MB keeps both halves of the policy
/// well under typical container memory budgets (1-2 GB).
pub const MMAP_SPILL_THRESHOLD_BYTES: usize = 512 * 1024 * 1024;
```

### Sites to migrate

In rough order of priority:

1. **`ops::pdist_euclidean` condensed output.** This is the single largest allocation in the cluster pipeline (1 GB at the cap). The output is read once by `kodama::linkage`, so spilling it to mmap is mostly the kernel paging it back in for sequential reads — well-suited to mmap.
2. **`reconstruct::algo` `clustered`/`clustered_mask`.** Both grow with `num_chunks * num_frames_per_chunk * num_clusters`. Hot-loop access pattern is `[c, f, k]` (sequential `f`, sparse `c`); sequential mmap reads.
3. **`reconstruct::algo` `aggregated`/`agg_mask`.** `num_output_frames × num_clusters`. Same access shape as above.
4. **`aggregate::count`'s `aggregated` + `overlapping_count`.** Grow with `num_output_frames`. Pure sequential write then read.
5. **`pipeline::algo` `train_embeddings`.** `num_train × embed_dim`. Less urgent — bounded by `MAX_AHC_TRAIN = 16k` and `embed_dim = 256` so cap is ~33 MB.

### Sites to leave on heap

Allocations that fit comfortably under the threshold even at the documented intended scale are kept on the heap:

- `build_qinit` (1 MB at scale, 40 MB at cap)
- `vbx_iterate` `gamma`/`new_gamma`/`rho` (~30 MB at scale)
- `cluster::spectral`/`agglomerative` matrices (bounded by `MAX_OFFLINE_INPUT = 1000`, all <8 MB)
- `cluster::ahc::l2_normalize_to_row_major` (~33 MB at cap)

### Adjusting the existing caps

With spill in place, the existing hard caps can be raised — they no longer represent OOM cliffs but soft "this is unusually large" limits. Proposed new values (post-spill):

| Cap | Pre-spill | Post-spill | Rationale |
|-----|-----------|------------|-----------|
| `MAX_AHC_TRAIN` | 16 000 (~1 GB pdist) | 32 000 (~4 GB pdist, mmap-spilled) | 4× headroom over documented 10k scale; `kodama::linkage` work is `O(N²)` time but mostly bound by cache lines. |
| `MAX_RECONSTRUCT_GRID_CELLS` | 1e8 (~800 MB) | 4e8 (~3.2 GB, mmap-spilled) | 4× headroom; trillions of frames is still nonsensical. |
| `MAX_OUTPUT_FRAMES` | 1e8 (~800 MB) | 4e8 (~3.2 GB, mmap-spilled) | matches above. |

The hard caps are still required even with spill — kernel-level mmap can fail on 32-bit address spaces, low-disk hosts, or sandboxes that block tempfile creation. The goal is to push the boundary out by 4×, not eliminate it.

## Cross-platform considerations

| OS | `tempfile::tempfile()` semantics | `memmap2` semantics |
|----|----------------------------------|----------------------|
| Linux | `open(O_TMPFILE)` then `unlink` — file is auto-removed even on crash. | `mmap(MAP_SHARED)` — kernel pages dirty cells back to disk. |
| macOS | Falls back to `/tmp/.tmpXXXXXX` + `unlink` — same effect. | `mmap(MAP_SHARED)` — same. |
| Windows | `CreateFile(FILE_FLAG_DELETE_ON_CLOSE)` — auto-removes. | `MapViewOfFile`; file-backed. |

All three platforms return a writeable mmap'd file that the OS auto-removes on drop or process exit. The Rust `tempfile` + `memmap2` crates handle the platform variance.

### Disk space considerations

A 1.6-hour conversation could require ~4 GB of tempfile disk space across the spilled allocations. We surface this in the public API:

- New `SpillError::TempfileAllocationFailed { requested: usize, source: io::Error }` — surfaces `ENOSPC` / `EACCES` failures from `tempfile::tempfile()`.
- The `try_*` API contract becomes "allocates from heap or spills to disk; surfaces a typed error if neither succeeds."

## Differential testing

The spill code path produces bit-equal results to the heap path. We add a differential test that runs the AHC pdist + linkage pipeline twice on the same input — once with `MMAP_SPILL_THRESHOLD_BYTES = usize::MAX` (forces heap) and once with `MMAP_SPILL_THRESHOLD_BYTES = 0` (forces mmap) — and asserts the outputs are bit-equal.

## Implementation phases

1. **Phase 1: SpillBuf type + tests.** Implement `SpillBuf<T>` in a new `crate::ops::spill` module. Cover the heap path, the mmap path, and the threshold-driven dispatch with unit tests.
2. **Phase 2: pdist migration.** Switch `ops::pdist_euclidean` to return `SpillBuf<f64>` instead of `Vec<f64>`. Update `kodama::linkage` call site to accept the slice view. Differential test against the pre-migration output.
3. **Phase 3: reconstruct migration.** Switch `reconstruct::algo`'s `clustered`/`aggregated`/`agg_mask` to `SpillBuf`.
4. **Phase 4: aggregate migration.** Switch `aggregate::count`'s `aggregated` + `overlapping_count`.
5. **Phase 5: cap relaxation.** Raise the hard caps to the post-spill values.
6. **Phase 6: docs.** Update `pipeline::Error::AhcTrainSizeAboveMax` etc. doc-comments to mention "or set up a smaller `MMAP_SPILL_THRESHOLD_BYTES`/disable spill via …".

Each phase ships independently behind the existing Result-API contract; no DER regression is acceptable, and the existing 10-fixture parity suite must remain bit-equal across phases.

## Open questions

1. **Should the threshold be configurable?** The docs propose a hardcoded `512 * 1024 * 1024`. Production deployments with constrained `/tmp` (Docker default 64 MB) might want to disable spill entirely; a `dia::set_spill_threshold(usize)` global or a per-call option could help.
2. **Should we cap spill disk usage?** `MAX_AHC_TRAIN = 32_000` post-spill = 4 GB pdist — that's a lot of `/tmp`. Cap `(num_train * (num_train-1) / 2 * 8) ≤ MAX_SPILL_BYTES` to keep total tempfile use bounded?
3. **Should mmap usage feature-flag itself?** Some embedded targets (no_std, embedded Linux) lack `tempfile`. A `feature = "mmap-spill"` (default-on) keeps the heap-only path available.
4. **Does `kodama::linkage` access pdist randomly or sequentially?** If random, mmap pays a paging cost; if sequential, mmap is essentially free. (Spot-check kodama source before committing.)

## Acceptance criteria

- [ ] `SpillBuf<T>` lands with unit tests covering both backends.
- [ ] `ops::pdist_euclidean` uses `SpillBuf` and the existing AHC parity tests still pass.
- [ ] DER baseline preserved across all 10 fixtures (pyannote 0.0037/0/0/0/0/0.0019, speakrs 0).
- [ ] `MAX_AHC_TRAIN` raised from 16 000 to 32 000.
- [ ] `MAX_RECONSTRUCT_GRID_CELLS` raised from 1e8 to 4e8.
- [ ] Differential test (heap vs mmap) added for at least the AHC pdist path.
- [ ] CI runs the heap path and the mmap path on the same fixture suite.
