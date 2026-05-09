#!/bin/bash
set -e

if [ -z "$1" ]; then
  echo "Error: TARGET is not provided"
  exit 1
fi

TARGET="$1"

# Install cross-compilation toolchain on Linux
if [ "$(uname)" = "Linux" ]; then
  case "$TARGET" in
    aarch64-unknown-linux-gnu)
      sudo apt-get update && sudo apt-get install -y gcc-aarch64-linux-gnu
      ;;
    i686-unknown-linux-gnu)
      sudo apt-get update && sudo apt-get install -y gcc-multilib
      ;;
    powerpc64-unknown-linux-gnu)
      sudo apt-get update && sudo apt-get install -y gcc-powerpc64-linux-gnu
      ;;
    s390x-unknown-linux-gnu)
      sudo apt-get update && sudo apt-get install -y gcc-s390x-linux-gnu
      ;;
    riscv64gc-unknown-linux-gnu)
      sudo apt-get update && sudo apt-get install -y gcc-riscv64-linux-gnu
      ;;
  esac
fi

rustup toolchain install nightly --component miri
rustup override set nightly
cargo miri setup

export MIRIFLAGS="-Zmiri-strict-provenance -Zmiri-disable-isolation -Zmiri-symbolic-alignment-check -Zmiri-tree-borrows"

# Scope and configuration:
#
# 1. Test filters `ops::` and `embed::fbank::tests` — every `unsafe`
#    block in this crate's production source lives under either
#    `src/ops/` (cluster + embed numerical primitives) or
#    `src/embed/fbank.rs` (NEON/SSE2/AVX2/AVX-512F window-mul,
#    power-spectrum, dot kernels added with the torchaudio fbank
#    port). The rest is safe Rust, so miri adds no signal there.
#
# 2. `--cfg diarization_force_scalar` — miri can't evaluate foreign
#    LLVM intrinsics like `llvm.aarch64.neon.faddv.f64.v2f64` (NEON)
#    or `llvm.x86.avx2.*`. Without this cfg, the dispatcher hits its
#    arch-specific path and miri errors `unsupported operation`. With
#    this cfg every `*_available()` helper short-circuits to `false`
#    and the dispatcher falls through to the scalar reference. Inside
#    `src/embed/fbank.rs` the same `if cfg!(diarization_force_scalar)`
#    guard at the top of `fma_dot_f32_to_f64` / `apply_window_inplace`
#    / `power_spectrum` ensures miri sees the scalar path. The
#    intrinsic paths themselves are exercised natively under SDE
#    (AVX2 and AVX-512 — see ci/sde_avx2.sh, ci/sde_avx512.sh) and on
#    the regular test job (NEON on aarch64 hosts; AVX2 on Linux x86
#    hosts that have it). Per-backend direct unsafe-call tests in
#    `embed::fbank::tests` (e.g. `dot_neon_agrees_with_scalar_directly`)
#    are filtered out under force_scalar because they call the unsafe
#    SIMD kernels directly — miri only exercises the dispatcher /
#    scratch / scalar paths.
#
# 3. `--no-default-features` — skips `ort` (the default feature) and
#    its `ort-sys` C++ runtime. miri can't execute foreign function
#    calls anyway, so this would error before our test code runs.
#
# — pattern mirrors siglip2's miri job.
export RUSTFLAGS="${RUSTFLAGS:-} --cfg diarization_force_scalar"
# Explicit allowlist for `embed::fbank::tests` rather than the whole
# module: realfft (`= 3` with default features) pulls rustfft, whose
# default planners select NEON/SSE/AVX kernels at runtime. Miri can't
# evaluate those intrinsics. The tests in the allowlist below DO NOT
# call into the FFT path under force-scalar — they exercise the
# scalar dot/window/power/log paths, length-mismatch guards, NaN
# propagation, and TLS scratch capacity bookkeeping. The
# `caps_oversized_scratch_capacity` test does call
# `compute_full_fbank` once with a single-frame input (one size-512
# FFT) — Miri tolerates that at the time of writing, but if rustfft
# regresses on Miri-supported intrinsics this is the test to drop.
cargo miri test \
  --lib --target "$TARGET" --no-default-features \
  -- \
  ops:: \
  embed::fbank::tests::dot_panics_on_length_mismatch_in_release \
  embed::fbank::tests::window_panics_on_length_mismatch_in_release \
  embed::fbank::tests::power_panics_on_length_mismatch_in_release \
  embed::fbank::tests::dot_kernels_agree_with_scalar \
  embed::fbank::tests::nan_propagates_through_log_floor \
  embed::fbank::tests::force_scalar_cfg_routes_through_scalar_when_set \
  embed::fbank::tests::shrink_before_resize_drops_oversized_when_call_small \
  embed::fbank::tests::shrink_before_resize_keeps_buffer_when_call_huge \
  embed::fbank::tests::shrink_before_resize_leaves_bounded_buffer \
  embed::fbank::tests::shrink_after_loop_drops_oversized \
  embed::fbank::tests::shrink_after_loop_keeps_bounded_buffer
