#!/bin/bash
set -ex

export ASAN_OPTIONS="detect_odr_violation=0 detect_leaks=0"

TARGET="x86_64-unknown-linux-gnu"

# Scope: SIMD modules — `src/ops/` (cluster/embed primitives) and
# `src/embed/fbank.rs` (the in-place fbank kernel).
#
# Every `unsafe` block in this crate's production source is under
# either `src/ops/` (dispatchers + arch::* kernels) or
# `src/embed/fbank.rs` (the NEON/SSE2/AVX2/AVX-512F window-mul,
# power-spectrum, and dot kernels added with the torchaudio fbank
# port). Both run unchecked raw-pointer vector loads behind
# `unsafe fn`, so ASAN/MSAN/LSAN coverage is mandatory before we ship.
# The rest of the codebase is safe Rust and adds no signal here.
#
# `--no-default-features` skips `ort` (the default feature). `ort`
# pulls C/C++ FFI (ort-sys) and `kaldi-native-fbank` (also C bindings
# via the dev-dep transitive graph). Neither is sanitizer-instrumented,
# so MSAN reports `use-of-uninitialized-value` inside them on every run.
# Not our bug, not fixable in our code; scoping to `ops::` skips them.
#
# This is the same pattern siglip2's CI uses for its SIMD-only sanitizer
# coverage.

# Run address sanitizer
RUSTFLAGS="-Z sanitizer=address" \
cargo test --lib --target "$TARGET" --no-default-features ops:: embed::fbank::tests

# Run leak sanitizer
RUSTFLAGS="-Z sanitizer=leak" \
cargo test --lib --target "$TARGET" --no-default-features ops:: embed::fbank::tests

# Run memory sanitizer (requires -Zbuild-std for instrumented std)
RUSTFLAGS="-Z sanitizer=memory" \
cargo -Zbuild-std test --lib --target "$TARGET" --no-default-features ops:: embed::fbank::tests

# Run thread sanitizer (requires -Zbuild-std for instrumented std).
# Note: `ops::*` has no concurrency primitives — TSAN is kept here for
# symmetry and to catch any future regression that introduces shared
# state. Cheap to run.
RUSTFLAGS="-Z sanitizer=thread" \
cargo -Zbuild-std test --lib --target "$TARGET" --no-default-features ops:: embed::fbank::tests
