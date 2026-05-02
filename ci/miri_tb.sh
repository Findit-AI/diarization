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
# 1. Test filter `ops::` — every `unsafe` block in this crate's
#    production source lives under `src/ops/` (verified by
#    `grep -rn "unsafe " src/ --include='*.rs'`). The rest is safe
#    Rust, so miri adds no signal there.
#
# 2. `--cfg diarization_force_scalar` — miri can't evaluate foreign
#    LLVM intrinsics like `llvm.aarch64.neon.faddv.f64.v2f64` (NEON)
#    or `llvm.x86.avx2.*`. Without this cfg, the dispatcher hits its
#    arch-specific path and miri errors `unsupported operation`. With
#    this cfg every `*_available()` helper short-circuits to `false`
#    and the dispatcher falls through to the scalar reference. The
#    intrinsic paths themselves are exercised natively under SDE
#    (AVX2 and AVX-512 — see ci/sde_avx2.sh, ci/sde_avx512.sh) and on
#    the regular test job (NEON on aarch64 hosts; AVX2 on Linux x86
#    hosts that have it).
#
# 3. `--no-default-features` — skips `ort` (the default feature) and
#    its `ort-sys` C++ runtime, plus the transitive
#    `kaldi-native-fbank` C bindings. miri can't execute foreign
#    function calls anyway, so these would error before our test
#    code runs.
#
# — pattern mirrors siglip2's miri job.
export RUSTFLAGS="${RUSTFLAGS:-} --cfg diarization_force_scalar"
cargo miri test --lib --target "$TARGET" --no-default-features ops::
