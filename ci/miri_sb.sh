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

export MIRIFLAGS="-Zmiri-strict-provenance -Zmiri-disable-isolation -Zmiri-symbolic-alignment-check"

# Same scope and configuration as `miri_tb.sh`: SIMD-only test filter
# (`ops::`), scalar dispatcher forced via `diarization_force_scalar`
# (miri can't evaluate intrinsics), `--no-default-features` (skips ort
# C++ runtime that miri can't FFI-call). See `miri_tb.sh` for the full
# rationale. Codex CI sweep.
export RUSTFLAGS="${RUSTFLAGS:-} --cfg diarization_force_scalar"
cargo miri test --lib --target "$TARGET" --no-default-features ops::
