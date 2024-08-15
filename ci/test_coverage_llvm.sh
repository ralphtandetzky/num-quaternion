#!/usr/bin/env bash

set -e

# Install cargo tarpaulin if not already installed
cargo clean

RUSTFLAGS="-C instrument-coverage" rustup run 1.61.0 cargo t
llvm-profdata merge default.profraw -o num-quaternion.profdata

# Find the newest file matching the pattern
newest_file=$(find target/debug/deps -name 'num_quaternion-*' -type f | grep -E 'num_quaternion-[0-9a-f]+$' | sort -r | head -n 1)

llvm-cov-14 show -Xdemangler=rustfilt -instr-profile=num-quaternion.profdata --use-color --object "$newest_file" --ignore-filename-regex=/.cargo/registry | less -R
