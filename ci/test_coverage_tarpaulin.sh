#!/usr/bin/env bash

set -e

CRATE=num-quaternion
MSRV=1.61  # minimum supported rust version

# Install cargo tarpaulin if not already installed
cargo install cargo-tarpaulin

# Generate test coverage data
cargo tarpaulin --engine Llvm --fail-under 80 --all-features
