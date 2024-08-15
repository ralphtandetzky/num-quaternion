#!/usr/bin/env bash

set -e

# Install cargo tarpaulin if not already installed
cargo install cargo-tarpaulin

# Generate test coverage data
cargo tarpaulin --engine Llvm --fail-under 80 --all-features
