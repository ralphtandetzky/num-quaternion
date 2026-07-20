#!/bin/bash
# This script is used to build all Bazel targets in the C++ benchmarks
# directory.

# Prerequisites:
# 1. Git must be installed and available in the system's PATH.
# 2. Bazelisk must be installed and available as "bazel" in the system's PATH.

set -ex

# Navigate to the workspace directory
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)/benches/c++" || exit

# Build all Bazel targets
bazel build //...

# Run clang-tidy using Bazel
bazel build //... --config=clang-tidy
