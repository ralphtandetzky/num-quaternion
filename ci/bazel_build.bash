#!/bin/bash
# This script is used to build all Bazel targets in the C++ benchmarks
# directory.

# Prerequisites:
# 1. Git must be installed and available in the system's PATH.
# 2. Bazel must be installed and available in the system's PATH.

set -ex

# Navigate to the workspace directory
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)/benches/c++" || exit

# Build all Bazel targets
USE_BAZEL_VERSION=8.3.1 bazel build //...

# Run clang-tidy using Bazel
USE_BAZEL_VERSION=8.3.1 bazel build //... --config=clang-tidy
