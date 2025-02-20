#!/bin/bash
# This script is used to build all Bazel targets in the C++ benchmarks
# directory.

# Prerequisites:
# 1. Git must be installed and available in the system's PATH.
# 2. Bazel must be installed and available in the system's PATH.

set -ex

# Save the current working directory
original_dir=$(pwd)

# Function to restore the original working directory
restore_dir() {
    cd "$original_dir"
}

# Set trap to restore the original directory on exit
trap restore_dir EXIT

# Navigate to the workspace directory
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)/benches/c++" || exit

# Build all Bazel targets
USE_BAZEL_VERSION=7.2.1 bazel build //...
