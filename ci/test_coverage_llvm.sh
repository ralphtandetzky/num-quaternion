#!/usr/bin/env bash

# This script runs the tests with coverage instrumentation and shows the coverage report.
#
# Usage:
#
# - Run the tests with coverage instrumentation and show the coverage report:
#     ./ci/test_coverage_llvm.sh
# - Run the tests with coverage instrumentation and show only the uncovered lines:
#     ./ci/test_coverage_llvm.sh --filter
# - Run the tests with coverage instrumentation and check that the number of uncovered lines is equal to 5:
#     ./ci/test_coverage_llvm.sh --check
#
# It requires the following tools to be installed:
#
# - llvm-14
# - rustup
# - coreutils, findutils, less which are usually installed by default on Linux.

set -e

# Currently, we need a full clean to get the correct `newest_file` later on
cargo clean

# Run the tests with coverage instrumentation
RUSTFLAGS="-C instrument-coverage" rustup run 1.66.0 cargo t --all-features

# Merge the coverage data
llvm-profdata-14 merge default.profraw -o num-quaternion.profdata

# Find the newest file matching the pattern
newest_file=$(find target/debug/deps -name 'num_quaternion-*' -type f | grep -E 'num_quaternion-[0-9a-f]+$' | sort -r | head -n 1)

if [[ "$1" == "--filter" ]]; then
    # Show only the uncovered lines
    llvm-cov-14 show -Xdemangler=rustfilt -instr-profile=num-quaternion.profdata --use-color \
                     --object "$newest_file" --ignore-filename-regex=/.cargo/registry \
                     | GREP_COLORS='mt=00' grep --color=always -P "(/.*\.rs:$|^ *\d+?\| *0\|)" \
                     | less -R
elif [[ "$1" == "--check" ]]; then
    # Check that the number of uncovered lines is equal to 5
    line_count=$(llvm-cov-14 show -Xdemangler=rustfilt -instr-profile=num-quaternion.profdata --use-color \
                                  --object "$newest_file" --ignore-filename-regex=/.cargo/registry \
                                  | grep -P "^ *\d+?\| *0\|" | wc -l)
    if [[ "$line_count" -ne 5 ]]; then
        echo "Error: The number of uncovered lines ($line_count) is not equal to 5."
        exit 1
    fi
else
    # Show the complete coverage report
    llvm-cov-14 show -Xdemangler=rustfilt -instr-profile=num-quaternion.profdata --use-color \
                     --object "$newest_file" --ignore-filename-regex=/.cargo/registry \
                     | less -R
fi
