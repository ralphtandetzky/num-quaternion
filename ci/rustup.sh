#!/bin/sh
# Use rustup to locally run the same suite of tests as .github/workflows/
# (You should first install/update all of the versions below.)

set -ex

# semver checks
cargo install cargo-semver-checks --locked
cargo semver-checks                     # all stable features
cargo semver-checks --default-features  # all default features

# clippy checks
rustup component add clippy
cargo clippy --all-features

# formatting checks
rustup component add rustfmt
cargo fmt --check

ci=$(dirname $0)
rustup update "1.63.0"
rustup run "1.63.0" "$ci/test_full.sh"
rustup update "stable"
rustup run "stable" "$ci/test_full.sh --run_tests"
