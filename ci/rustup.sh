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
for version in stable 1.60.0; do
    rustup update "$version"
    rustup run "$version" "$ci/test_full.sh"
done
