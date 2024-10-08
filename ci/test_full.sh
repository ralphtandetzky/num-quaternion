#!/usr/bin/env bash

set -e

CRATE=num-quaternion
MSRV=1.63  # minimum supported rust version

get_rust_version() {
  local array=($(rustc --version));
  echo "${array[1]}";
  return 0;
}
RUST_VERSION=$(get_rust_version)

check_version() {
  IFS=. read -ra rust <<< "$RUST_VERSION"
  IFS=. read -ra want <<< "$1"
  [[ "${rust[0]}" -gt "${want[0]}" ||
   ( "${rust[0]}" -eq "${want[0]}" &&
     "${rust[1]}" -ge "${want[1]}" )
  ]]
}

echo "Testing $CRATE on rustc $RUST_VERSION"
if ! check_version $MSRV ; then
  echo "The minimum for $CRATE is rustc $MSRV"
  exit 1
fi

FEATURES=(libm rand serde unstable)
echo "Testing supported features: ${FEATURES[*]}"

cargo generate-lockfile

set -x

# test the default
cargo build
if [[ "$*" == *--run_tests* ]]; then
  cargo test --lib
fi

# test `no_std`
cargo build --no-default-features
if [[ "$*" == *--run_tests* ]]; then
  cargo test --no-default-features
fi

# test each isolated feature, with and without std
for feature in ${FEATURES[*]}; do
  cargo build --no-default-features --features="std $feature"
  if [[ "$*" == *--run_tests* ]]; then
    cargo test --no-default-features --features="std $feature"
  fi

  cargo build --no-default-features --features="$feature"
  if [[ "$*" == *--run_tests* ]]; then
    cargo test --no-default-features --features="$feature"
  fi
done

# test all supported features, with and without std
cargo build --features="std ${FEATURES[*]}"
if [[ "$*" == *--run_tests* ]]; then
  cargo test --features="std ${FEATURES[*]}"
fi

cargo build --features="${FEATURES[*]}"
if [[ "$*" == *--run_tests* ]]; then
  cargo test --features="${FEATURES[*]}"
fi
