# Repository Instructions

`num-quaternion` is a Rust crate designed for robust, efficient and easy to
use quaternion arithmetic and operations.

## Repository Guidelines

- Document all public functions, parameters, types, fields, and modules thoroughly.
- Document preconditions, postconditions and invariants.
- Write tests based on documented properties.
- Seek 100% test coverage for all new code.
- Update release notes for relevant changes.
- Handle edge cases like zero values, NaN, and infinity, subnormals and very
  large floating point values accurately.
- Provide the fastest possible implementation without sacrificing correctness
  for `f32` and `f64` types.

## Verification

to verify that all features of the library work correctly, you can run the following command:
```bash
cargo test
```
You may add arguments to select specific tests or features.

To statically check the rust code, you may run
```bash
cargo fmt --check
cargo clippy --all-features
cargo semver-checks
```

To check that the C++ code builds and clang-tidy issues are addressed, run
```bash
./ci/bazel_build.bash
```

