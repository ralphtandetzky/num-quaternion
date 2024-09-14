# Releases in Reverse Chronological Order

## Release 0.2.29 (2024-09-15)

- Added `pose_animation` example.
- Added examples Rustdoc examples for all methods.

## Release 0.2.28 (2024-09-07)

- Split `lib.rs` into new modules `quaternion`, `unit_quaternion`, and `arithmetics`.

## Release 0.2.27 (2024-09-04)

- Reached 100% test coverage.

**Contributors**: @ralphtandetzky

## Release 0.2.26 (2024-08-29)

- Added `rand` feature to crate.
- Implemented sampling `UnitQuaternion<T>` from a uniform distribution.
- Increased test coverage.

**Contributors**: @ralphtandetzky

## Release 0.2.25 (2024-08-21)

- Fix bug in gimbal lock case of `UnitQuaternion::to_euler_angles`.
- Improved test coverage.

**Contributors**: @ralphtandetzky

## Release 0.2.24 (2024-08-18)

- Fixed edge case in `Quaternion::ln`.
- Added scripts for test coverage.
- Improved test coverage.

**Contributors**: @ralphtandetzky

## Release 0.2.23 (2024-08-12)

- Added `has_nan` and `is_all_nan` methods for `Quaternion<T>`.
- Improved documentation and tests.

**Contributors**: @ralphtandetzky

## Release 0.2.22 (2024-08-10)

- Improved documentation.
- Added `to_inner` method for `UnitQuaternion<T>`.
- Made `is_finite` method public.
- Improved existing tests and added more tests.

**Contributors**: @ralphtandetzky

## Release 0.2.21 (2024-08-01)

- Updated minimum required Rust compiler version to 1.61.
- Added [Design Rationale](DESIGN_RATIONALE.md) laying out the design goals and
  the error handling strategy.
- Adjusted formatting of the code.

**Contributors**: @ralphtandetzky

## Release 0.2.20 (2024-07-28)

- Implemented `Quaternion::powf` method.
- Changed method `Quaternion::ln` for zero arguments to return `NEG_INFINITY`
  independent of the signs of the input quaternion components.

**Contributors**: @ralphtandetzky

## Release 0.2.19 (2024-07-27)

- Changed orientation of some `UnitQuaternion` functions to conform to common conventions.

**Contributors**: @ralphtandetzky

## Release 0.2.18 (2024-07-23)

- Implemented `Quaternion::expf` method.

**Contributors**: @ralphtandetzky

## Release 0.2.17 (2024-07-22)

- Implemented edge cases of `Quaternion::exp` method.

**Contributors**: @ralphtandetzky

## Release 0.2.16 (2024-07-16)

- Fixed inaccuracy in `Quaternion::norm` for very large and very small results. (https://github.com/ralphtandetzky/num-quaternion/issues/51)
- Added new function `Quaternion::fast_norm` for branchless fast (but possibly inaccurate) norm calculation.

**Contributors**: @ralphtandetzky

## Release 0.2.15 (2024-07-13)

- Added new function `UnitQuaternion::from_rotation_matrix3x3`.

**Contributors**: @ralphtandetzky

## Release 0.2.14 (2024-07-11)

- Added new function `UnitQuaternion::to_rotation_matrix3x3`.

**Contributors**: @ralphtandetzky

## Release 0.2.13 (2024-07-08)

- Added new function `UnitQuaternion::from_two_vectors`.

**Contributors**: @ralphtandetzky

## Release 0.2.12 (2024-07-05)

- Fixed failing rustdoc tests for disabled features.
- Improvements in CI.
- Added scripts for running CI tests locally.

**Contributors**: @ralphtandetzky

## Release 0.2.11 (2024-07-03)

- Implemented the function `sqrt` for `UnitQuaternion`.
- Fixed warnings in markdown files.

**Contributors**: @ralphtandetzky

## Release 0.2.10 (2024-07-02)

- Implemented the function `sqrt` for `Quaternion`.

**Contributors**: @ralphtandetzky

## Release 0.2.9 (2024-06-26)

- Implemented the functions `exp` and `ln`.

**Contributors**: @ralphtandetzky

## Release 0.2.8 (2024-06-21)

- Added feature `serde`. Implemented `Serialize` and `Deserialize` traits for
  all data structures.

**Contributors**: @ralphtandetzky

## Release 0.2.7 (2024-06-19)

- Added a [Code of Conduct](CODE_OF_CONDUCT.md).
- Added [Contribution information](CONTRIBUTING.md).
- Added a [Security Policy](SECURITY.md)
- Added scripts for better security checks on GitHub.
- Added pull request template.
- Updated crate's keywords and categories.
- Minor updates in documentation and scripts.

**Contributors**: @ralphtandetzky

## Release 0.2.6 (2024-06-15)

- Added arithmetic operators which take references as arguments.

**Contributors**: @ralphtandetzky

## Release 0.2.5 (2024-06-13)

- Added a lot of top level documentation.

**Contributors**: @ralphtandetzky

## Release 0.2.4 (2024-06-10)

- Added dot product.
- Added spherical linear interpolation.

**Contributors**: @ralphtandetzky

## Release 0.2.3 (2024-06-08)

- Added the function `UnitQuaternion::to_rotation_vector`.
- Added new unstable function `from_euler_angles_struct`. (It is marked
  as unstable since the naming may change in the future.)

**Contributors**: @ralphtandetzky

## Release 0.2.2 (2024-06-05)

- Added the function `UnitQuaternion::to_euler_angles`.
- Added the struct `EulerAngles<T>`.
- Small improvements in the docs.

**Contributors**: @ralphtandetzky

## Release 0.2.1 (2024-06-04)

- Added the function `UnitQuaternion::rotate_vector`.

**Contributors**: @ralphtandetzky

## Release 0.2.0 (2024-06-03)

- **Breaking change:** Renamed member fields of `Quaternion` from `a`, `b`, `c`,
  and `d` to `w`, `x`, `y`, and `z`.
- Added `UnitQuaternion` type with tons of new functionality.
- Made a couple of functions `#[inline]`.

**Contributors**: @ralphtandetzky

## Release 0.1.2 (2024-05-26)

- Fixed issue with Rust Doc.

**Contributors**: @ralphtandetzky

## Release 0.1.1 (2024-05-26)

- Improved documentation including math formulas.
- Added operators for real-values left hand operands.

**Contributors**: @ralphtandetzky

## Release 0.1.0 (2024-05-18)

- Initial release.

**Contributors**: @ralphtandetzky
