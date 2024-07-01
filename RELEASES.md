# Release 0.2.10 (2024-07-02)

- Implemented the function `sqrt` for `Quaternion`.

**Contributors**: @ralphtandetzky


# Release 0.2.9 (2024-06-26)

- Implemented the functions `exp` and `ln`.

**Contributors**: @ralphtandetzky


# Release 0.2.8 (2024-06-21)

- Added feature `serde`. Implemented `Serialize` and `Deserialize` traits for
  all data structures.

**Contributors**: @ralphtandetzky


# Release 0.2.7 (2024-06-19)

- Added a [Code of Conduct](CODE_OF_CONDUCT.md).
- Added [Contribution information](CONTRIBUTING.md).
- Added a [Security Policy](SECURITY.md)
- Added scripts for better security checks on GitHub.
- Added pull request template.
- Updated crate's keywords and categories.
- Minor updates in documentation and scripts.

**Contributors**: @ralphtandetzky


# Release 0.2.6 (2024-06-15)

- Added arithmetic operators which take references as arguments.

**Contributors**: @ralphtandetzky


# Release 0.2.5 (2024-06-13)

- Added a lot of top level documentation.

**Contributors**: @ralphtandetzky


# Release 0.2.4 (2024-06-10)

- Added dot product.
- Added spherical linear interpolation.

**Contributors**: @ralphtandetzky


# Release 0.2.3 (2024-06-08)

- Added the function `UnitQuaternion::to_rotation_vector`.
- Added new unstable function `from_euler_angles_struct`. (It is marked
  as unstable since the naming may change in the future.)

**Contributors**: @ralphtandetzky


# Release 0.2.2 (2024-06-05)

- Added the function `UnitQuaternion::to_euler_angles`.
- Added the struct `EulerAngles<T>`.
- Small improvements in the docs.

**Contributors**: @ralphtandetzky


# Release 0.2.1 (2024-06-04)

- Added the function `UnitQuaternion::rotate_vector`.

**Contributors**: @ralphtandetzky


# Release 0.2.0 (2024-06-03)

- **Breaking change:** Renamed member fields of `Quaternion` from `a`, `b`, `c`,
  and `d` to `w`, `x`, `y`, and `z`.
- Added `UnitQuaternion` type with tons of new functionality.
- Made a couple of functions `#[inline]`.

**Contributors**: @ralphtandetzky


# Release 0.1.2 (2024-05-26)

- Fixed issue with Rust Doc.

**Contributors**: @ralphtandetzky


# Release 0.1.1 (2024-05-26)

- Improved documentation including math formulas.
- Added operators for real-values left hand operands.

**Contributors**: @ralphtandetzky


# Release 0.1.0 (2024-05-18)

- Initial release.

**Contributors**: @ralphtandetzky
