# num-quaternion

[![Build](https://img.shields.io/github/actions/workflow/status/ralphtandetzky/num-quaternion/cargo_build_and_test.yml?branch=master)](https://github.com/ralphtandetzky/num-quaternion/actions)
[![Docs.rs](https://docs.rs/num-quaternion/badge.svg)](https://docs.rs/num-quaternion)
[![Downloads](https://img.shields.io/crates/d/num-quaternion)](https://crates.io/crates/num-quaternion)
[![Crates.io](https://img.shields.io/crates/v/num-quaternion.svg)](https://crates.io/crates/num-quaternion)
[![MIT License](https://img.shields.io/badge/license-MIT-blue)](LICENSE-MIT.md)
[![Apache License](https://img.shields.io/badge/license-Apache_2.0-blue)](LICENSE-APACHE.md)

Quaternions for Rust.

`num-quaternion` is a Rust library designed for robust, efficient and easy to
use quaternion arithmetic and operations.
[Quaternions](https://en.wikipedia.org/wiki/Quaternion) are used extensively in
computer graphics, robotics, and physics for representing rotations and
orientations.

## Features

- **Basic Quaternion Operations**: Addition, subtraction, multiplication,
  division, and conjugation.
- **Unit Quaternions**: Special support for unit quaternions with optimized
  operations.
- **Conversion Functions**: Convert to/from Euler angles, rotation vectors,
  and more.
- **Interpolation**: Spherical linear interpolation (SLERP) for smooth
  rotations.
- **Comprehensive Documentation**: Detailed documentation with examples to
  help you get started quickly.

## Installation

Add `num-quaternion` to your `Cargo.toml`:

```toml
[dependencies]
num-quaternion = "0.2.11"
```

For `#![no_std]` environments, disable the default `std` feature and enable
`libm` to benefit from the advanced mathematical functions of `num-quaternion`:

```toml
[dependencies]
num-quaternion = { version = "0.2.11", default-features = false, features = ["libm"] }
```

Then, include it in your crate:

```rust
use num_quaternion::{Quaternion, UnitQuaternion, Q32, Q64, UQ32, UQ64};
```

## Usage

### Creating Quaternions

```rust
// Create a quaternion with explicit components
let q1 = Q32::new(1.0, 2.0, 3.0, 4.0);  // = 1 + 2i + 3j + 4k

// Create a quaternion using shorthand notation
let q2 = 1.0 + Q32::I;  // = 1 + i
```

### Basic Operations

```rust
let q3 = q1 + q2;        // Quaternion addition
let q4 = q1 * q2;        // Quaternion multiplication
let q_conj = q1.conj();  // Quaternion conjugation
```

### Unit Quaternions

```rust
let uq1 = q1.normalize().expect("Normalization failed"); // Normalize quaternion
let uq2 = UQ32::I;  // Unit quaternion representing the imaginary unit
```

### Conversion Functions

```rust
// From Euler angles
let (roll, pitch, yaw) = (1.5, 1.0, 3.0);
let uq = UnitQuaternion::from_euler_angles(roll, pitch, yaw);

// To Euler angles
let euler_angles = uq.to_euler_angles();

// From rotation vector
let rotation_vector = [1.0, 0.0, 0.0]; // x axis rotation, 1 radian
let uq = UnitQuaternion::from_rotation_vector(&rotation_vector);

// To rotation vector
let rotation_vector = uq.to_rotation_vector();
```

### Spherical Linear Interpolation (SLERP)

```rust
let uq1 = UQ32::ONE;  // Create a unit quaternion
let uq2 = UQ32::I;    // Create another unit quaternion
let interpolated = uq1.slerp(&uq2, 0.3);  // Perform SLERP with t=0.3
```

## Documentation

Comprehensive documentation with examples can be found on
[docs.rs](https://docs.rs/num-quaternion/latest/num-quaternion/).

## Releases

Detailed release notes are provided in [RELEASES.md](RELEASES.md).

## Contributing

Contributions are welcome! Please fork
[the repository](https://github.com/ralphtandetzky/num-quaternion) and submit
pull requests. By contributing, you agree that your contributions will be
dual-licensed under the Apache-2.0 and MIT licenses.

If you have any questions or need help, feel free to open an
[issue on GitHub](https://github.com/ralphtandetzky/num-quaternion/issues).

Further instructions can be found in the [CONTRIBUTING.md](CONTRIBUTING.md)
guidelines.

## License

Licensed under either of

- [Apache License, Version 2.0](LICENSE-APACHE.md)
- [MIT license](LICENSE-MIT.md)

at your option.

## Acknowledgements

Special thanks to [@cuviper](https://github.com/cuviper) for the
[`num-complex` crate](https://crates.io/crates/num-complex) which served
as a model for this crate. `num-quaternion` is designed to integrate seamlessly
with the [`rust-num` family](https://github.com/rust-num) of crates.
