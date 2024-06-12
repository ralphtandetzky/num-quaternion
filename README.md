# num-quaternion

[![Build](https://img.shields.io/github/actions/workflow/status/ralphtandetzky/num-quaternion/cargo_build_and_test.yml?branch=master)](https://github.com/ralphtandetzky/num-quaternion/actions)
[![Docs.rs](https://docs.rs/num-quaternion/badge.svg)](https://docs.rs/num-quaternion)
[![Downloads](https://img.shields.io/crates/d/num-quaternion)](https://crates.io/crates/num-quaternion)
[![Crates.io](https://img.shields.io/crates/v/num-quaternion.svg)](https://crates.io/crates/num-quaternion)
[![MIT License](https://img.shields.io/badge/license-MIT-blue)](LICENSE-MIT.md)
[![Apache License](https://img.shields.io/badge/license-Apache_2.0-blue)](LICENSE-APACHE.md)

`num-quaternion` is a Rust library for quaternion arithmetic and operations.
It provides a robust and efficient implementation of quaternions, including
support for unit quaternions and various common operations like quaternion
multiplication, normalization, and spherical linear interpolation (SLERP).

## Features

- **Basic Quaternion Operations**: Addition, subtraction, multiplication, and conjugation.
- **Unit Quaternions**: Special support for unit quaternions with optimized operations.
- **Conversion Functions**: Convert to/from Euler angles, rotation vectors, and more.
- **Interpolation**: Spherical linear interpolation (SLERP) for smooth rotations.


## Installation

Add `num-quaternion` to your `Cargo.toml`:

```toml
[dependencies]
num-quaternion = "0.2.4"
```

This crate can be used without the standard library (`#![no_std]`) by disabling
the default `std` feature. Use this in `Cargo.toml`:

```toml
[dependencies]
num-quaternion = { version = "0.2.4", default-features = false, features = ["libm"] }
```

Then, include it in your crate:

```rust
use num_quaternion::{Quaternion, UnitQuaternion, Q32, Q64, UQ32, UQ64};
```

## Usage

### Creating Quaternions

```rust
// The abbreviation `Q32` stands for `Quaternion<f32>`:
let q1 = Q32::new(1.0, 2.0, 3.0, 4.0);  // = 1 + 2i + 3j + 4k
let q2 = 1.0 + Q32::I;  // = 1 + i
```

### Basic Operations

```rust
let q3 = q1 + q2;
let q4 = q1 * q2;
let q_conj = q1.conj();
```

### Unit Quaternions

```rust
let uq1 = q1.normalized()?;
let uq2 = UQ32::I;
```

### Conversion Functions

```rust
// From Euler angles
let (roll, pitch, yaw) = (1.5, 1.0, 3.0);
let uq = UnitQuaternion::from_euler_angles(roll, pitch, yaw);

// To Euler angles
let euler_angles = uq.to_euler_angles();

// From rotation vector
let rotation_vector = [1.0, 0.0, 0.0];
let uq = UnitQuaternion::from_rotation_vector(&rotation_vector);

// To rotation vector
let rotation_vector = uq.to_rotation_vector();
```

### Spherical Linear Interpolation (SLERP)

```rust
let uq1 = UnitQuaternion::new(1.0, 0.0, 0.0, 0.0);
let uq2 = UnitQuaternion::new(0.0, 1.0, 0.0, 0.0);
let interpolated = uq1.slerp(&uq2, 0.3);
```

## Documentation

Comprehensive documentation with examples can be found on
[docs.rs](https://docs.rs/num-quaternion/latest/num-quaternion/).


## Releases

Release notes are available in [RELEASES.md](RELEASES.md).


## Contributing

Contributions are welcome! Please fork the repository and submit pull requests.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall
be dual licensed as above, without any additional terms or conditions.


## License

Licensed under either of

 * [Apache License, Version 2.0](LICENSE-APACHE.md)
 * [MIT license](LICENSE-MIT.md)

at your option.


## Acknowledgements

Thanks to [@cuviper](https://github.com/cuviper) for the
[`num-complex` crate](https://crates.io/crates/num-complex) which served
as a model for this crate. It borrows a lot from it. This is by design,
so this crate can be used consistently with the other crates from the
[`rust-num` family](https://github.com/rust-num) of crates.
