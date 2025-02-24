//! Quaternions for Rust.
//!
//! `num-quaternion` is a Rust library designed for robust, efficient and easy
//! to use quaternion arithmetic and operations. [`Quaternion`]s and
//! [`UnitQuaternion`]s are used extensively in computer graphics, robotics,
//! and physics for representing rotations and orientations.
//!
//! # Features
//!
//! - **Basic Quaternion Operations**: Addition, subtraction, multiplication,
//!   division, and conjugation.
//! - **Unit Quaternions**: Special support for unit quaternions with optimized
//!   operations.
//! - **Conversion Functions**: Convert to/from Euler angles, rotation vectors,
//!   and more.
//! - **Interpolation**: Spherical linear interpolation (SLERP) for smooth
//!   rotations.
//! - **Interoperability**: Works with the `serde` and the `rand` crates.
//! - **Comprehensive Documentation**: Detailed documentation with examples to
//!   help you get started quickly.
//!
//! For `#![no_std]` environments, disable the default `std` feature and enable
//! `libm` to benefit from the advanced mathematical functions of `num-quaternion`:
//!
//! ```toml
//! [dependencies]
//! num-quaternion = { version = "1.0.3", default-features = false, features = ["libm"] }
//! ```
//!
//! Then, include it in your crate:
//!
//! ```rust
//! use num_quaternion::{Quaternion, UnitQuaternion, Q32, Q64, UQ32, UQ64};
//! ```
//!
//! # Usage
//!
//! ## Creating Quaternions
//!
//! ```rust
//! // Create a quaternion with explicit components
//! # use num_quaternion::Q32;
//! let q1 = Q32::new(1.0, 2.0, 3.0, 4.0);  // = 1 + 2i + 3j + 4k
//!
//! // Create a quaternion using shorthand notation
//! let q2 = 1.0 + Q32::I;  // = 1 + i
//! ```
//!
//! ## Basic Operations
//!
//! ```rust
//! # use num_quaternion::Q32;
//! # let q1 = Q32::ONE;
//! # let q2 = Q32::ONE;
//! let q3 = q1 + q2;        // Quaternion addition
//! let q4 = q1 * q2;        // Quaternion multiplication
//! let q_conj = q1.conj();  // Quaternion conjugation
//! ```
//!
//! ## Unit Quaternions
//!
//! ```rust
//! # #[cfg(feature = "std")]
//! # {
//! # use num_quaternion::{Q32, UQ32};
//! # let q1 = Q32::ONE;
//! let uq1 = q1.normalize().expect("Normalization failed"); // Normalize quaternion
//! let uq2 = UQ32::I;  // Unit quaternion representing the imaginary unit
//! # }
//! ```
//!
//! ## Conversion Functions
//!
//! ```rust
//! # #[cfg(feature = "std")]
//! # {
//! # use num_quaternion::UnitQuaternion;
//! // From Euler angles
//! let (roll, pitch, yaw) = (1.5, 1.0, 3.0);
//! let uq = UnitQuaternion::from_euler_angles(roll, pitch, yaw);
//!
//! // To Euler angles
//! let euler_angles = uq.to_euler_angles();
//!
//! // From rotation vector
//! let rotation_vector = [1.0, 0.0, 0.0]; // x axis rotation, 1 radian
//! let uq = UnitQuaternion::from_rotation_vector(&rotation_vector);
//!
//! // To rotation vector
//! let rotation_vector = uq.to_rotation_vector();
//! # }
//! ```
//!
//! ## Spherical Linear Interpolation (SLERP)
//!
//! ```rust
//! # #[cfg(feature = "std")]
//! # {
//! # use num_quaternion::UQ32;
//! let uq1 = UQ32::ONE;  // Create a unit quaternion
//! let uq2 = UQ32::I;    // Create another unit quaternion
//! let interpolated = uq1.slerp(&uq2, 0.3);  // Perform SLERP with t=0.3
//! # }
//! ```
//!
//! # Cargo Features
//!
//! The crate offers the following features which can be freely enabled or
//! disabled:
//!
//! - `std`: Enables the use of the Rust standard library. This feature is on
//!   by default. If disabled (`default-features = false` in `Cargo.toml`),
//!   the crate can be used in environments where the standard library is not
//!   available or desired.
//!
//! - `libm`: This can be used as a fallback library to provide mathematical
//!   functions which are otherwise provided by the standard library. Use
//!   this feature if you want to work without standard library, but still
//!   want features that internally require floating point functions like
//!   `sqrt` or `acos`, etc. This includes functionality like computing
//!   the norm, converting from and to Euler angles and spherical linear
//!   interpolation.
//!
//! - `unstable`: Enables unstable features. Items marked as `unstable` may
//!   undergo breaking changes in future releases without a major version
//!   update. Use with caution in production environments.
//!
//! - `serde`: Implements the `Serialize` and `Deserialize` traits for all
//!   data structures where possible. Useful for easy integration with
//!   serialization frameworks, enabling data storage and communication
//!   functionalities.
//!
//! - `rand`: Implements the `Distribution` trait for `UnitQuaternion`. This
//!   feature allows you to randomly sample unit quaternions using the `rand`
//!   crate.
//!
//!
//! # Design Rationale and Error Handling
//!
//! For detailed design principles and the error handling strategy see the
//! [Design Rationale](DESIGN_RATIONALE.md).
//!
//! # Contributing
//!
//! Contributions are welcome! Please fork
//! [the repository](https://github.com/ralphtandetzky/num-quaternion) and submit
//! pull requests. By contributing, you agree that your contributions will be
//! dual-licensed under the Apache-2.0 and MIT licenses.
//!
//! If you have any questions or need help, feel free to open an
//! [issue on GitHub](https://github.com/ralphtandetzky/num-quaternion/issues).
//!
//! Further instructions can be found in the [CONTRIBUTING.md](CONTRIBUTING.md)
//! guidelines.
//!
//! # Acknowledgements
//!
//! Special thanks to [@cuviper](https://github.com/cuviper) for the
//! [`num-complex` crate](https://crates.io/crates/num-complex) which served
//! as a model for this crate. `num-quaternion` is designed to integrate seamlessly
//! with the [`rust-num` family](https://github.com/rust-num) of crates.

#![deny(missing_docs)]
#![no_std]

#[cfg(feature = "std")]
extern crate std;

mod arithmetics;
mod pure_quaternion;
mod quaternion;
mod unit_quaternion;

pub use {
    quaternion::{Quaternion, Q32, Q64},
    unit_quaternion::{EulerAngles, ReadMat3x3, UnitQuaternion, UQ32, UQ64},
};

#[cfg(feature = "unstable")]
pub use pure_quaternion::{PureQuaternion, PQ32, PQ64};
