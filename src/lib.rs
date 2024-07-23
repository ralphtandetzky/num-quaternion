//! Quaternions for Rust.
//!
//! `num-quaternion` is a Rust library designed for robust, efficient and easy to
//! use quaternion arithmetic and operations.
//! [`Quaternion`]s and [`UnitQuaternion`]s are used extensively in
//! computer graphics, robotics, and physics for representing rotations and
//! orientations.
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
//! - **Comprehensive Documentation**: Detailed documentation with examples to
//!   help you get started quickly.
//!
//! For `#![no_std]` environments, disable the default `std` feature and enable
//! `libm` to benefit from the advanced mathematical functions of `num-quaternion`:
//!
//! ```toml
//! [dependencies]
//! num-quaternion = { version = "0.2.18", default-features = false, features = ["libm"] }
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

use core::{
    borrow::Borrow,
    cmp::Ordering,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};
use num_traits::{ConstOne, ConstZero, Inv, Num, One, Zero};

#[cfg(any(feature = "std", feature = "libm"))]
use {
    core::num::FpCategory,
    num_traits::{float::Float, FloatConst},
};

/// Quaternion type.
///
/// We follow the naming conventions from
/// [Wikipedia](https://en.wikipedia.org/wiki/Quaternion) for quaternions.
/// You can generate quaternions using the [`new`](Quaternion::new) function:
///
/// ```
/// // 1 + 2i + 3j + 4k
/// # use num_quaternion::Quaternion;
/// let q = Quaternion::new(1.0f32, 2.0, 3.0, 4.0);
/// ```
///
/// Alternatively, you can construct quaternions directly with the member data
/// fields:
///
/// ```
/// # use num_quaternion::Q32;
/// let q = Q32 { w: 1.0, x: 2.0, y: 3.0, z: 4.0 };
/// ```
///
/// This is exactly equivalent to the first method. The latter example uses the
/// shorthand `Q32` for `Quaternion<f32>`. For your convenience, there are also
/// the member constants [`ONE`](Quaternion::ONE), [`I`](Quaternion::I),
/// [`J`](Quaternion::J), and [`K`](Quaternion::K) for the mathematical
/// values $1$, $i$, $j$, and $k$, respectively.
///
/// `Quaternion`s support the usual arithmetic operations of addition,
/// subtraction, multiplication, and division. You can compute the
/// norm with [`norm`](Quaternion::norm) and its square with
/// [`norm_sqr`](Quaternion::norm_sqr). Quaternion conjugation is done by the
/// member function [`conj`](Quaternion::conj). You can normalize a
/// quaternion by calling [`normalize`](Quaternion::normalize), which returns
/// a [`UnitQuaternion`].
///
/// To work with rotations, please use [`UnitQuaternion`]s.
///
/// # Examples
///
/// Basic usage:
///
/// ```rust
/// # use num_quaternion::Quaternion;
/// let q1 = Quaternion::new(1.0f32, 0.0, 0.0, 0.0);
/// let q2 = Quaternion::new(0.0, 1.0, 0.0, 0.0);
/// let q3 = q1 + q2;
/// assert_eq!(q3, Quaternion::new(1.0, 1.0, 0.0, 0.0));
/// ```
///
/// # Fields
///
/// - `w`: Real part of the quaternion.
/// - `x`: The coefficient of $i$.
/// - `y`: The coefficient of $j$.
/// - `z`: The coefficient of $k$.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct Quaternion<T> {
    /// Real part of the quaternion.
    pub w: T,
    /// The coefficient of $i$.
    pub x: T,
    /// The coefficient of $j$.
    pub y: T,
    /// The coefficient of $k$.
    pub z: T,
}

/// Alias for a [`Quaternion<f32>`].
pub type Q32 = Quaternion<f32>;
/// Alias for a [`Quaternion<f64>`].
pub type Q64 = Quaternion<f64>;

impl<T> Quaternion<T> {
    /// Create a new quaternion $a + bi + cj + dk$.
    #[inline]
    pub const fn new(w: T, x: T, y: T, z: T) -> Self {
        Self { w, x, y, z }
    }
}

impl<T> Quaternion<T>
where
    T: ConstZero,
{
    /// A constant zero `Quaternion`.
    pub const ZERO: Self = Self::new(T::ZERO, T::ZERO, T::ZERO, T::ZERO);
}

impl<T> ConstZero for Quaternion<T>
where
    T: ConstZero,
{
    const ZERO: Self = Self::ZERO;
}

impl<T> Zero for Quaternion<T>
where
    T: Zero,
{
    #[inline]
    fn zero() -> Self {
        Self::new(Zero::zero(), Zero::zero(), Zero::zero(), Zero::zero())
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.w.is_zero() && self.x.is_zero() && self.y.is_zero() && self.z.is_zero()
    }

    #[inline]
    fn set_zero(&mut self) {
        self.w.set_zero();
        self.x.set_zero();
        self.y.set_zero();
        self.z.set_zero();
    }
}

impl<T> Quaternion<T>
where
    T: ConstZero + ConstOne,
{
    /// A constant `Quaternion` of value $1$.
    ///
    /// See also [`Quaternion::one`].
    pub const ONE: Self = Self::new(T::ONE, T::ZERO, T::ZERO, T::ZERO);

    /// A constant `Quaternion` of value $i$.
    ///
    /// See also [`Quaternion::i`].
    pub const I: Self = Self::new(T::ZERO, T::ONE, T::ZERO, T::ZERO);

    /// A constant `Quaternion` of value $j$.
    ///
    /// See also [`Quaternion::j`].
    pub const J: Self = Self::new(T::ZERO, T::ZERO, T::ONE, T::ZERO);

    /// A constant `Quaternion` of value $k$.
    ///
    /// See also [`Quaternion::k`].
    pub const K: Self = Self::new(T::ZERO, T::ZERO, T::ZERO, T::ONE);
}

impl<T> ConstOne for Quaternion<T>
where
    T: ConstZero + ConstOne + Num + Clone,
{
    const ONE: Self = Self::ONE;
}

impl<T> One for Quaternion<T>
where
    T: Num + Clone,
{
    #[inline]
    fn one() -> Self {
        Self::new(One::one(), Zero::zero(), Zero::zero(), Zero::zero())
    }

    #[inline]
    fn is_one(&self) -> bool {
        self.w.is_one() && self.x.is_zero() && self.y.is_zero() && self.z.is_zero()
    }

    #[inline]
    fn set_one(&mut self) {
        self.w.set_one();
        self.x.set_zero();
        self.y.set_zero();
        self.z.set_zero();
    }
}

impl<T> Quaternion<T>
where
    T: Zero + One,
{
    /// Returns the imaginary unit $i$.
    ///
    /// See also [`Quaternion::I`].
    #[inline]
    pub fn i() -> Self {
        Self::new(T::zero(), T::one(), T::zero(), T::zero())
    }

    /// Returns the imaginary unit $j$.
    ///
    /// See also [`Quaternion::J`].
    #[inline]
    pub fn j() -> Self {
        Self::new(T::zero(), T::zero(), T::one(), T::zero())
    }

    /// Returns the imaginary unit $k$.
    ///
    /// See also [`Quaternion::K`].
    #[inline]
    pub fn k() -> Self {
        Self::new(T::zero(), T::zero(), T::zero(), T::one())
    }
}

#[cfg(any(feature = "std", feature = "libm"))]
impl<T> Quaternion<T>
where
    T: Float,
{
    /// Returns a quaternion filled with `NAN` values.
    #[inline]
    pub fn nan() -> Self {
        let nan = T::nan();
        Self::new(nan, nan, nan, nan)
    }
}

impl<T> Quaternion<T>
where
    T: Clone + Mul<T, Output = T> + Add<T, Output = T>,
{
    /// Returns the square of the norm.
    ///
    /// The result is $w^2 + x^2 + y^2 + z^2$ with some rounding errors.
    /// The rounding error is at most 2
    /// [ulps](https://en.wikipedia.org/wiki/Unit_in_the_last_place).
    ///
    /// This is guaranteed to be more efficient than [`norm`](Quaternion::norm()).
    /// Furthermore, `T` only needs to support addition and multiplication
    /// and therefore, this function works for more types than
    /// [`norm`](Quaternion::norm()).
    #[inline]
    pub fn norm_sqr(&self) -> T {
        (self.w.clone() * self.w.clone() + self.y.clone() * self.y.clone())
            + (self.x.clone() * self.x.clone() + self.z.clone() * self.z.clone())
    }
}

impl<T> Quaternion<T>
where
    T: Clone + Neg<Output = T>,
{
    /// Returns the conjugate quaternion. i.e. the imaginary part is negated.
    #[inline]
    pub fn conj(&self) -> Self {
        Self::new(
            self.w.clone(),
            -self.x.clone(),
            -self.y.clone(),
            -self.z.clone(),
        )
    }
}

impl<T> Quaternion<T>
where
    for<'a> &'a Self: Inv<Output = Quaternion<T>>,
{
    /// Returns the multiplicative inverse `1/self`.
    #[inline]
    pub fn inv(&self) -> Self {
        Inv::inv(self)
    }
}

impl<T> Inv for &Quaternion<T>
where
    T: Clone + Neg<Output = T> + Num,
{
    type Output = Quaternion<T>;

    #[inline]
    fn inv(self) -> Self::Output {
        let norm_sqr = self.norm_sqr();
        Quaternion::new(
            self.w.clone() / norm_sqr.clone(),
            -self.x.clone() / norm_sqr.clone(),
            -self.y.clone() / norm_sqr.clone(),
            -self.z.clone() / norm_sqr,
        )
    }
}

impl<T> Inv for Quaternion<T>
where
    for<'a> &'a Self: Inv<Output = Quaternion<T>>,
{
    type Output = Quaternion<T>;

    #[inline]
    fn inv(self) -> Self::Output {
        Inv::inv(&self)
    }
}

#[cfg(any(feature = "std", feature = "libm"))]
impl<T> Quaternion<T>
where
    T: Float,
{
    /// Calculates |self|.
    ///
    /// The result is $\sqrt{w^2+x^2+y^2+z^2}$ with some possible rounding
    /// errors. The total relative rounding error is at most two
    /// [ulps](https://en.wikipedia.org/wiki/Unit_in_the_last_place).
    ///
    /// If any of the components of the input quaternion is `NaN`, then `NaN`
    /// is returned. Otherwise, if any of the components is infinite, then
    /// a positive infinite value is returned.
    #[inline]
    pub fn norm(self) -> T {
        let Self { w, x, y, z } = self;
        let norm_sqr = w * w + x * x + y * y + z * z;
        if norm_sqr.is_normal() {
            // The most likely case first: everything is normal.
            norm_sqr.sqrt()
        } else {
            // This function call may not be inlined each time `norm()` is
            // inlined. This can avoid code bloat. At the same time is keeps
            // the norm function simple.
            self.handle_non_normal_cases(norm_sqr)
        }
    }

    /// Computes the norm of self under the precondition that the square norm
    /// of self is not a normal floating point number.
    fn handle_non_normal_cases(self, norm_sqr: T) -> T {
        debug_assert!(!norm_sqr.is_normal());
        let s = T::min_positive_value();
        if norm_sqr < s {
            // norm_sqr is either subnormal or zero.
            if self.is_zero() {
                // Likely, the whole vector is zero. If so, we can return
                // zero directly and avoid expensive floating point math.
                T::zero()
            } else {
                // Otherwise, scale up, such that the norm will be in the
                // normal floating point range, then scale down the result.
                (self / s).fast_norm() * s
            }
        } else if norm_sqr.is_infinite() {
            // There are two possible cases:
            //   1. one of w, x, y, z is infinite, or
            //   2. none of them is infinite.
            // In the first case, multiplying by s or dividing by it does
            // not change the infiniteness and thus the correct result is
            // returned. In the second case, multiplying by s makes sure
            // that the square norm is a normal floating point number.
            // Dividing by it will rescale the result to the correct
            // magnitude.
            (self * s).fast_norm() / s
        } else {
            debug_assert!(norm_sqr.is_nan(), "norm_sqr is not NaN");
            T::nan()
        }
    }

    /// Calculates |self| without branching.
    ///
    /// This function returns the same result as [`norm`](Self::norm), if
    /// |self|² is a normal floating point number (i. e. there is no overflow
    /// nor underflow), or if `self` is zero. In these cases the maximum
    /// relative error of the result is guaranteed to be less than two ulps.
    /// In all other cases, there's no guarantee on the precision of the
    /// result:
    ///
    /// * If |self|² overflows, then $\infty$ is returned.
    /// * If |self|² underflows to zero, then zero will be returned.
    /// * If |self|² is a subnormal number (very small floating point value
    ///   with reduced relative precision), then the result is the square root
    ///   of that.
    ///
    /// In other words, this function can be imprecise for very large and very
    /// small floating point numbers, but it is generally faster than
    /// [`norm`](Self::norm), because it does not do any branching. So if you
    /// are interested in maximum speed of your code, feel free to use this
    /// function. If you need to be precise results for the whole range of the
    /// floating point type `T`, stay with [`norm`](Self::norm).
    #[inline]
    pub fn fast_norm(self) -> T {
        self.norm_sqr().sqrt()
    }

    /// Normalizes the quaternion to length $1$.
    ///
    /// The sign of the real part will be the same as the sign of the input.
    /// If the input quaternion
    ///   * is zero, or
    ///   * has infinite length, or
    ///   * has a `NaN` value,
    /// then `None` will be returned.
    #[inline]
    pub fn normalize(self) -> Option<UnitQuaternion<T>> {
        let norm = self.norm();
        match norm.classify() {
            FpCategory::Normal | FpCategory::Subnormal => Some(UnitQuaternion(self / norm)),
            _ => None,
        }
    }
}

impl<T> From<T> for Quaternion<T>
where
    T: Zero,
{
    #[inline]
    fn from(a: T) -> Self {
        Self::new(a, T::zero(), T::zero(), T::zero())
    }
}

impl<'a, T> From<&'a T> for Quaternion<T>
where
    T: Clone + Zero,
{
    #[inline]
    fn from(a: &T) -> Self {
        From::from(a.clone())
    }
}

impl<T> From<UnitQuaternion<T>> for Quaternion<T> {
    #[inline]
    fn from(q: UnitQuaternion<T>) -> Self {
        q.0
    }
}

impl<'a, T> From<&'a UnitQuaternion<T>> for &'a Quaternion<T> {
    #[inline]
    fn from(q: &'a UnitQuaternion<T>) -> Self {
        &q.0
    }
}

impl<T> Add<Quaternion<T>> for Quaternion<T>
where
    T: Add<T, Output = T>,
{
    type Output = Quaternion<T>;

    #[inline]
    fn add(self, rhs: Quaternion<T>) -> Self::Output {
        Self::new(
            self.w + rhs.w,
            self.x + rhs.x,
            self.y + rhs.y,
            self.z + rhs.z,
        )
    }
}

impl<T> Add<T> for Quaternion<T>
where
    T: Add<T, Output = T>,
{
    type Output = Quaternion<T>;

    #[inline]
    fn add(self, rhs: T) -> Self::Output {
        Self::new(self.w + rhs, self.x, self.y, self.z)
    }
}

impl<T> Add<UnitQuaternion<T>> for Quaternion<T>
where
    T: Add<T, Output = T>,
{
    type Output = Quaternion<T>;

    #[inline]
    fn add(self, rhs: UnitQuaternion<T>) -> Self::Output {
        self + rhs.0
    }
}

impl<T> Sub<Quaternion<T>> for Quaternion<T>
where
    T: Sub<T, Output = T>,
{
    type Output = Quaternion<T>;

    #[inline]
    fn sub(self, rhs: Quaternion<T>) -> Self::Output {
        Self::new(
            self.w - rhs.w,
            self.x - rhs.x,
            self.y - rhs.y,
            self.z - rhs.z,
        )
    }
}

impl<T> Sub<T> for Quaternion<T>
where
    T: Sub<T, Output = T>,
{
    type Output = Quaternion<T>;

    #[inline]
    fn sub(self, rhs: T) -> Self::Output {
        Self::new(self.w - rhs, self.x, self.y, self.z)
    }
}

impl<T> Sub<UnitQuaternion<T>> for Quaternion<T>
where
    T: Sub<T, Output = T>,
{
    type Output = Quaternion<T>;

    #[inline]
    fn sub(self, rhs: UnitQuaternion<T>) -> Self::Output {
        self - rhs.0
    }
}

impl<T> Mul<Quaternion<T>> for Quaternion<T>
where
    T: Add<T, Output = T> + Sub<T, Output = T> + Mul<T, Output = T> + Clone,
{
    type Output = Quaternion<T>;

    #[inline]
    fn mul(self, rhs: Quaternion<T>) -> Self::Output {
        let a = self.w.clone() * rhs.w.clone()
            - self.x.clone() * rhs.x.clone()
            - self.y.clone() * rhs.y.clone()
            - self.z.clone() * rhs.z.clone();
        let b = self.w.clone() * rhs.x.clone()
            + self.x.clone() * rhs.w.clone()
            + self.y.clone() * rhs.z.clone()
            - self.z.clone() * rhs.y.clone();
        let c = self.w.clone() * rhs.y.clone() - self.x.clone() * rhs.z.clone()
            + self.y.clone() * rhs.w.clone()
            + self.z.clone() * rhs.x.clone();
        let d = self.w * rhs.z + self.x * rhs.y - self.y * rhs.x + self.z * rhs.w;
        Self::new(a, b, c, d)
    }
}

impl<T> Mul<T> for Quaternion<T>
where
    T: Mul<T, Output = T> + Clone,
{
    type Output = Quaternion<T>;

    #[inline]
    fn mul(self, rhs: T) -> Self::Output {
        Self::new(
            self.w * rhs.clone(),
            self.x * rhs.clone(),
            self.y * rhs.clone(),
            self.z * rhs,
        )
    }
}

impl<T> Mul<UnitQuaternion<T>> for Quaternion<T>
where
    Quaternion<T>: Mul<Output = Quaternion<T>>,
{
    type Output = Quaternion<T>;

    #[inline]
    fn mul(self, rhs: UnitQuaternion<T>) -> Self::Output {
        self * rhs.0
    }
}

impl<T> Div<Quaternion<T>> for Quaternion<T>
where
    T: Num + Clone + Neg<Output = T>,
{
    type Output = Quaternion<T>;

    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline]
    fn div(self, rhs: Quaternion<T>) -> Self::Output {
        self * rhs.inv()
    }
}

impl<T> Div<T> for Quaternion<T>
where
    T: Div<T, Output = T> + Clone,
{
    type Output = Quaternion<T>;

    #[inline]
    fn div(self, rhs: T) -> Self::Output {
        Self::new(
            self.w / rhs.clone(),
            self.x / rhs.clone(),
            self.y / rhs.clone(),
            self.z / rhs,
        )
    }
}

impl<T> Div<UnitQuaternion<T>> for Quaternion<T>
where
    Quaternion<T>: Mul<Output = Quaternion<T>>,
    T: Neg<Output = T> + Clone,
{
    type Output = Quaternion<T>;

    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline]
    fn div(self, rhs: UnitQuaternion<T>) -> Self::Output {
        self * rhs.0.conj()
    }
}

macro_rules! impl_bin_op_assign {
    (impl $bin_op_assign_trait:ident, $bin_op_assign:ident, $bin_op_trait:ident, $bin_op:ident) => {
        impl<T, S> $bin_op_assign_trait<S> for Quaternion<T>
        where
            Self: $bin_op_trait<S, Output = Self> + Clone,
        {
            #[inline]
            fn $bin_op_assign(&mut self, other: S) {
                *self = self.clone().$bin_op(other);
            }
        }
    };
}

impl_bin_op_assign!(impl AddAssign, add_assign, Add, add);
impl_bin_op_assign!(impl SubAssign, sub_assign, Sub, sub);
impl_bin_op_assign!(impl MulAssign, mul_assign, Mul, mul);
impl_bin_op_assign!(impl DivAssign, div_assign, Div, div);

macro_rules! impl_op_with_ref {
    (impl<$T:ident> $bin_op_trait:ident::$bin_op:ident for $lhs_type:ty, $rhs_type:ty) => {
        impl<$T> $bin_op_trait<&$rhs_type> for $lhs_type
        where
            Self: $bin_op_trait<$rhs_type>,
            $rhs_type: Clone,
        {
            type Output = <Self as $bin_op_trait<$rhs_type>>::Output;

            fn $bin_op(self, rhs: &$rhs_type) -> Self::Output {
                self.$bin_op(rhs.clone())
            }
        }

        impl<$T> $bin_op_trait<&$rhs_type> for &$lhs_type
        where
            $lhs_type: $bin_op_trait<$rhs_type> + Clone,
            $rhs_type: Clone,
        {
            type Output = <$lhs_type as $bin_op_trait<$rhs_type>>::Output;

            fn $bin_op(self, rhs: &$rhs_type) -> Self::Output {
                self.clone().$bin_op(rhs.clone())
            }
        }

        impl<$T> $bin_op_trait<$rhs_type> for &$lhs_type
        where
            $lhs_type: $bin_op_trait<$rhs_type> + Clone,
        {
            type Output = <$lhs_type as $bin_op_trait<$rhs_type>>::Output;

            fn $bin_op(self, rhs: $rhs_type) -> Self::Output {
                self.clone().$bin_op(rhs)
            }
        }
    };
}

impl_op_with_ref!(impl<T> Add::add for Quaternion<T>, Quaternion<T>);
impl_op_with_ref!(impl<T> Sub::sub for Quaternion<T>, Quaternion<T>);
impl_op_with_ref!(impl<T> Mul::mul for Quaternion<T>, Quaternion<T>);
impl_op_with_ref!(impl<T> Div::div for Quaternion<T>, Quaternion<T>);
impl_op_with_ref!(impl<T> Add::add for Quaternion<T>, UnitQuaternion<T>);
impl_op_with_ref!(impl<T> Sub::sub for Quaternion<T>, UnitQuaternion<T>);
impl_op_with_ref!(impl<T> Mul::mul for Quaternion<T>, UnitQuaternion<T>);
impl_op_with_ref!(impl<T> Div::div for Quaternion<T>, UnitQuaternion<T>);
impl_op_with_ref!(impl<T> Add::add for Quaternion<T>, T);
impl_op_with_ref!(impl<T> Sub::sub for Quaternion<T>, T);
impl_op_with_ref!(impl<T> Mul::mul for Quaternion<T>, T);
impl_op_with_ref!(impl<T> Div::div for Quaternion<T>, T);

macro_rules! impl_ops_lhs_real {
    ($($real:ty),*) => {
        $(
        impl Add<Quaternion<$real>> for $real {
            type Output = Quaternion<$real>;

            #[inline]
            fn add(self, mut rhs: Quaternion<$real>) -> Self::Output {
                rhs.w += self;
                rhs
            }
        }

        impl Sub<Quaternion<$real>> for $real {
            type Output = Quaternion<$real>;

            #[inline]
            fn sub(self, rhs: Quaternion<$real>) -> Self::Output {
                let zero = <$real>::zero();
                Self::Output::new(self - rhs.w, zero - rhs.x, zero - rhs.y, zero - rhs.z)
            }
        }

        impl Mul<Quaternion<$real>> for $real {
            type Output = Quaternion<$real>;

            #[inline]
            fn mul(self, rhs: Quaternion<$real>) -> Self::Output {
                Self::Output::new(self * rhs.w, self * rhs.x, self * rhs.y, self * rhs.z)
            }
        }
    )*
    };
}

impl_ops_lhs_real!(usize, u8, u16, u32, u64, u128, isize, i8, i16, i32, i64, i128, f32, f64);

impl Div<Q32> for f32 {
    type Output = Q32;

    #[inline]
    fn div(mut self, rhs: Q32) -> Self::Output {
        self /= rhs.norm_sqr();
        Self::Output::new(self * rhs.w, self * -rhs.x, self * -rhs.y, self * -rhs.z)
    }
}

impl Div<Q64> for f64 {
    type Output = Q64;

    #[inline]
    fn div(mut self, rhs: Q64) -> Self::Output {
        self /= rhs.norm_sqr();
        Self::Output::new(self * rhs.w, self * -rhs.x, self * -rhs.y, self * -rhs.z)
    }
}

impl<T> Neg for Quaternion<T>
where
    T: Neg<Output = T>,
{
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self::new(-self.w, -self.x, -self.y, -self.z)
    }
}

impl<T> Quaternion<T>
where
    T: Add<T, Output = T> + Mul<T, Output = T>,
{
    /// Computes the dot product of two quaternions interpreted as
    /// 4D real vectors.
    #[inline]
    pub fn dot(self, other: Self) -> T {
        self.w * other.w + self.y * other.y + (self.x * other.x + self.z * other.z)
    }
}

impl<T> Quaternion<T>
where
    T: Num + Clone,
{
    /// Raises `self` to an unsigned integer power `n`, i. e. $q^n$.
    pub fn powu(&self, mut n: u32) -> Self {
        if n == 0 {
            Self::one()
        } else {
            let mut base = self.clone();
            while n & 1 == 0 {
                n /= 2;
                base = base.clone() * base;
            }

            if n == 1 {
                return base;
            }

            let mut acc = base.clone();
            while n > 1 {
                n /= 2;
                base = base.clone() * base;
                if n & 1 == 1 {
                    acc *= base.clone();
                }
            }
            acc
        }
    }
}

impl<T> Quaternion<T>
where
    T: Clone + Num + Neg<Output = T>,
{
    /// Raises `self` to a signed integer power `n`, i. e. $q^n$
    ///
    /// For $n=0$ the result is exactly $1$.
    #[inline]
    pub fn powi(&self, n: i32) -> Self {
        if n >= 0 {
            self.powu(n as u32)
        } else {
            self.inv().powu(n.wrapping_neg() as u32)
        }
    }
}

#[cfg(any(feature = "std", feature = "libm"))]
impl<T> Quaternion<T>
where
    T: Float + FloatConst,
{
    /// Given a quaternion $q$, returns $e^q$, where $e$ is the base of the
    /// natural logarithm.
    ///
    /// This method computes the exponential of a quaternion, handling various
    /// edge cases to ensure numerical stability and correctness:
    ///
    /// 1. **Negative Real Part**: If the real part is sufficiently negative,
    ///    such that $e^{\Re q}$ is approximately zero, the method returns
    ///    zero. This is done even if the imaginary part contains infinite or
    ///    NaN values.
    ///
    /// 2. **NaN Input**: If any component of the input quaternion is `NaN`,
    ///    the method returns a quaternion filled with `NaN` values.
    ///
    /// 3. **Large Imaginary Norm**: If the norm of the imaginary part is too
    ///    large, the method may return a `NaN` quaternion or a quaternion with
    ///    the correct magnitude but inaccurate direction.
    ///
    /// 4. **Infinite Result**: If $e^{\Re q}$ results in `+∞`, the method
    ///    computes the direction and returns an infinite quaternion in that
    ///    direction, ensuring that `∞ * 0` values are mapped to zero instead
    ///    of `NaN`.
    ///
    /// 5. **Finite Norm**: For finite norms, the method ensures a very small
    ///    relative error in all components, depending on the accuracy of the
    ///    underlying floating point function implementations.
    pub fn exp(self) -> Self {
        let one = T::one();
        let two = one + one;
        let four = two + two;
        let half = one / two;
        let quarter = one / four;
        let inf = T::infinity();

        // Compute the exponential of the real part, which gives the norm of
        // the result
        let result_norm = self.w.exp();

        match result_norm.partial_cmp(&inf) {
            Some(Ordering::Less) => {
                if result_norm.is_zero() {
                    return Self::zero();
                }
                // Case: 0 < result_norm < ∞

                // Compute the squared norm of the imaginary part
                let sqr_angle = self.x * self.x + self.y * self.y + self.z * self.z;

                if sqr_angle <= T::epsilon() {
                    // Use Taylor series approximation for small angles to
                    // maintain numerical stability. By Taylor expansion of
                    // `cos(angle)` we get
                    //     cos(angle) >= 1 - angle² / 2
                    // and thus |cos(angle) - 1| is less than half a floating
                    // point epsilon. Similarly,
                    //     sinc(angle) >= 1 - angle² / 6
                    // and thus |sinc(angle) - 1| is less than a sixth of a
                    // floating point epsilon.
                    let w = result_norm;
                    let x = result_norm * self.x;
                    let y = result_norm * self.y;
                    let z = result_norm * self.z;
                    Self::new(w, x, y, z)
                } else {
                    // Standard computation for larger angles
                    let angle = sqr_angle.sqrt();
                    let cos_angle = angle.cos();
                    let sinc_angle = angle.sin() / angle;
                    let w = result_norm * cos_angle;
                    let x = result_norm * self.x * sinc_angle;
                    let y = result_norm * self.y * sinc_angle;
                    let z = result_norm * self.z * sinc_angle;
                    Self::new(w, x, y, z)
                }
            }
            Some(_) => {
                // Case: result_norm == ∞
                let map = |a: T| {
                    // Map zero to zero with same sign and everything else to
                    // infinity with same sign as the input.
                    if a.is_zero() {
                        a
                    } else {
                        inf.copysign(a)
                    }
                };
                let sqr_angle = self.x * self.x + self.y * self.y + self.z * self.z;
                if sqr_angle < T::PI() * T::PI() * quarter {
                    // Angle less than 90 degrees
                    Self::new(inf, map(self.x), map(self.y), map(self.z))
                } else if sqr_angle.is_finite() {
                    // Angle 90 degrees or more -> careful sign handling
                    let angle = sqr_angle.sqrt();
                    let angle_revolutions_fract = (angle * T::FRAC_1_PI() * half).fract();
                    let cos_angle_signum = (angle_revolutions_fract - half).abs() - quarter;
                    let sin_angle_signum = one.copysign(half - angle_revolutions_fract);
                    Self::new(
                        inf.copysign(cos_angle_signum),
                        map(self.x) * sin_angle_signum,
                        map(self.y) * sin_angle_signum,
                        map(self.z) * sin_angle_signum,
                    )
                } else {
                    // Angle is super large or NaN
                    assert!(sqr_angle.is_infinite() || sqr_angle.is_nan());
                    Self::nan()
                }
            }
            None => {
                // Case: result_norm is NaN
                debug_assert!(result_norm.is_nan());
                Self::nan()
            }
        }
    }

    /// Raises a real value (`base`) to a quaternion (`self`) power.
    ///
    /// Given a quaternion $q$ and a real value $t$, this function computes
    /// $t^q := e^{q \ln t}$. The function handles special cases as follows:
    ///
    /// - If $t = \pm 0$ and $\Re q > 0$, or $t = +\infty$ and $\Re q < 0$,
    ///   a zero quaternion is returned.
    /// - If $t < 0$ or $t$ is `NaN`, a `NaN` quaternion (filled with `NaN`
    ///   values) is returned.
    /// - If $t$ is $+0$, $-0$, or $+\infty$ and $\Re q = 0$, a `NaN`
    ///   quaternion is returned.
    /// - If $t = +\infty$ and $q$ is a positive real number, $+\infty$ is
    ///   returned. For other values of $q$ with $\Re q > 0$, a `NaN`
    ///   quaternion is returned.
    /// - If $t = \pm 0$ and $q$ is a negative real number, $+\infty$ is
    ///   returned. For other values of $q$ with $\Re q < 0$, a `NaN`
    ///   quaternion is returned.
    ///
    /// For finite positive $t$, the following conventions for boundary values
    /// of $q$ are applied:
    ///
    /// - If any component of $q$ is `NaN` or any imaginary component of $q$ is
    ///   infinite, a `NaN` quaternion is returned.
    /// - Otherwise, if $\Re q = -\infty$ and $t > 1$, a zero quaternion is
    ///   returned.
    /// - Otherwise, if $\Re q = +\infty$ and $0 < t < 1$, a zero quaternion is
    ///   returned.
    /// - Otherwise, if $\Re q$ is infinite and $t = 1$, a `NaN` quaternion is
    ///   returned.
    /// - Otherwise, if $\Re q = +\infty$ and $t > 1$, an infinite quaternion
    ///   without `NaN` values is returned.
    /// - Otherwise, if $\Re q = -\infty$ and $0 < t < 1$, an infinite
    ///   quaternion without `NaN` values is returned.
    ///
    /// If the true result's norm is neither greater than the largest
    /// representable floating point value nor less than the smallest
    /// representable floating point value, and the direction of the output
    /// quaternion cannot be accurately determined, a `NaN` quaternion may or
    /// may not be returned to indicate inaccuracy. This can occur when
    /// $\|\Im(q) \ln t\|$ is on the order of $1/\varepsilon$, where
    /// $\varepsilon$ is the machine precision of the floating point type used.
    #[inline]
    pub fn expf(self, base: T) -> Self {
        if (base.is_infinite()
            && self.w > T::zero()
            && self.x.is_zero()
            && self.y.is_zero()
            && self.z.is_zero())
            || (base.is_zero()
                && self.w < T::zero()
                && self.x.is_zero()
                && self.y.is_zero()
                && self.z.is_zero())
        {
            T::infinity().into()
        } else {
            (self * base.ln()).exp()
        }
    }

    fn is_finite(&self) -> bool {
        self.w.is_finite() && self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }

    /// Computes the natural logarithm of a quaternion.
    ///
    /// The function implements the following guarantees for extreme input
    /// values:
    ///
    /// - The function is continuous onto the branch cut taking into account
    ///   the sign of the coefficient of $i$.
    /// - For all quaternions $q$ it holds `q.conj().ln() == q.ln().conj()`.
    /// - The signs of the coefficients of the imaginary parts of the outputs
    ///   are equal to the signs of the respective coefficients of the inputs.
    ///   This also holds for signs of zeros, but not for `NaNs`.
    /// - If $q = -0 + 0i$, the result is $-\infty+\pi i$. (The coefficients
    ///   of $j$ and $k$ are zero with the signs copied.)
    /// - If $q = +0$, the result is $-\infty$.
    /// - If the input has a `NaN` value, then the result is `NaN` in all
    ///   components.
    /// - Otherwise, if $q = w + xi + yj + zk$ where at least one of
    ///   $w, x, y, z$ is infinite, then the real part of the result is
    ///   $+\infty$ and the imaginary part is the imaginary part
    ///   of the logarithm of $f(w) + f(x)i + f(y)j + f(z)k$ where
    ///     - $f(+\infty) := 1$,
    ///     - $f(-\infty) :=-1$, and
    ///     - $f(s) := 0$ for finite values of $s$.
    pub fn ln(self) -> Self {
        // The square norm of the imaginary part.
        let sqr_norm_im = self.x * self.x + self.y * self.y + self.z * self.z;
        // The square norm of `self`.
        let norm_sqr = self.w * self.w + sqr_norm_im;

        match norm_sqr.classify() {
            FpCategory::Normal => {
                // The normal case: First compute the real part of the result.
                let w = norm_sqr.ln() * T::from(0.5).expect("Conversion failed");

                if sqr_norm_im <= self.w * self.w * T::epsilon() {
                    // We're close to or on the positive real axis
                    if self.w.is_sign_positive() {
                        // This approximation leaves a relative error of less
                        // than a floating point epsilon for the imaginary part
                        let x = self.x / self.w;
                        let y = self.y / self.w;
                        let z = self.z / self.w;
                        Self::new(w, x, y, z)
                    } else if self.x.is_zero() && self.y.is_zero() && self.z.is_zero() {
                        // We're on the negative real axis.
                        Self::new(w, T::PI().copysign(self.x), self.y, self.z)
                    } else {
                        // We're close the the negative real axis. Compute the
                        // norm of the imaginary part.
                        let norm_im = if sqr_norm_im.is_normal() {
                            // In this case we get maximum precision by using
                            // `sqr_norm_im`.
                            sqr_norm_im.sqrt()
                        } else {
                            // Otherwise, using `sqr_norm_im` is imprecise.
                            // We magnify the imaginary part first, so we can
                            // get around this problem.
                            let f = T::min_positive_value().sqrt();
                            let xf = self.x / f;
                            let yf = self.y / f;
                            let zf = self.z / f;
                            let sqr_sum = xf * xf + yf * yf + zf * zf;
                            sqr_sum.sqrt() * f
                        };

                        // The angle of `self` to the positive real axis is
                        // pi minus the angle from the negative real axis.
                        // The angle from the negative real axis
                        // can be approximated by `norm_im / self.w.abs()``
                        // which is equal to `-norm_im / self.w`. This the
                        // angle from the positive real axis is
                        // `pi + norm_im / self.w`. We obtain the imaginary
                        // part of the result by multiplying this value by
                        // the imaginary part of the input normalized, or
                        // equivalently, by multiplying the imaginary part
                        // of the input by the following factor:
                        let f = T::PI() / norm_im + self.w.recip();

                        Self::new(w, f * self.x, f * self.y, f * self.z)
                    }
                } else {
                    // The most natural case: We're far enough from the real
                    // axis and the norm of the input quaternion is large
                    // enough to exclude any numerical instabilities.
                    let norm_im = if sqr_norm_im.is_normal() {
                        // `sqr_norm_im` has maximum precision.
                        sqr_norm_im.sqrt()
                    } else {
                        // Otherwise, using `sqr_norm_im` is imprecise.
                        // We magnify the imaginary part first, so we can
                        // get around this problem.
                        let f = T::min_positive_value().sqrt();
                        let xf = self.x / f;
                        let yf = self.y / f;
                        let zf = self.z / f;
                        let sqr_sum = xf * xf + yf * yf + zf * zf;
                        sqr_sum.sqrt() * f
                    };
                    let angle = norm_im.atan2(self.w);
                    let x = self.x * angle / norm_im;
                    let y = self.y * angle / norm_im;
                    let z = self.z * angle / norm_im;
                    Self::new(w, x, y, z)
                }
            }
            FpCategory::Zero if self.is_zero() => {
                let x = if self.w.is_sign_positive() {
                    self.x
                } else {
                    T::PI().copysign(self.x)
                };
                Self::new(T::neg_infinity(), x, self.y, self.z)
            }
            FpCategory::Nan => Self::nan(),
            FpCategory::Infinite => {
                // The square norm overflows.
                if self.is_finite() {
                    // There is no infinity entry in the quaternion. Hence,
                    // We can scale the quaternion down and recurse.
                    let factor = T::one() / T::max_value().sqrt();
                    (self * factor).ln() - factor.ln()
                } else {
                    // There is an infinite value in the input quaternion.
                    // Let's map the infinite entries to `±1` and all other
                    // entries to `±0` maintaining the sign.
                    let f = |r: T| {
                        if r.is_infinite() {
                            r.signum()
                        } else {
                            T::zero().copysign(r)
                        }
                    };
                    let q = Self::new(f(self.w), f(self.x), f(self.y), f(self.z));
                    // TODO: Optimize this. There are only a few possible
                    // angles which could be hard-coded. Recursing here
                    // may be a bit heavy.
                    q.ln() + T::infinity()
                }
            }
            _ => {
                // Square norm is less than smallest positive normal value,
                // but `self` is not zero. Let's scale up the value to obtain
                // the precision by recursing and then fix the factor
                // afterwards.
                let factor = T::one() / T::min_positive_value();
                (self * factor).ln() - factor.ln()
            }
        }
    }

    // Computes the square root of a quaternion.
    ///
    /// Given the input quaternion $c$, this function returns the quaternion
    /// $q$ which satisfies $q^2 = c$ and has a real part with a positive sign.
    ///
    /// For extreme values, the following guarantees are implemented:
    ///
    /// - If any coefficient in $c$ is `NaN`, then the result is `NaN` in all
    ///   components.
    /// - Otherwise, for any input $c$, the expression `c.sqrt().conj()` is
    ///   exactly equivalent to `c.conj().sqrt()`, including the signs of
    ///   zeros and infinities, if any.
    /// - For any input $c$, `c.sqrt().w` always has a positive sign.
    /// - For any input $c$, the signs of the three output imaginary parts are
    ///   the same as the input imaginary parts in their respective order,
    ///   except in the case of a `NaN` input.
    /// - For negative real inputs $c$, the result is $\pm\sqrt{-c} i$, where
    ///   the sign is determined by the sign of the input's coefficient of $i$.
    /// - If there is at least one infinite coefficient in the imaginary part,
    ///   then the result will have the same infinite imaginary coefficients
    ///   and the real part is $+\infty$. All other coefficients of the result
    ///   are $0$ with the sign of the respective input.
    /// - If the real part is $-\infty$ and the imaginary part is finite, then
    ///   the result is $\pm\infty i$ with the sign of the coefficient of $i$
    ///   from the input.
    pub fn sqrt(self) -> Self {
        let zero = T::zero();
        let one = T::one();
        let two = one + one;
        let half = one / two;
        let inf = T::infinity();
        let s = one / T::min_positive_value(); // large scale factor
        let norm_sqr = self.norm_sqr();
        match norm_sqr.classify() {
            FpCategory::Normal => {
                // norm of the input
                let norm = norm_sqr.sqrt();

                if self.w.is_sign_positive() {
                    // Compute double the real part of the result directly and
                    // robustly.
                    //
                    // Note: We could also compute the real part directly.
                    // However, this would be inferior for the following
                    //  reasons:
                    //
                    // - To compute the imaginary parts of the result, we
                    //   would need to double the real part anyway, which
                    //   would require an extra arithmetic operation, adding
                    //   to the latency of the computation.
                    // - To avoid this latency, we could also multiply
                    //   `self.x`, `self.y`, and `self.z` by 1/2 and then
                    //   divide by the real part (which takes longer to
                    //   compute). However, this could cost some accuracy
                    //   for subnormal imaginary parts.
                    let wx2 = ((self.w + norm) * two).sqrt();

                    Self::new(wx2 * half, self.x / wx2, self.y / wx2, self.z / wx2)
                } else {
                    // The first formula for the real part of the result may
                    //  not be robust if the sign of the input real part is
                    // negative.
                    let im_norm_sqr = self.y * self.y + (self.x * self.x + self.z * self.z);
                    if im_norm_sqr >= T::min_positive_value() {
                        // Second formula for the real part of the result,
                        // which is robust for inputs with a negative real
                        // part.
                        let wx2 = (im_norm_sqr * two / (norm - self.w)).sqrt();

                        Self::new(wx2 * half, self.x / wx2, self.y / wx2, self.z / wx2)
                    } else if self.x.is_zero() && self.y.is_zero() && self.z.is_zero() {
                        // The input is a negative real number.
                        Self::new(zero, (-self.w).sqrt().copysign(self.x), self.y, self.z)
                    } else {
                        // `im_norm_sqr` is subnormal. Compute the norm of the
                        // imaginary part by scaling up first.
                        let sx = s * self.x;
                        let sy = s * self.y;
                        let sz = s * self.z;
                        let im_norm = (sy * sy + (sx * sx + sz * sz)).sqrt() / s;

                        // Compute the real part according to the second
                        // formula from above.
                        let w = im_norm / (half * (norm - self.w)).sqrt();

                        Self::new(w * half, self.x / w, self.y / w, self.z / w)
                    }
                }
            }
            FpCategory::Zero if self.is_zero() => Self::new(zero, self.x, self.y, self.z),
            FpCategory::Infinite => {
                if self.w == inf
                    || self.x.is_infinite()
                    || self.y.is_infinite()
                    || self.z.is_infinite()
                {
                    let f = |a: T| if a.is_infinite() { a } else { zero.copysign(a) };
                    Self::new(inf, f(self.x), f(self.y), f(self.z))
                } else if self.w == -inf {
                    Self::new(
                        zero,
                        inf.copysign(self.x),
                        zero.copysign(self.y),
                        zero.copysign(self.z),
                    )
                } else {
                    // Input has no infinities. Therefore, the square norm
                    // must have overflowed. Let's scale down.
                    // In release mode, the compiler turns the division into
                    // a multiplication, because `s` is a power of two. Thus,
                    // it's fast.
                    (self / s).sqrt() * s.sqrt()
                }
            }
            FpCategory::Nan => Self::nan(),
            _ => {
                // Square norm is subnormal or zero (underflow), but `self`
                // is not zero. Let's scale up.
                // In release mode, the compiler turns the division into a
                // multiplication, because `s.sqrt()` is a power of two. Thus,
                // it's fast.
                (self * s).sqrt() / s.sqrt()
            }
        }
    }
}

#[cfg(feature = "serde")]
impl<T> serde::Serialize for Quaternion<T>
where
    T: serde::Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        (&self.w, &self.x, &self.y, &self.z).serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de, T> serde::Deserialize<'de> for Quaternion<T>
where
    T: serde::Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let (w, x, y, z) = serde::Deserialize::deserialize(deserializer)?;
        Ok(Self::new(w, x, y, z))
    }
}

/// A quaternion with norm $1$.
///
/// Unit quaternions form a non-commutative group that can be conveniently used
/// for rotating 3D vectors. A 3D vector can be interpreted as a pure
/// quaternion (a quaternion with a real part of zero). Such a pure quaternion
/// $v$ can be rotated in 3D space by computing $q^{-1} \cdot v \cdot q$ for a
/// unit quaternion $q$. The resulting product is again a pure quaternion,
/// which is $v$ rotated around the axis given by the imaginary part of $q$.
/// The method [`rotate_vector`](UnitQuaternion::rotate_vector) performs this
/// operation efficiently. The angle of rotation is double the angle between
/// $1$ and $q$ interpreted as 4D vectors.
///
/// You can create a `UnitQuaternion` by normalizing a `Quaternion` using the
/// [`Quaternion::normalize`](Quaternion::normalize) method. Alternatively, you can use
/// [`from_euler_angles`](UnitQuaternion::from_euler_angles) or
/// [`from_rotation_vector`](UnitQuaternion::from_rotation_vector) to obtain
/// one. The inverse functions
/// [`to_euler_angles`](UnitQuaternion::to_euler_angles) and
/// [`to_rotation_vector`](UnitQuaternion::to_rotation_vector) are also
/// provided.
///
/// [`UnitQuaternion`] offers the same arithmetic operations as [`Quaternion`].
/// Multiplying two unit quaternions yields a unit quaternion in theory.
/// However, due to limited machine precision, rounding errors accumulate
/// in practice and the resulting norm may deviate from $1$ over time.
/// Thus, when you multiply unit quaternions many times, you may need
/// to adjust the norm to maintain accuracy. This can be done by calling
/// the function [`adjust_norm`](UnitQuaternion::adjust_norm).
///
/// Furthermore, you can interpolate uniformly between two quaternions using
/// the [`slerp`](UnitQuaternion::slerp) method, which stands for spherical
/// linear interpolation. This can be used for smooth transitions between
/// 3D rotations.
///
/// See also [`Quaternion`].
///
/// # Examples
///
/// Basic usage:
///
/// ```rust
/// # #[cfg(feature = "std")]
/// # {
/// # use num_quaternion::UnitQuaternion;
/// // Creating a UnitQuaternion from Euler angles
/// let (roll, pitch, yaw) = (1.5, 1.0, 3.0);
/// let uq = UnitQuaternion::from_euler_angles(roll, pitch, yaw);
///
/// // Rotating a vector using the UnitQuaternion
/// let vector = [1.0, 0.0, 0.0];
/// let rotated_vector = uq.rotate_vector(vector);
/// # }
/// ```
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct UnitQuaternion<T>(Quaternion<T>);

/// Alias for a [`UnitQuaternion<f32>`].
pub type UQ32 = UnitQuaternion<f32>;
/// Alias for a [`UnitQuaternion<f64>`].
pub type UQ64 = UnitQuaternion<f64>;

/// Contains the roll, pitch and yaw angle of a rotation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct EulerAngles<T> {
    /// The roll angle.
    pub roll: T,
    /// The pitch angle.
    pub pitch: T,
    /// The yaw angle.
    pub yaw: T,
}

#[cfg(feature = "serde")]
impl<T> serde::Serialize for EulerAngles<T>
where
    T: serde::Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        (&self.roll, &self.pitch, &self.yaw).serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de, T> serde::Deserialize<'de> for EulerAngles<T>
where
    T: serde::Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let (roll, pitch, yaw) = serde::Deserialize::deserialize(deserializer)?;
        Ok(Self { roll, pitch, yaw })
    }
}

#[cfg(any(feature = "std", feature = "libm"))]
impl<T> UnitQuaternion<T>
where
    T: Float,
{
    /// Creates a new Quaternion from roll, pitch and yaw angles.
    pub fn from_euler_angles(roll: T, pitch: T, yaw: T) -> Self {
        let half = T::one() / (T::one() + T::one());
        let (sr, cr) = (roll * half).sin_cos();
        let (sp, cp) = (pitch * half).sin_cos();
        let (sy, cy) = (yaw * half).sin_cos();
        Self(Quaternion::new(
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ))
    }

    /// Creates a new Quaternion from Euler angles.
    ///
    /// *Note.* The reason that this function is marked as `unstable` is that I'm not 100%
    /// confident about the naming of the function.
    #[cfg(feature = "unstable")]
    pub fn from_euler_angles_struct(angles: EulerAngles<T>) -> Self {
        let EulerAngles { roll, pitch, yaw } = angles;
        Self::from_euler_angles(roll, pitch, yaw)
    }
}

#[cfg(any(feature = "std", feature = "libm"))]
impl<T> UnitQuaternion<T>
where
    T: Float + FloatConst,
{
    /// Converts the UnitQuaternion to roll, pitch, and yaw angles.
    pub fn to_euler_angles(&self) -> EulerAngles<T> {
        let &Self(Quaternion { w, x, y, z }) = self;

        let one = T::one();
        let two = one + one;
        let epsilon = T::epsilon();
        let half_pi = T::FRAC_PI_2();

        // Compute the sin of the pitch angle
        let sin_pitch = two * (w * y - z * x);

        // Check for gimbal lock, which occurs when sin_pitch is close to 1 or -1
        if sin_pitch.abs() >= one - epsilon {
            // Gimbal lock case
            let pitch = if sin_pitch >= one - epsilon {
                half_pi // 90 degrees
            } else {
                -half_pi // -90 degrees
            };

            // In the gimbal lock case, roll and yaw are dependent
            let roll = T::zero();
            let yaw = T::atan2(two * (x * y + w * z), one - two * (y * y + z * z));
            EulerAngles { roll, pitch, yaw }
        } else {
            // General case
            let pitch = sin_pitch.asin();
            let roll = T::atan2(two * (w * x + y * z), one - two * (x * x + y * y));
            let yaw = T::atan2(two * (w * z + x * y), one - two * (y * y + z * z));
            EulerAngles { roll, pitch, yaw }
        }
    }
}

#[cfg(any(feature = "std", feature = "libm"))]
impl<T> UnitQuaternion<T>
where
    T: Float,
{
    /// Returns a quaternion from a vector which is parallel to the rotation
    /// axis and whose norm is the rotation angle.
    ///
    /// This function is the inverse of
    /// [`to_rotation_vector`](UnitQuaternion::to_rotation_vector).
    pub fn from_rotation_vector(v: &[T; 3]) -> Self {
        let sqr_norm = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
        let two = T::one() + T::one();
        match sqr_norm.classify() {
            FpCategory::Normal => {
                // TODO: Optimize this further for norms that are not above pi.
                let norm = sqr_norm.sqrt();
                let (sine, cosine) = (norm / two).sin_cos();
                let f = sine / norm;
                Self(Quaternion::new(cosine, v[0] * f, v[1] * f, v[2] * f))
            }
            FpCategory::Zero | FpCategory::Subnormal => Self(Quaternion::new(
                // This formula could be used for norm <= epsilon generally,
                // where epsilon is the floating point epsilon.
                T::one(),
                v[0] / two,
                v[1] / two,
                v[2] / two,
            )),
            FpCategory::Nan | FpCategory::Infinite => Self(Quaternion::nan()),
        }
    }
}

#[cfg(any(feature = "std", feature = "libm"))]
impl<T> UnitQuaternion<T>
where
    T: Float + FloatConst,
{
    /// Returns a rotation vector which is parallel to the rotation
    /// axis and whose norm is the rotation angle.
    ///
    /// This function is the inverse of
    /// [`from_rotation_vector`](UnitQuaternion::from_rotation_vector).
    pub fn to_rotation_vector(&self) -> [T; 3] {
        let q = self.as_quaternion();
        let one = T::one();
        let two = one + one;
        let epsilon = T::epsilon();

        // Check if the absolute value of the quaternion's real part is small
        // enough, so the angle can be computed quickly via `2 * q.w.acos()`.
        // If the value is too large, then the arccosine becomes numerically
        // unstable and we need to compute the angle differently.
        let small_abs_real_part = q.w.abs() < T::from(0.9).unwrap();

        // Compute the sin of half the angle
        let sin_half_angle = if small_abs_real_part {
            (one - q.w * q.w).sqrt()
        } else {
            // This is more expensive, but numerically more accurate than
            // the first branch, if small_abs_real_part is false.
            (q.x * q.x + q.y * q.y + q.z * q.z).sqrt()
        };

        // Compute the angle
        let angle = two
            * if small_abs_real_part {
                q.w.acos()
            } else if q.w.is_sign_positive() {
                // The angle is less than 180 degrees.
                if sin_half_angle < epsilon.sqrt() {
                    // The angle is very close to zero. In this case we can
                    // avoid division by zero and make the computation cheap
                    // at the same time by returning the following immediately.
                    return [two * q.x, two * q.y, two * q.z];
                }
                sin_half_angle.asin()
            } else {
                // The angle is more than 180 degrees.
                if sin_half_angle.is_zero() {
                    // The angle is exactly 360 degrees. To avoid division by zero we
                    // return the following immediately.
                    return [two * T::PI(), T::zero(), T::zero()];
                }
                let pi_minus_half_angle = sin_half_angle.asin();
                T::PI() - pi_minus_half_angle
            };

        // Compute the normalized rotation vector components
        let x = q.x / sin_half_angle;
        let y = q.y / sin_half_angle;
        let z = q.z / sin_half_angle;

        [x * angle, y * angle, z * angle]
    }
}

impl<T> UnitQuaternion<T>
where
    T: Add<Output = T> + Sub<Output = T> + Mul<Output = T> + One + Clone,
{
    /// Computes the rotation matrix implied by a unit quaternion.
    ///
    /// The matrix is returned in row major order, i. e. the indices into the
    /// result array yield the elements in the following order:
    ///
    /// ```text
    ///     [0, 1, 2,
    ///      3, 4, 5,
    ///      6, 7, 8]
    /// ```
    ///
    /// Multiplying by the returned matrix gives the same result as using
    /// [`rotate_vector`](UnitQuaternion::rotate_vector) modulo slightly
    /// different rounding errors.
    ///
    /// # Runtime Considerations
    ///
    /// The matrix multiplication itself can be assumed to be more runtime
    /// efficient than [`rotate_vector`](UnitQuaternion::rotate_vector).
    /// However, computing the matrix also comes with additional cost. Thus
    /// general advice is: Use [`rotate_vector`](UnitQuaternion::rotate_vector),
    /// if you want to rotate a single vector. Perform the matrix
    /// multiplication, if more than one vector needs to be rotated.
    #[inline]
    pub fn to_rotation_matrix3x3(self) -> [T; 9] {
        let two = T::one() + T::one();

        let Self(Quaternion { w, x, y, z }) = self;

        let wx = two.clone() * w.clone() * x.clone();
        let wy = two.clone() * w.clone() * y.clone();
        let wz = two.clone() * w * z.clone();
        let xx = two.clone() * x.clone() * x.clone();
        let xy = two.clone() * x.clone() * y.clone();
        let xz = two.clone() * x * z.clone();
        let yy = two.clone() * y.clone() * y.clone();
        let yz = two.clone() * y * z.clone();
        let zz = two * z.clone() * z;

        [
            T::one() - yy.clone() - zz.clone(),
            xy.clone() + wz.clone(),
            xz.clone() - wy.clone(),
            xy - wz,
            T::one() - xx.clone() - zz,
            yz.clone() + wx.clone(),
            xz + wy,
            yz - wx,
            T::one() - xx - yy,
        ]
    }
}

/// Interface for reading entries of a 3x3 matrix.
pub trait ReadMat3x3<T> {
    /// Returns the entry at the given row and column.
    ///
    /// Both `row` and `col` are zero-based.
    fn at(&self, row: usize, col: usize) -> &T;
}

impl<T> ReadMat3x3<T> for [T; 9] {
    fn at(&self, row: usize, col: usize) -> &T {
        &self[col + 3 * row]
    }
}

impl<T> ReadMat3x3<T> for [[T; 3]; 3] {
    fn at(&self, row: usize, col: usize) -> &T {
        &self[row][col]
    }
}

// TODO: Provide interop with other linear algebra libraries, such as
// * nalgebra
// * cgmath
// * ndarray
// In other words, implement `ReadMat3x3` for the 3x3 matrix implementations.

#[cfg(any(feature = "std", feature = "libm"))]
impl<T> UnitQuaternion<T>
where
    T: Float,
{
    /// Computes a quaternion from a 3x3 rotation matrix.
    ///
    /// The input matrix $O$ is required to be an actual rotation matrix, i. e.
    /// $O^TO$ is the identity matrix and $\det O = 1$ (neglecting floating
    /// point rounding errors).
    ///
    /// The quaternion solution with non-negative real part is returned. This
    /// function reverses the method
    /// [`to_rotation_matrix3x3`](UnitQuaternion::to_rotation_matrix3x3).
    pub fn from_rotation_matrix3x3(mat: &impl ReadMat3x3<T>) -> UnitQuaternion<T> {
        let zero = T::zero();
        let one = T::one();
        let two = one + one;
        let quarter = one / (two * two);
        let m00 = mat.at(0, 0);
        let m11 = mat.at(1, 1);
        let m22 = mat.at(2, 2);
        let trace: T = *m00 + *m11 + *m22;

        // We distinguish different cases and select a computation method which
        // provides robust results. See
        // https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions#Rotation_matrix_%E2%86%94_quaternion
        // where some details of the computation are described.
        if trace > zero {
            let s = (trace + one).sqrt() * two; // s=4*qw
            let qw = quarter * s;
            let qx = (*mat.at(1, 2) - *mat.at(2, 1)) / s;
            let qy = (*mat.at(2, 0) - *mat.at(0, 2)) / s;
            let qz = (*mat.at(0, 1) - *mat.at(1, 0)) / s;
            Self(Quaternion::new(qw, qx, qy, qz))
        } else if (m00 > m11) && (m00 > m22) {
            let s = (one + *m00 - *m11 - *m22).sqrt() * two; // s=4*qx
            let qw = (*mat.at(1, 2) - *mat.at(2, 1)) / s;
            let qx = quarter * s;
            let qy = (*mat.at(1, 0) + *mat.at(0, 1)) / s;
            let qz = (*mat.at(2, 0) + *mat.at(0, 2)) / s;
            if *mat.at(1, 2) >= *mat.at(2, 1) {
                Self(Quaternion::new(qw, qx, qy, qz))
            } else {
                Self(Quaternion::new(-qw, -qx, -qy, -qz))
            }
        } else if m11 > m22 {
            let s = (one + *m11 - *m00 - *m22).sqrt() * two; // s=4*qy
            let qw = (*mat.at(2, 0) - *mat.at(0, 2)) / s;
            let qx = (*mat.at(1, 0) + *mat.at(0, 1)) / s;
            let qy = quarter * s;
            let qz = (*mat.at(2, 1) + *mat.at(1, 2)) / s;
            if *mat.at(2, 0) >= *mat.at(0, 2) {
                Self(Quaternion::new(qw, qx, qy, qz))
            } else {
                Self(Quaternion::new(-qw, -qx, -qy, -qz))
            }
        } else {
            let s = (one + *m22 - *m00 - *m11).sqrt() * two; // s=4*qz
            let qw = (*mat.at(0, 1) - *mat.at(1, 0)) / s;
            let qx = (*mat.at(2, 0) + *mat.at(0, 2)) / s;
            let qy = (*mat.at(2, 1) + *mat.at(1, 2)) / s;
            let qz = quarter * s;
            if *mat.at(0, 1) >= *mat.at(1, 0) {
                Self(Quaternion::new(qw, qx, qy, qz))
            } else {
                Self(Quaternion::new(-qw, -qx, -qy, -qz))
            }
        }
    }
}

#[cfg(any(feature = "std", feature = "libm"))]
impl<T> UnitQuaternion<T>
where
    T: Float,
{
    /// Given a unit vector $\vec a$, returns a unit vector that is
    /// perpendicular to $\vec a$.
    fn make_perpendicular_unit_vector(a: &[T; 3]) -> [T; 3] {
        // Find the component of `a` with the smallest absolute value and form
        // a normalized perpendicular vector using the other two components.
        let zero = T::zero();
        let a_sqr = [a[0] * a[0], a[1] * a[1], a[2] * a[2]];
        if a_sqr[0] <= a_sqr[1] {
            if a_sqr[0] <= a_sqr[2] {
                // component 0 is minimal
                let norm = (a_sqr[1] + a_sqr[2]).sqrt();
                [zero, a[2] / norm, -a[1] / norm]
            } else {
                // component 2 is minimal
                let norm = (a_sqr[0] + a_sqr[1]).sqrt();
                [a[1] / norm, -a[0] / norm, zero]
            }
        } else if a_sqr[1] <= a_sqr[2] {
            // component 1 is minimal
            let norm = (a_sqr[0] + a_sqr[2]).sqrt();
            [a[2] / norm, zero, -a[0] / norm]
        } else {
            // component 2 is minimal
            let norm = (a_sqr[0] + a_sqr[1]).sqrt();
            [a[1] / norm, -a[0] / norm, zero]
        }
    }

    /// Returns a unit quaternion that rotates vector $\vec a$ to vector
    /// $\vec b$ with the minimum angle of rotation.
    ///
    /// The method [`rotate_vector`](UnitQuaternion::rotate_vector) can be used
    /// to apply the rotation. The resulting unit quaternion maps the ray
    /// $\{t\vec{a} : t > 0\}$ to the ray $\{t\vec{b} : t > 0\}$.
    ///
    /// Note that the input vectors neither need to be normalized nor have the
    /// same magnitude. In the case where the input vectors point in opposite
    /// directions, there are multiple solutions to the problem, and one will
    /// be returned. If one (or both) of the input vectors is the zero vector,
    /// the unit quaternion $1$ is returned.
    pub fn from_two_vectors(a: &[T; 3], b: &[T; 3]) -> UnitQuaternion<T> {
        // Normalize vectors `a` and `b`. Let alpha be the angle between `a`
        // and `b`. We aim to compute the quaternion
        //     q = cos(alpha / 2) + sin(alpha / 2) * normalized_axis
        // Start with the cross product and the dot product:
        //     v := a × b = normalized_axis * sin(alpha)
        //     d := a.dot(b) = cos(alpha) = 2 * cos(alpha / 2)² - 1
        // Thus, we can compute double the real part as
        //     wx2 := 2 * cos(alpha / 2) = √(2d + 2).
        // Since
        //     sin(alpha) = 2 * sin(alpha / 2) * cos(alpha / 2),
        // it follows that
        //     sin(alpha / 2) = sin(alpha) / cos(alpha / 2) / 2
        //                    = sin(alpha) / wx2.
        // Consequently, we can compute the quaternion as
        //     q = wx2 / 2 + v / wx2.
        // If d -> -1, then wx2 -> 0, leading to large relative errors in
        // the computation of wx2, making the imaginary part of q inaccurate.
        // This occurs when `a` and `b` point in nearly opposite directions.
        // We can check if they are exactly opposite by testing if `v` is a
        // null vector. If not, then
        //     sin(alpha) = |v|, and
        //     cos(alpha) = -√(1-|v|²) = d
        //     wx2 = √(2d + 2)
        //         = √(2 - 2√(1-|v|²))
        //         = 2|v| / √(2 + 2√(1-|v|²))
        //         = |v| / √(1/2 - d/2).
        // This last formula is numerically stable for negative values of d.
        // Therefore,
        //     q = |v| / √(1/2 - d/2) / 2 + v / |v| * √(1/2 - d/2)

        // Constants
        let zero = T::zero();
        let one = T::one();
        let two = one + one;
        let half = one / two;

        // Normalize vector inputs
        let a_norm = a[0].hypot(a[1].hypot(a[2]));
        let b_norm = b[0].hypot(b[1].hypot(b[2]));
        if a_norm.is_zero() || b_norm.is_zero() {
            return UnitQuaternion::one();
        }
        let a = [a[0] / a_norm, a[1] / a_norm, a[2] / a_norm];
        let b = [b[0] / b_norm, b[1] / b_norm, b[2] / b_norm];

        // Cross product
        let v = [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ];

        // Dot product
        let d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

        let wx2 = if d >= -half {
            // Simple stable formula for the general case
            (two * d + two).sqrt()
        } else {
            // `a` and `b` may be close to anti-parallel
            let v_norm_sqr = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
            if v_norm_sqr.is_zero() {
                // Exactly anti-parallel
                let [x, y, z] = Self::make_perpendicular_unit_vector(&a);
                return UnitQuaternion(Quaternion::new(zero, x, y, z));
            }
            // Stable, more expensive formula for wx2
            (v_norm_sqr / (half - d * half)).sqrt()
        };

        // Return the computed quaternion
        UnitQuaternion(Quaternion::new(
            wx2 / two,
            v[0] / wx2,
            v[1] / wx2,
            v[2] / wx2,
        ))
    }
}

impl<T> Default for UnitQuaternion<T>
where
    T: Num + Clone,
{
    #[inline]
    fn default() -> Self {
        Self::one()
    }
}

impl<T> UnitQuaternion<T>
where
    T: ConstZero + ConstOne,
{
    /// A constant `UnitQuaternion` of value $1$.
    ///
    /// See also [`UnitQuaternion::one`], [`Quaternion::ONE`].
    pub const ONE: Self = Self(Quaternion::ONE);

    /// A constant `UnitQuaternion` of value $i$.
    ///
    /// See also [`UnitQuaternion::i`], [`Quaternion::I`].
    pub const I: Self = Self(Quaternion::I);

    /// A constant `UnitQuaternion` of value $j$.
    ///
    /// See also [`UnitQuaternion::j`], [`Quaternion::J`].
    pub const J: Self = Self(Quaternion::J);

    /// A constant `UnitQuaternion` of value $k$.
    ///
    /// See also [`UnitQuaternion::k`], [`Quaternion::K`].
    pub const K: Self = Self(Quaternion::K);
}

impl<T> ConstOne for UnitQuaternion<T>
where
    T: ConstZero + ConstOne + Num + Clone,
{
    const ONE: Self = Self::ONE;
}

impl<T> One for UnitQuaternion<T>
where
    T: Num + Clone,
{
    #[inline]
    fn one() -> Self {
        Self(Quaternion::one())
    }

    #[inline]
    fn is_one(&self) -> bool {
        self.0.is_one()
    }

    #[inline]
    fn set_one(&mut self) {
        self.0.set_one();
    }
}

impl<T> UnitQuaternion<T>
where
    T: Zero + One,
{
    /// Returns the imaginary unit $i$.
    ///
    /// See also [`UnitQuaternion::I`], [`Quaternion::i`].
    #[inline]
    pub fn i() -> Self {
        Self(Quaternion::i())
    }

    /// Returns the imaginary unit $j$.
    ///
    /// See also [`UnitQuaternion::J`], [`Quaternion::j`].
    #[inline]
    pub fn j() -> Self {
        Self(Quaternion::j())
    }

    /// Returns the imaginary unit $k$.
    ///
    /// See also [`UnitQuaternion::K`], [`Quaternion::k`].
    #[inline]
    pub fn k() -> Self {
        Self(Quaternion::k())
    }
}

impl<T> UnitQuaternion<T> {
    /// Returns the inner quaternion.
    #[inline]
    pub fn into_quaternion(self) -> Quaternion<T> {
        self.0
    }

    /// Returns a reference to the inner quaternion.
    #[inline]
    pub fn as_quaternion(&self) -> &Quaternion<T> {
        &self.0
    }
}

impl<T> Borrow<Quaternion<T>> for UnitQuaternion<T> {
    fn borrow(&self) -> &Quaternion<T> {
        self.as_quaternion()
    }
}

impl<T> UnitQuaternion<T>
where
    T: Clone + Neg<Output = T>,
{
    /// Returns the conjugate quaternion. i.e. the imaginary part is negated.
    #[inline]
    pub fn conj(&self) -> Self {
        Self(self.0.conj())
    }
}

impl<T> UnitQuaternion<T>
where
    T: Clone + Neg<Output = T>,
{
    /// Returns the multiplicative inverse `1/self`.
    ///
    /// This is the same as the conjugate of `self`.
    #[inline]
    pub fn inv(&self) -> Self {
        self.conj()
    }
}

impl<T> Inv for &UnitQuaternion<T>
where
    T: Clone + Neg<Output = T>,
{
    type Output = UnitQuaternion<T>;

    #[inline]
    fn inv(self) -> Self::Output {
        self.conj()
    }
}

impl<T> Inv for UnitQuaternion<T>
where
    T: Clone + Neg<Output = T>,
{
    type Output = UnitQuaternion<T>;

    #[inline]
    fn inv(self) -> Self::Output {
        self.conj()
    }
}

impl<T> Add<UnitQuaternion<T>> for UnitQuaternion<T>
where
    Quaternion<T>: Add<Output = Quaternion<T>>,
{
    type Output = Quaternion<T>;

    #[inline]
    fn add(self, rhs: UnitQuaternion<T>) -> Self::Output {
        self.0 + rhs.0
    }
}

impl<T> Add<Quaternion<T>> for UnitQuaternion<T>
where
    Quaternion<T>: Add<Output = Quaternion<T>>,
{
    type Output = Quaternion<T>;

    #[inline]
    fn add(self, rhs: Quaternion<T>) -> Self::Output {
        self.0 + rhs
    }
}

impl<T> Add<T> for UnitQuaternion<T>
where
    Quaternion<T>: Add<T, Output = Quaternion<T>>,
{
    type Output = Quaternion<T>;

    #[inline]
    fn add(self, rhs: T) -> Self::Output {
        self.0 + rhs
    }
}

impl Add<UQ32> for f32 {
    type Output = Q32;

    #[inline]
    fn add(self, rhs: UQ32) -> Self::Output {
        self + rhs.0
    }
}

impl Add<UQ64> for f64 {
    type Output = Q64;

    #[inline]
    fn add(self, rhs: UQ64) -> Self::Output {
        self + rhs.0
    }
}

impl<T> Sub<UnitQuaternion<T>> for UnitQuaternion<T>
where
    Quaternion<T>: Sub<Output = Quaternion<T>>,
{
    type Output = Quaternion<T>;

    #[inline]
    fn sub(self, rhs: UnitQuaternion<T>) -> Self::Output {
        self.0 - rhs.0
    }
}

impl<T> Sub<Quaternion<T>> for UnitQuaternion<T>
where
    Quaternion<T>: Sub<Output = Quaternion<T>>,
{
    type Output = Quaternion<T>;

    #[inline]
    fn sub(self, rhs: Quaternion<T>) -> Self::Output {
        self.0 - rhs
    }
}

impl<T> Sub<T> for UnitQuaternion<T>
where
    Quaternion<T>: Sub<T, Output = Quaternion<T>>,
{
    type Output = Quaternion<T>;

    #[inline]
    fn sub(self, rhs: T) -> Self::Output {
        self.0 - rhs
    }
}

impl Sub<UQ32> for f32 {
    type Output = Q32;

    #[inline]
    fn sub(self, rhs: UQ32) -> Self::Output {
        self - rhs.0
    }
}

impl Sub<UQ64> for f64 {
    type Output = Q64;

    #[inline]
    fn sub(self, rhs: UQ64) -> Self::Output {
        self - rhs.0
    }
}

impl<T> Mul<UnitQuaternion<T>> for UnitQuaternion<T>
where
    Quaternion<T>: Mul<Output = Quaternion<T>>,
{
    type Output = UnitQuaternion<T>;

    #[inline]
    fn mul(self, rhs: UnitQuaternion<T>) -> Self::Output {
        Self(self.0 * rhs.0)
    }
}

impl<T> Mul<Quaternion<T>> for UnitQuaternion<T>
where
    Quaternion<T>: Mul<Output = Quaternion<T>>,
{
    type Output = Quaternion<T>;

    #[inline]
    fn mul(self, rhs: Quaternion<T>) -> Self::Output {
        self.0 * rhs
    }
}

impl<T> Mul<T> for UnitQuaternion<T>
where
    Quaternion<T>: Mul<T, Output = Quaternion<T>>,
{
    type Output = Quaternion<T>;

    #[inline]
    fn mul(self, rhs: T) -> Self::Output {
        self.0 * rhs
    }
}

impl Mul<UQ32> for f32 {
    type Output = Q32;

    #[inline]
    fn mul(self, rhs: UQ32) -> Self::Output {
        self * rhs.0
    }
}

impl Mul<UQ64> for f64 {
    type Output = Q64;

    #[inline]
    fn mul(self, rhs: UQ64) -> Self::Output {
        self * rhs.0
    }
}

impl<T> Div<UnitQuaternion<T>> for UnitQuaternion<T>
where
    Quaternion<T>: Mul<Output = Quaternion<T>>,
    T: Clone + Neg<Output = T>,
{
    type Output = UnitQuaternion<T>;

    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline]
    fn div(self, rhs: UnitQuaternion<T>) -> Self::Output {
        self * rhs.conj()
    }
}

impl<T> Div<Quaternion<T>> for UnitQuaternion<T>
where
    Quaternion<T>: Div<Output = Quaternion<T>>,
{
    type Output = Quaternion<T>;

    #[inline]
    fn div(self, rhs: Quaternion<T>) -> Self::Output {
        self.0 / rhs
    }
}

impl<T> Div<T> for UnitQuaternion<T>
where
    Quaternion<T>: Div<T, Output = Quaternion<T>>,
{
    type Output = Quaternion<T>;

    #[inline]
    fn div(self, rhs: T) -> Self::Output {
        self.0 / rhs
    }
}

impl Div<UQ32> for f32 {
    type Output = Q32;

    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline]
    fn div(self, rhs: UQ32) -> Self::Output {
        self * rhs.inv().0
    }
}

impl Div<UQ64> for f64 {
    type Output = Q64;

    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline]
    fn div(self, rhs: UQ64) -> Self::Output {
        self * rhs.inv().0
    }
}

impl_op_with_ref!(impl<T> Add::add for UnitQuaternion<T>, UnitQuaternion<T>);
impl_op_with_ref!(impl<T> Sub::sub for UnitQuaternion<T>, UnitQuaternion<T>);
impl_op_with_ref!(impl<T> Mul::mul for UnitQuaternion<T>, UnitQuaternion<T>);
impl_op_with_ref!(impl<T> Div::div for UnitQuaternion<T>, UnitQuaternion<T>);
impl_op_with_ref!(impl<T> Add::add for UnitQuaternion<T>, Quaternion<T>);
impl_op_with_ref!(impl<T> Sub::sub for UnitQuaternion<T>, Quaternion<T>);
impl_op_with_ref!(impl<T> Mul::mul for UnitQuaternion<T>, Quaternion<T>);
impl_op_with_ref!(impl<T> Div::div for UnitQuaternion<T>, Quaternion<T>);
impl_op_with_ref!(impl<T> Add::add for UnitQuaternion<T>, T);
impl_op_with_ref!(impl<T> Sub::sub for UnitQuaternion<T>, T);
impl_op_with_ref!(impl<T> Mul::mul for UnitQuaternion<T>, T);
impl_op_with_ref!(impl<T> Div::div for UnitQuaternion<T>, T);

impl<T> Neg for UnitQuaternion<T>
where
    T: Neg<Output = T>,
{
    type Output = UnitQuaternion<T>;

    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}

impl<T> UnitQuaternion<T>
where
    T: Add<T, Output = T> + Mul<T, Output = T>,
{
    /// Computes the dot product of two unit quaternions interpreted as
    /// 4D real vectors.
    #[inline]
    pub fn dot(self, other: Self) -> T {
        self.0.dot(other.0)
    }
}

#[cfg(any(feature = "std", feature = "libm"))]
impl<T> UnitQuaternion<T>
where
    T: Float,
{
    /// Renormalizes `self`.
    ///
    /// By many multiplications of unit quaternions, round off errors can lead
    /// to norms which are deviating from $1$ significantly. This function
    /// fixes that inaccuracy.
    #[inline]
    pub fn adjust_norm(self) -> Self {
        // TODO: Optimize for norms which are close to 1.
        self.0
            .normalize()
            .expect("Unit quaternion value too inaccurate. Cannot renormalize.")
    }
}

impl<T> UnitQuaternion<T>
where
    T: Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Clone,
{
    /// Rotates a vector using a quaternion.
    ///
    /// Given a unit quaternion $q$ and a pure quaternion $v$ (i. e. a
    /// quaternion with real part zero), the mapping $v \mapsto q^*vq$
    /// is a 3D rotation in the space of pure quaternions. This function
    /// performs this 3D rotation efficiently.
    pub fn rotate_vector(self, vector: [T; 3]) -> [T; 3] {
        let q = self.into_quaternion();
        let [vx, vy, vz] = vector;
        let q_inv_v = Quaternion::<T>::new(
            q.x.clone() * vx.clone() + q.y.clone() * vy.clone() + q.z.clone() * vz.clone(),
            q.w.clone() * vx.clone() - q.y.clone() * vz.clone() + q.z.clone() * vy.clone(),
            q.w.clone() * vy.clone() + q.x.clone() * vz.clone() - q.z.clone() * vx.clone(),
            q.w.clone() * vz - q.x.clone() * vy + q.y.clone() * vx,
        );
        let result = q_inv_v * q;
        [result.x, result.y, result.z]
    }
}

#[cfg(any(feature = "std", feature = "libm"))]
impl<T> UnitQuaternion<T>
where
    T: Float,
{
    /// Spherical linear interpolation between two unit quaternions.
    ///
    /// `t` should be in the range [0, 1], where 0 returns `self` and 1 returns
    /// `other` or `-other`, whichever is closer to `self`.
    pub fn slerp(&self, other: &Self, t: T) -> Self {
        let one = T::one();
        let dot = self.dot(*other);

        // If the dot product is negative, slerp won't take the shorter path.
        // We fix this by reversing one quaternion.
        let (dot, other) = if dot.is_sign_positive() {
            (dot, *other)
        } else {
            (-dot, -*other)
        };

        // Use a threshold to decide when to use linear interpolation to avoid
        // precision issues
        let threshold = one - T::epsilon().sqrt();
        if dot > threshold {
            // Perform linear interpolation and normalize the result
            return Self(*self + (other - *self) * t);
        }

        // theta_0 = angle between input quaternions
        let theta_0 = dot.acos();
        // theta = angle between self and result
        let theta = theta_0 * t;

        // Compute the spherical interpolation coefficients
        let sin_theta = theta.sin();
        let sin_theta_0 = theta_0.sin();
        let s0 = ((one - t) * theta_0).sin() / sin_theta_0;
        let s1 = sin_theta / sin_theta_0;

        // The following result is already normalized, if the inputs are
        // normalized (which we assume).
        Self(*self * s0 + other * s1)
    }
}

#[cfg(any(feature = "std", feature = "libm"))]
impl<T> UnitQuaternion<T>
where
    T: Float + FloatConst,
{
    /// Computes the square root of a unit quaternion.
    ///
    /// Given an input unit quaternion $c$, this function returns the unit
    /// quaternion $q$ which satisfies $q^2 = c$ and has a real part with a
    /// positive sign.
    ///
    /// For $c = -1$, there are multiple solutions to these constraints. In
    /// that case $q = \pm i$ is returned. The sign is determined by the input
    /// coefficient of the imaginary unit $i$.
    ///
    /// In any case, the three imaginary parts of the result have the same sign
    /// as the three imaginary parts of the input.
    pub fn sqrt(self) -> Self {
        let zero = T::zero();
        let one = T::one();
        let two = one + one;
        let half = one / two;
        let UnitQuaternion(c) = self;

        if c.w >= -half {
            // Compute double the real part of the result directly and
            // robustly.
            //
            // Note: We could also compute the real part directly.
            // However, this would be inferior for the following reasons:
            //
            // - To compute the imaginary parts of the result, we would need
            //   to double the real part anyway, which would require an
            //   extra arithmetic operation, adding to the latency of the
            //   computation.
            // - To avoid this latency, we could also multiply `c.x`, `c.y`,
            //   and `c.z` by 1/2 and then divide by the real part (which
            //   takes longer to compute). However, this could cost some
            //   accuracy for subnormal imaginary parts.
            let wx2 = (c.w * two + two).sqrt();

            UnitQuaternion(Quaternion::new(wx2 * half, c.x / wx2, c.y / wx2, c.z / wx2))
        } else {
            // For cases where the real part is too far in the negative direction.
            //
            // Note: We could also use the formula 1 - c.w * c.w for the
            // square norm of the imaginary part. However, if the real part
            // `c.w` is close to -1, then this becomes inaccurate. This is
            // especially the case, if the actual norm of the input
            // quaternion $c$ is not close enough to one.
            let im_norm_sqr = c.y * c.y + (c.x * c.x + c.z * c.z);
            if im_norm_sqr >= T::min_positive_value() {
                // Robust computation for negative real part inputs.
                let wx2 = (im_norm_sqr * two / (one - c.w)).sqrt();
                UnitQuaternion(Quaternion::new(wx2 * half, c.x / wx2, c.y / wx2, c.z / wx2))
            } else if c.x.is_zero() && c.y.is_zero() && c.z.is_zero() {
                // Special case: input is -1. The result is `±i` with the same
                // signs for the imaginary parts as the input.
                UnitQuaternion(Quaternion::new(zero, one.copysign(c.x), c.y, c.z))
            } else {
                // `im_norm_sqr` is subnormal, scale up first.
                let s = one / T::min_positive_value();
                let sx = s * c.x;
                let sy = s * c.y;
                let sz = s * c.z;
                let im_norm = (sy * sy + (sx * sx + sz * sz)).sqrt() / s;
                UnitQuaternion(Quaternion::new(
                    im_norm * half,
                    c.x / im_norm,
                    c.y / im_norm,
                    c.z / im_norm,
                ))
            }
        }
    }
}

#[cfg(feature = "serde")]
impl<T> serde::Serialize for UnitQuaternion<T>
where
    T: serde::Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.0.serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de, T> serde::Deserialize<'de> for UnitQuaternion<T>
where
    T: serde::Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let q = serde::Deserialize::deserialize(deserializer)?;
        Ok(Self(q))
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "std")]
    use core::hash::Hash;
    #[cfg(feature = "std")]
    use core::hash::Hasher;
    #[cfg(feature = "std")]
    use std::collections::hash_map::DefaultHasher;

    use num_traits::ConstOne;
    use num_traits::ConstZero;
    use num_traits::FloatConst;
    use num_traits::Inv;
    use num_traits::One;
    use num_traits::Zero;

    #[cfg(any(feature = "std", feature = "libm", feature = "serde"))]
    use crate::EulerAngles;
    use crate::Quaternion;
    use crate::UnitQuaternion;
    use crate::Q32;
    use crate::Q64;
    use crate::UQ32;
    use crate::UQ64;

    #[test]
    fn test_new() {
        let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(q.w, 1.0);
        assert_eq!(q.x, 2.0);
        assert_eq!(q.y, 3.0);
        assert_eq!(q.z, 4.0);
    }

    #[test]
    fn test_zero_constant() {
        let q1 = Q32::ZERO;
        let q2 = Q32::default();
        assert_eq!(q1, q2);
    }

    #[test]
    fn test_const_zero_trait() {
        let q3 = <Q64 as ConstZero>::ZERO;
        let q4 = Q64::default();
        assert_eq!(q3, q4);
    }

    #[test]
    fn test_zero_trait() {
        let q1 = Q32::zero();
        let q2 = Q32::default();
        assert_eq!(q1, q2);
        assert!(q1.is_zero());
    }

    #[test]
    fn test_zero_trait_set_zero() {
        let mut q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
        q.set_zero();
        assert!(q.is_zero());
    }

    #[test]
    fn test_one_constant() {
        assert_eq!(Q32::ONE, Q32::new(1.0, 0.0, 0.0, 0.0))
    }

    #[test]
    fn test_i_constant() {
        assert_eq!(Q64::I, Q64::new(0.0, 1.0, 0.0, 0.0))
    }

    #[test]
    fn test_j_constant() {
        assert_eq!(Q32::J, Q32::new(0.0, 0.0, 1.0, 0.0))
    }

    #[test]
    fn test_k_constant() {
        assert_eq!(Q64::K, Q64::new(0.0, 0.0, 0.0, 1.0))
    }

    #[test]
    fn test_const_one_trait() {
        assert_eq!(<Q32 as ConstOne>::ONE, Q32::new(1.0, 0.0, 0.0, 0.0))
    }

    #[test]
    fn test_one_trait_one() {
        assert_eq!(<Q64 as One>::one(), Q64::ONE);
        assert!(Q64::ONE.is_one());
    }

    #[test]
    fn test_one_trait_set_one() {
        let mut q = Q32::new(2.0, 3.0, 4.0, 5.0);
        q.set_one();
        assert!(q.is_one());
    }

    #[test]
    fn test_i_func() {
        assert_eq!(Q64::i(), Q64::new(0.0, 1.0, 0.0, 0.0));
    }

    #[test]
    fn test_j_func() {
        assert_eq!(Q64::j(), Q64::new(0.0, 0.0, 1.0, 0.0));
    }

    #[test]
    fn test_k_func() {
        assert_eq!(Q64::k(), Q64::new(0.0, 0.0, 0.0, 1.0));
    }

    #[test]
    fn test_norm_sqr() {
        assert_eq!(Q32::ZERO.norm_sqr(), 0.0);
        assert_eq!(Q64::ONE.norm_sqr(), 1.0);
        assert_eq!(Q32::I.norm_sqr(), 1.0);
        assert_eq!(Q64::J.norm_sqr(), 1.0);
        assert_eq!(Q32::K.norm_sqr(), 1.0);
        assert_eq!(Q64::new(-8.0, 4.0, 2.0, -1.0).norm_sqr(), 85.0);
        assert_eq!(Q32::new(1.0, -2.0, 3.0, -4.0).norm_sqr(), 30.0);
    }

    #[test]
    fn test_conj() {
        assert_eq!(Q64::ONE.conj(), Q64::ONE);
        assert_eq!(Q32::I.conj(), -Q32::I);
        assert_eq!(Q64::J.conj(), -Q64::J);
        assert_eq!(Q32::K.conj(), -Q32::K);
        assert_eq!(
            Q64::new(-8.0, 4.0, 2.0, -1.0).conj(),
            Q64::new(-8.0, -4.0, -2.0, 1.0)
        );
        assert_eq!(
            Q32::new(1.0, -2.0, 3.0, -4.0).conj(),
            Q32::new(1.0, 2.0, -3.0, 4.0)
        );
    }

    #[test]
    fn test_inv_func() {
        assert_eq!(Q64::ONE.inv(), Q64::ONE);
        assert_eq!(Q32::I.inv(), -Q32::I);
        assert_eq!(Q64::J.inv(), -Q64::J);
        assert_eq!(Q32::K.inv(), -Q32::K);
        assert_eq!(
            Q64::new(1.0, 1.0, -1.0, -1.0).inv(),
            Q64::new(0.25, -0.25, 0.25, 0.25)
        );
        assert_eq!(
            Q32::new(1.0, -2.0, 2.0, -4.0).inv(),
            Q32::new(0.04, 0.08, -0.08, 0.16)
        );
    }

    #[test]
    fn test_inv_trait_for_ref() {
        assert_eq!(Inv::inv(&Q64::ONE), Q64::ONE);
        assert_eq!(Inv::inv(&Q32::I), -Q32::I);
        assert_eq!(Inv::inv(&Q64::J), -Q64::J);
        assert_eq!(Inv::inv(&Q32::K), -Q32::K);
        assert_eq!(
            Inv::inv(&Q64::new(1.0, 1.0, -1.0, -1.0)),
            Q64::new(0.25, -0.25, 0.25, 0.25)
        );
        assert_eq!(
            Inv::inv(&Q32::new(1.0, -2.0, 2.0, -4.0)),
            Q32::new(0.04, 0.08, -0.08, 0.16)
        );
    }

    #[test]
    fn test_inv_trait_for_val() {
        assert_eq!(Inv::inv(Q64::ONE), Q64::ONE);
        assert_eq!(Inv::inv(Q32::I), -Q32::I);
        assert_eq!(Inv::inv(Q64::J), -Q64::J);
        assert_eq!(Inv::inv(Q32::K), -Q32::K);
        assert_eq!(
            Inv::inv(Q64::new(1.0, 1.0, -1.0, -1.0)),
            Q64::new(0.25, -0.25, 0.25, 0.25)
        );
        assert_eq!(
            Inv::inv(Q32::new(1.0, -2.0, 2.0, -4.0)),
            Q32::new(0.04, 0.08, -0.08, 0.16)
        );
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_norm_normal_values() {
        let q = Q64::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(q.norm(), 30.0f64.sqrt());
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_norm_zero_quaternion() {
        let q = Q32::new(0.0, 0.0, 0.0, 0.0);
        assert_eq!(q.norm(), 0.0, "Norm of zero quaternion should be 0");
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_norm_subnormal_values() {
        let s = f64::MIN_POSITIVE;
        let q = Q64::new(s, s, s, s);
        assert!(
            (q.norm() - 2.0 * s).abs() <= 2.0 * s * f64::EPSILON,
            "Norm of subnormal is computed correctly"
        );
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_norm_infinite_values() {
        let inf = f32::INFINITY;
        assert_eq!(Q32::new(inf, 1.0, 1.0, 1.0).norm(), inf);
        assert_eq!(Q32::new(1.0, inf, 1.0, 1.0).norm(), inf);
        assert_eq!(Q32::new(1.0, 1.0, inf, 1.0).norm(), inf);
        assert_eq!(Q32::new(1.0, 1.0, 1.0, inf).norm(), inf);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_norm_nan_values() {
        let nan = f32::NAN;
        assert!(Q32::new(nan, 1.0, 1.0, 1.0).norm().is_nan());
        assert!(Q32::new(1.0, nan, 1.0, 1.0).norm().is_nan());
        assert!(Q32::new(1.0, 1.0, nan, 1.0).norm().is_nan());
        assert!(Q32::new(1.0, 1.0, 1.0, nan).norm().is_nan());
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_fast_norm_normal_values() {
        let q = Q64 {
            w: 1.1,
            x: 2.7,
            y: 3.4,
            z: 4.9,
        };
        assert_eq!(
            q.fast_norm(),
            q.norm(),
            "Fast norm is equal to norm for normal floating point values"
        );
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_fast_norm_zero_quaternion() {
        assert_eq!(
            Q32::zero().fast_norm(),
            0.0,
            "Fast norm of zero quaternion should be 0"
        );
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_fast_norm_infinite_values() {
        let inf = f32::INFINITY;
        assert_eq!(Q32::new(inf, 1.0, 1.0, 1.0).fast_norm(), inf);
        assert_eq!(Q32::new(1.0, inf, 1.0, 1.0).fast_norm(), inf);
        assert_eq!(Q32::new(1.0, 1.0, inf, 1.0).fast_norm(), inf);
        assert_eq!(Q32::new(1.0, 1.0, 1.0, inf).fast_norm(), inf);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_fast_norm_nan_values() {
        let nan = f32::NAN;
        assert!(Q32::new(nan, 1.0, 1.0, 1.0).fast_norm().is_nan());
        assert!(Q32::new(1.0, nan, 1.0, 1.0).fast_norm().is_nan());
        assert!(Q32::new(1.0, 1.0, nan, 1.0).fast_norm().is_nan());
        assert!(Q32::new(1.0, 1.0, 1.0, nan).fast_norm().is_nan());
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_fast_norm_for_norm_sqr_underflow() {
        let s = f64::MIN_POSITIVE;
        let q = Q64::new(s, s, s, s);
        assert_eq!(q.fast_norm(), 0.0);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_fast_norm_for_norm_sqr_overflow() {
        let s = f32::MAX / 16.0;
        let q = Q32::new(s, s, s, s);
        assert_eq!(q.fast_norm(), f32::INFINITY);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_normalize() {
        assert_eq!(Q64::ONE.normalize().unwrap(), UQ64::ONE);
        assert_eq!(Q32::I.normalize().unwrap(), UQ32::I);
        assert_eq!(Q64::J.normalize().unwrap(), UQ64::J);
        assert_eq!(Q32::K.normalize().unwrap(), UQ32::K);
        assert_eq!(
            Q64::new(9.0, 12.0, -12.0, -16.0)
                .normalize()
                .unwrap()
                .into_quaternion(),
            Q64::new(0.36, 0.48, -0.48, -0.64)
        );
        assert_eq!(
            Q32::new(-1.0, -1.0, 1.0, -1.0)
                .normalize()
                .unwrap()
                .into_quaternion(),
            Q32::new(-0.5, -0.5, 0.5, -0.5)
        );
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_normalize_zero_infinity_nan() {
        assert_eq!(Q64::ZERO.normalize(), None);
        assert_eq!(Q64::new(f64::INFINITY, 0.0, 0.0, 0.0).normalize(), None);
        assert_eq!(Q64::new(0.0, f64::NEG_INFINITY, 0.0, 0.0).normalize(), None);
        assert_eq!(Q64::new(0.0, 0.0, f64::NEG_INFINITY, 0.0).normalize(), None);
        assert_eq!(Q64::new(0.0, 0.0, 0.0, f64::INFINITY).normalize(), None);
        assert_eq!(Q64::new(f64::NAN, 0.0, 0.0, 0.0).normalize(), None);
        assert_eq!(Q64::new(0.0, f64::NAN, 0.0, 0.0).normalize(), None);
        assert_eq!(Q64::new(0.0, 0.0, f64::NAN, 0.0).normalize(), None);
        assert_eq!(Q64::new(0.0, 0.0, 0.0, f64::NAN).normalize(), None);
        assert_eq!(
            Q64::new(f64::INFINITY, f64::NAN, 1.0, 0.0).normalize(),
            None
        );
        assert_eq!(
            Q64::new(1.0, 0.0, f64::INFINITY, f64::NAN).normalize(),
            None
        );
    }

    #[test]
    fn test_from_underlying_type_val() {
        assert_eq!(Q64::from(-5.0), Q64::new(-5.0, 0.0, 0.0, 0.0));
        assert_eq!(Into::<Q32>::into(42.0), Q32::new(42.0, 0.0, 0.0, 0.0));
    }

    #[allow(clippy::needless_borrows_for_generic_args)]
    #[test]
    fn test_from_underlying_type_ref() {
        assert_eq!(Q64::from(&-5.0), Q64::new(-5.0, 0.0, 0.0, 0.0));
        assert_eq!(Into::<Q32>::into(&42.0), Q32::new(42.0, 0.0, 0.0, 0.0));
    }

    #[test]
    fn test_from_unit_quaternion() {
        assert_eq!(Q32::from(UQ32::ONE), Q32::ONE);
        assert_eq!(Q64::from(UQ64::I), Q64::I);
        assert_eq!(Q32::from(UQ32::J), Q32::J);
        assert_eq!(Q64::from(UQ64::K), Q64::K);
    }

    #[allow(clippy::needless_borrows_for_generic_args)]
    #[test]
    fn test_from_unit_quaternion_ref() {
        assert_eq!(<&Q32 as From<&UQ32>>::from(&UQ32::ONE), &Q32::ONE);
        assert_eq!(<&Q64 as From<&UQ64>>::from(&UQ64::I), &Q64::I);
        assert_eq!(<&Q32 as From<&UQ32>>::from(&UQ32::J), &Q32::J);
        assert_eq!(<&Q64 as From<&UQ64>>::from(&UQ64::K), &Q64::K);
    }

    #[test]
    fn test_add_quaternion() {
        assert_eq!(Q32::ONE + Q32::J, Q32::new(1.0, 0.0, 1.0, 0.0));
        assert_eq!(
            Q64::new(1.0, 2.0, 3.0, 4.0) + Q64::new(1.0, 3.0, 10.0, -5.0),
            Q64::new(2.0, 5.0, 13.0, -1.0)
        );
    }

    #[test]
    fn test_add_real() {
        assert_eq!(Q32::I + 1.0, Q32::new(1.0, 1.0, 0.0, 0.0));
        assert_eq!(
            Q32::new(1.0, 2.0, 3.0, 4.0) + 42.0,
            Q32::new(43.0, 2.0, 3.0, 4.0)
        );
    }

    #[test]
    fn test_sub_quaternion() {
        assert_eq!(Q32::ONE - Q32::J, Q32::new(1.0, 0.0, -1.0, 0.0));
        assert_eq!(
            Q64::new(1.0, 2.0, 3.0, 4.0) - Q64::new(1.0, 3.0, 10.0, -5.0),
            Q64::new(0.0, -1.0, -7.0, 9.0)
        );
    }

    #[test]
    fn test_sub_real() {
        assert_eq!(Q32::I - 1.0, Q32::new(-1.0, 1.0, 0.0, 0.0));
        assert_eq!(
            Q32::new(1.0, 2.0, 3.0, 4.0) - 42.0,
            Q32::new(-41.0, 2.0, 3.0, 4.0)
        );
    }

    #[test]
    fn test_mul_quaternion() {
        assert_eq!(Q32::ONE * Q32::ONE, Q32::ONE);
        assert_eq!(Q32::ONE * Q32::I, Q32::I);
        assert_eq!(Q32::ONE * Q32::J, Q32::J);
        assert_eq!(Q32::ONE * Q32::K, Q32::K);
        assert_eq!(Q32::I * Q32::ONE, Q32::I);
        assert_eq!(Q32::J * Q32::ONE, Q32::J);
        assert_eq!(Q32::K * Q32::ONE, Q32::K);
        assert_eq!(Q32::I * Q32::I, -Q32::ONE);
        assert_eq!(Q32::J * Q32::J, -Q32::ONE);
        assert_eq!(Q32::K * Q32::K, -Q32::ONE);
        assert_eq!(Q32::I * Q32::J, Q32::K);
        assert_eq!(Q32::J * Q32::K, Q32::I);
        assert_eq!(Q32::K * Q32::I, Q32::J);
        assert_eq!(Q32::J * Q32::I, -Q32::K);
        assert_eq!(Q32::K * Q32::J, -Q32::I);
        assert_eq!(Q32::I * Q32::K, -Q32::J);
        assert_eq!(
            Q64::new(1.0, 2.0, 3.0, 4.0) * Q64::new(1.0, 3.0, 10.0, -5.0),
            Q64::new(-15.0, -50.0, 35.0, 10.0)
        );
    }

    #[test]
    fn test_mul_real() {
        assert_eq!(Q32::I * 1.0, Q32::I);
        assert_eq!(
            Q32::new(1.0, 2.0, 3.0, 4.0) * 42.0,
            Q32::new(42.0, 84.0, 126.0, 168.0)
        );
    }

    #[test]
    fn test_mul_quaternion_by_unit_quaternion() {
        assert_eq!(Q32::I * UQ32::J, Q32::K);
        assert_eq!(Q64::J * UQ64::K, Q64::I);
        assert_eq!(
            Q32::new(1.0, 2.0, 3.0, 4.0) * UQ32::K,
            Q32::new(-4.0, 3.0, -2.0, 1.0)
        );
    }

    #[test]
    fn test_div_quaternion() {
        assert_eq!(Q32::ONE / Q32::ONE * Q32::ONE, Q32::ONE);
        assert_eq!(Q32::ONE / Q32::I * Q32::I, Q32::ONE);
        assert_eq!(Q32::ONE / Q32::J * Q32::J, Q32::ONE);
        assert_eq!(Q32::ONE / Q32::K * Q32::K, Q32::ONE);
        assert_eq!(Q32::I / Q32::ONE * Q32::ONE, Q32::I);
        assert_eq!(Q32::J / Q32::ONE * Q32::ONE, Q32::J);
        assert_eq!(Q32::K / Q32::ONE * Q32::ONE, Q32::K);
        assert_eq!(Q32::I / Q32::I * Q32::I, Q32::I);
        assert_eq!(Q32::J / Q32::J * Q32::J, Q32::J);
        assert_eq!(Q32::K / Q32::K * Q32::K, Q32::K);
        assert_eq!(Q32::I / Q32::J * Q32::J, Q32::I);
        assert_eq!(Q32::J / Q32::K * Q32::K, Q32::J);
        assert_eq!(Q32::K / Q32::I * Q32::I, Q32::K);
        assert_eq!(Q32::J / Q32::I * Q32::I, Q32::J);
        assert_eq!(Q32::K / Q32::J * Q32::J, Q32::K);
        assert_eq!(Q32::I / Q32::K * Q32::K, Q32::I);
        let q = Q64::new(1.0, 2.0, 3.0, 4.0);
        let r = Q64::new(1.0, 3.0, 10.0, -5.0);
        assert!((q / r * r - q).norm_sqr() < f64::EPSILON);
    }

    #[test]
    fn test_div_real() {
        assert_eq!(Q32::I * 1.0, Q32::I);
        assert_eq!(
            Q32::new(1.0, 2.0, 3.0, 4.0) / 4.0,
            Q32::new(0.25, 0.5, 0.75, 1.0)
        );
    }

    #[test]
    fn test_div_quaternion_by_unit_quaternion() {
        assert_eq!(Q32::I / UQ32::J, -Q32::K);
        assert_eq!(Q64::J / UQ64::K, -Q64::I);
        assert_eq!(
            Q32::new(1.0, 2.0, 3.0, 4.0) / UQ32::K,
            Q32::new(4.0, -3.0, 2.0, -1.0)
        );
    }

    #[test]
    fn test_add_assign() {
        let mut q = Q32::new(1.0, 2.0, 3.0, 4.0);
        q += 4.0;
        assert_eq!(q, Quaternion::new(5.0, 2.0, 3.0, 4.0));
    }

    #[test]
    fn test_sub_assign() {
        let mut q = Q64::new(1.0, 2.0, 3.0, 4.0);
        q -= Q64::new(4.0, 8.0, 6.0, 1.0);
        assert_eq!(q, Quaternion::new(-3.0, -6.0, -3.0, 3.0));
    }

    #[test]
    fn test_mul_assign() {
        let mut q = Q32::new(1.0, 2.0, 3.0, 4.0);
        q *= Q32::I;
        assert_eq!(q, Quaternion::new(-2.0, 1.0, 4.0, -3.0));
    }

    #[test]
    fn test_div_assign() {
        let mut q = Quaternion::new(1.0f32, 2.0f32, 3.0f32, 4.0f32);
        q /= 4.0f32;
        assert_eq!(q, Quaternion::new(0.25f32, 0.5f32, 0.75f32, 1.0f32));
    }

    #[test]
    #[allow(clippy::op_ref)]
    fn test_add_with_ref() {
        let lhs = Quaternion::new(1.0, 2.0, 3.0, 4.0);
        let rhs = Quaternion::new(5.0, 6.0, 7.0, 8.0);
        assert_eq!(lhs + rhs, &lhs + rhs);
        assert_eq!(lhs + rhs, lhs + &rhs);
        assert_eq!(lhs + rhs, &lhs + &rhs);
    }

    #[test]
    #[allow(clippy::op_ref)]
    fn test_sub_with_ref() {
        let lhs = Quaternion::new(1.0, 2.0, 3.0, 4.0);
        let rhs = Quaternion::new(5.0, 6.0, 7.0, 8.0);
        assert_eq!(lhs - rhs, &lhs - rhs);
        assert_eq!(lhs - rhs, lhs - &rhs);
        assert_eq!(lhs - rhs, &lhs - &rhs);
    }

    #[test]
    #[allow(clippy::op_ref)]
    fn test_mul_with_ref() {
        let lhs = Quaternion::new(1.0, 2.0, 3.0, 4.0);
        let rhs = Quaternion::new(5.0, 6.0, 7.0, 8.0);
        assert_eq!(lhs * rhs, &lhs * rhs);
        assert_eq!(lhs * rhs, lhs * &rhs);
        assert_eq!(lhs * rhs, &lhs * &rhs);
    }

    #[test]
    #[allow(clippy::op_ref)]
    fn test_div_with_ref() {
        let lhs = Quaternion::new(1.0, 2.0, 3.0, 4.0);
        let rhs = Quaternion::new(5.0, 6.0, 7.0, 8.0);
        assert_eq!(lhs / rhs, &lhs / rhs);
        assert_eq!(lhs / rhs, lhs / &rhs);
        assert_eq!(lhs / rhs, &lhs / &rhs);
    }

    #[test]
    #[cfg(any(feature = "std", feature = "libm"))]
    #[allow(clippy::op_ref)]
    fn test_add_unit_quaternion_with_ref() {
        let lhs = Quaternion::new(1.0, 2.0, 3.0, 4.0);
        let rhs = Quaternion::new(5.0, 6.0, 7.0, 8.0).normalize().unwrap();
        assert_eq!(lhs + rhs, &lhs + rhs);
        assert_eq!(lhs + rhs, lhs + &rhs);
        assert_eq!(lhs + rhs, &lhs + &rhs);
    }

    #[test]
    #[cfg(any(feature = "std", feature = "libm"))]
    #[allow(clippy::op_ref)]
    fn test_sub_unit_quaternion_with_ref() {
        let lhs = Quaternion::new(1.0, 2.0, 3.0, 4.0);
        let rhs = Quaternion::new(5.0, 6.0, 7.0, 8.0).normalize().unwrap();
        assert_eq!(lhs - rhs, &lhs - rhs);
        assert_eq!(lhs - rhs, lhs - &rhs);
        assert_eq!(lhs - rhs, &lhs - &rhs);
    }

    #[test]
    #[cfg(any(feature = "std", feature = "libm"))]
    #[allow(clippy::op_ref)]
    fn test_mul_unit_quaternion_with_ref() {
        let lhs = Quaternion::new(1.0, 2.0, 3.0, 4.0);
        let rhs = Quaternion::new(5.0, 6.0, 7.0, 8.0).normalize().unwrap();
        assert_eq!(lhs * rhs, &lhs * rhs);
        assert_eq!(lhs * rhs, lhs * &rhs);
        assert_eq!(lhs * rhs, &lhs * &rhs);
    }

    #[test]
    #[cfg(any(feature = "std", feature = "libm"))]
    #[allow(clippy::op_ref)]
    fn test_div_unit_quaternion_with_ref() {
        let lhs = Quaternion::new(1.0, 2.0, 3.0, 4.0);
        let rhs = Quaternion::new(5.0, 6.0, 7.0, 8.0).normalize().unwrap();
        assert_eq!(lhs / rhs, &lhs / rhs);
        assert_eq!(lhs / rhs, lhs / &rhs);
        assert_eq!(lhs / rhs, &lhs / &rhs);
    }

    #[test]
    #[allow(clippy::op_ref)]
    fn test_add_real_with_ref() {
        let lhs = Quaternion::new(1.0, 2.0, 3.0, 4.0);
        let rhs = 5.0;
        assert_eq!(lhs + rhs, &lhs + rhs);
        assert_eq!(lhs + rhs, lhs + &rhs);
        assert_eq!(lhs + rhs, &lhs + &rhs);
    }

    #[test]
    #[allow(clippy::op_ref)]
    fn test_sub_real_with_ref() {
        let lhs = Quaternion::new(1.0, 2.0, 3.0, 4.0);
        let rhs = 5.0;
        assert_eq!(lhs - rhs, &lhs - rhs);
        assert_eq!(lhs - rhs, lhs - &rhs);
        assert_eq!(lhs - rhs, &lhs - &rhs);
    }

    #[test]
    #[allow(clippy::op_ref)]
    fn test_mul_real_with_ref() {
        let lhs = Quaternion::new(1.0, 2.0, 3.0, 4.0);
        let rhs = 5.0;
        assert_eq!(lhs * rhs, &lhs * rhs);
        assert_eq!(lhs * rhs, lhs * &rhs);
        assert_eq!(lhs * rhs, &lhs * &rhs);
    }

    #[test]
    #[allow(clippy::op_ref)]
    fn test_div_real_with_ref() {
        let lhs = Quaternion::new(1.0, 2.0, 3.0, 4.0);
        let rhs = 5.0;
        assert_eq!(lhs / rhs, &lhs / rhs);
        assert_eq!(lhs / rhs, lhs / &rhs);
        assert_eq!(lhs / rhs, &lhs / &rhs);
    }

    #[test]
    fn test_add_lhs_real() {
        assert_eq!(42.0 + Quaternion::I, Quaternion::new(42.0, 1.0, 0.0, 0.0));
        assert_eq!(1 + Quaternion::new(2, 4, 6, 8), Quaternion::new(3, 4, 6, 8));
    }

    #[test]
    fn test_sub_lhs_real() {
        assert_eq!(42.0 - Quaternion::I, Quaternion::new(42.0, -1.0, 0.0, 0.0));
        assert_eq!(
            1 - Quaternion::new(2, 4, 6, 8),
            Quaternion::new(-1, -4, -6, -8)
        );
    }

    #[test]
    fn test_mul_lhs_real() {
        assert_eq!(42.0 * Quaternion::I, Quaternion::new(0.0, 42.0, 0.0, 0.0));
        assert_eq!(2 * Quaternion::new(1, 2, 3, 4), Quaternion::new(2, 4, 6, 8));
    }

    #[test]
    fn test_div_lhs_real() {
        assert_eq!(
            42.0f32 / Quaternion::I,
            Quaternion::new(0.0, -42.0, 0.0, 0.0)
        );
        assert_eq!(
            4.0f64 / Quaternion::new(1.0, 1.0, 1.0, 1.0),
            Quaternion::new(1.0, -1.0, -1.0, -1.0)
        );
    }

    #[test]
    fn test_neg() {
        assert_eq!(
            -Q64::new(1.0, -2.0, 3.0, -4.0),
            Q64::new(-1.0, 2.0, -3.0, 4.0)
        );
    }

    #[test]
    fn test_powu() {
        for q in [
            Q32::ONE,
            Q32::ZERO,
            Q32::I,
            Q32::new(1.0, 1.0, 1.0, 1.0),
            Q32::new(1.0, 2.0, -3.0, 4.0),
        ] {
            let mut expected = Q32::ONE;
            for e in 0..16 {
                assert_eq!(q.powu(e), expected);
                expected *= q;
            }
        }
    }

    #[test]
    fn test_powi() {
        for q in [
            Q32::ONE,
            Q32::I,
            Q32::new(1.0, 1.0, 1.0, 1.0),
            Q32::new(1.0, 2.0, -3.0, 4.0),
        ] {
            let mut expected = Q32::ONE;
            for e in 0..16 {
                assert_eq!(q.powi(e), expected);
                assert!(
                    (q.powi(-e) - expected.inv()).norm_sqr() / expected.norm_sqr() < f32::EPSILON
                );
                expected *= q;
            }
        }
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_exp_zero_quaternion() {
        assert_eq!(Q32::ZERO.exp(), Q32::ONE);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_exp_real_part_only() {
        assert_eq!(Q32::ONE.exp(), core::f32::consts::E.into())
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_exp_imaginary_part_only() {
        assert!(
            (Q64::I.exp() - Q64::new(1.0f64.cos(), 1.0f64.sin(), 0.0, 0.0)).norm() <= f64::EPSILON
        );
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_exp_complex_quaternion() {
        let q = Q32::new(1.0, 1.0, 1.0, 1.0);
        let exp_q = q.exp();
        let expected_norm = 1.0f32.exp();
        let angle = 3.0f32.sqrt();
        assert!(
            (exp_q
                - Q32::new(
                    expected_norm * angle.cos(),
                    expected_norm * angle.sin() / angle,
                    expected_norm * angle.sin() / angle,
                    expected_norm * angle.sin() / angle
                ))
            .norm()
                <= 2.0 * expected_norm * f32::EPSILON
        );
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_exp_negative_real_part() {
        let q = Q64::new(-1000.0, 0.0, f64::INFINITY, f64::NAN);
        let exp_q = q.exp();
        assert_eq!(exp_q, Q64::zero());
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_exp_nan_input() {
        let q = Q32::new(f32::NAN, 1.0, 1.0, 1.0);
        let exp_q = q.exp();
        assert!(exp_q.w.is_nan());
        assert!(exp_q.x.is_nan());
        assert!(exp_q.y.is_nan());
        assert!(exp_q.z.is_nan());
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_exp_large_imaginary_norm() {
        let q = Q32::new(1.0, 1e30, 1e30, 1e30);
        let exp_q = q.exp();
        assert!(exp_q.w.is_nan());
        assert!(exp_q.x.is_nan());
        assert!(exp_q.y.is_nan());
        assert!(exp_q.z.is_nan());
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_exp_infinite_real_part() {
        let inf = f64::INFINITY;
        let q = Quaternion::new(inf, 1.0, 1.0, 1.0);
        let exp_q = q.exp();
        assert_eq!(exp_q.w, -inf);
        assert_eq!(exp_q.x, inf);
        assert_eq!(exp_q.y, inf);
        assert_eq!(exp_q.z, inf);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_exp_infinite_imaginary_part() {
        let q = Q32::new(1.0, f32::INFINITY, 0.0, 0.0);
        let exp_q = q.exp();
        assert!(exp_q.w.is_nan());
        assert!(exp_q.x.is_nan());
        assert!(exp_q.y.is_nan());
        assert!(exp_q.z.is_nan());
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_exp_small_imaginary_norm() {
        let epsilon = f32::EPSILON;
        let q = Quaternion::new(0.5, epsilon, epsilon, epsilon);
        let exp_q = q.exp();
        let result_norm = q.w.exp();
        let expected_exp_q = Quaternion::new(
            result_norm,
            result_norm * q.x,
            result_norm * q.y,
            result_norm * q.z,
        );
        assert!((exp_q - expected_exp_q).norm() <= epsilon);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_exp_infinite_result_angle_greater_than_90_degrees() {
        let angle = f32::PI() * 0.75; // Angle > 90 degrees
        let q = Q32::new(f32::INFINITY, angle, 0.0, 0.0);
        let exp_q = q.exp();
        assert_eq!(exp_q.w, -f32::INFINITY);
        assert_eq!(exp_q.x, f32::INFINITY);
        assert_eq!(exp_q.y, 0.0);
        assert_eq!(exp_q.z, 0.0);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_exp_infinite_result_angle_greater_than_180_degrees() {
        let angle = f64::PI() * 1.25; // Angle > 180 degrees
        let q = Q64::new(f64::INFINITY, 0.0, angle, 0.0);
        let exp_q = q.exp();
        assert_eq!(exp_q.w, -f64::INFINITY);
        assert_eq!(exp_q.x, 0.0);
        assert_eq!(exp_q.y, -f64::INFINITY);
        assert_eq!(exp_q.z, 0.0);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_expf_zero_base_positive_real_quaternion() {
        assert_eq!(Q64::new(1.0, 0.0, 0.0, 0.0).expf(0.0), Q64::ZERO);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_expf_zero_base_negative_real_quaternion() {
        assert_eq!(
            Q32::new(-1.0, 0.0, 0.0, 0.0).expf(0.0),
            f32::INFINITY.into()
        );
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_expf_infinity_base_positive_real_quaternion() {
        let inf = f64::INFINITY;
        assert_eq!(Q64::new(1.0, 0.0, 0.0, 0.0).expf(inf), inf.into());
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_expf_infinity_base_negative_real_quaternion() {
        assert_eq!(
            Q32::new(-1.0, 0.0, 0.0, 0.0).expf(f32::INFINITY),
            0.0f32.into()
        );
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_expf_negative_base() {
        let q = Q64::new(1.0, 0.0, 0.0, 0.0).expf(-1.0);
        assert!(q.w.is_nan());
        assert!(q.x.is_nan());
        assert!(q.y.is_nan());
        assert!(q.z.is_nan());
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_expf_nan_base() {
        let q = Q32::new(1.0, 0.0, 0.0, 0.0).expf(f32::NAN);
        assert!(q.w.is_nan());
        assert!(q.x.is_nan());
        assert!(q.y.is_nan());
        assert!(q.z.is_nan());
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_expf_finite_positive_base() {
        let q = Q64::new(1.0, 2.0, 3.0, 4.0);
        let base = 2.0;
        let result = q.expf(base);
        let expected = (q * base.ln()).exp();
        assert!((result - expected).norm() <= expected.norm() * f64::EPSILON);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_expf_nan_quaternion_component() {
        let q = Q64::new(f64::NAN, 1.0, 1.0, 1.0).expf(3.0);
        assert!(q.w.is_nan());
        assert!(q.x.is_nan());
        assert!(q.y.is_nan());
        assert!(q.z.is_nan());
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_expf_infinity_quaternion_component() {
        let q = Q32::new(1.0, f32::INFINITY, 1.0, 1.0).expf(2.0);
        assert!(q.w.is_nan());
        assert!(q.x.is_nan());
        assert!(q.y.is_nan());
        assert!(q.z.is_nan());
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_expf_infinite_real_component_with_t_greater_than_1() {
        let inf = f64::INFINITY;
        assert_eq!(Q64::new(inf, 0.0, 0.0, 0.0).expf(5.0), inf.into());
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_expf_infinite_real_component_with_t_between_0_and_1() {
        assert_eq!(
            Quaternion::new(f32::NEG_INFINITY, 0.0, 0.0, 0.0).expf(0.5),
            f32::INFINITY.into()
        );
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_ln_normal_case() {
        // Test a normal quaternion
        let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
        let ln_q = q.ln();
        assert!((q.w - 30.0f64.ln() / 2.0) <= 4.0 * f64::EPSILON);
        assert!((ln_q.z / ln_q.x - q.z / q.x) <= 2.0 * f64::EPSILON);
        assert!((ln_q.y / ln_q.x - q.y / q.x) <= 2.0 * f64::EPSILON);
        assert!((ln_q.x.hypot(ln_q.y.hypot(ln_q.z)) - 29.0f64.sqrt().atan()) <= 4.0 * f64::EPSILON);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_ln_positive_real_axis() {
        // Test close to the positive real axis
        let q = Quaternion::new(1.0, 1e-10, 1e-10, 1e-10);
        let ln_q = q.ln();
        let expected = Quaternion::new(0.0, 1e-10, 1e-10, 1e-10); // ln(1) = 0 and imaginary parts small
        assert!((ln_q - expected).norm() <= 1e-11);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_ln_negative_real_axis() {
        // Test close to the negative real axis
        let q = Q32::new(-1.0, 0.0, 0.0, 0.0);
        let ln_q = q.ln();
        let expected = Q32::new(0.0, core::f32::consts::PI, 0.0, 0.0); // ln(-1) = pi*i
        assert!((ln_q - expected).norm() <= core::f32::consts::PI * f32::EPSILON);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_ln_zero() {
        // Test the zero quaternion
        let q = Q32::new(0.0, 0.0, 0.0, 0.0);
        let ln_q = q.ln();
        let expected = f32::NEG_INFINITY.into();
        assert_eq!(ln_q, expected);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_ln_negative_zero() {
        // Test the negative zero quaternion
        let q = Q64::new(-0.0, 0.0, 0.0, 0.0);
        let ln_q = q.ln();
        let expected = Q64::new(f64::NEG_INFINITY, core::f64::consts::PI, 0.0, 0.0);
        assert_eq!(ln_q.w, expected.w);
        assert!((ln_q.x - expected.x).abs() <= core::f64::consts::PI * f64::EPSILON);
        assert_eq!(ln_q.y, expected.y);
        assert_eq!(ln_q.z, expected.z);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_ln_nan() {
        // Test a quaternion with NaN
        let q = Quaternion::new(f32::NAN, 1.0, 1.0, 1.0);
        let ln_q = q.ln();
        assert!(ln_q.w.is_nan());
        assert!(ln_q.x.is_nan());
        assert!(ln_q.y.is_nan());
        assert!(ln_q.z.is_nan());
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_ln_infinite() {
        // Test a quaternion with infinite components
        let q = Q32::new(f32::INFINITY, 1.0, 1.0, 1.0);
        let ln_q = q.ln();
        let expected = Quaternion::new(f32::INFINITY, 0.0, 0.0, 0.0); // Real part infinity, imaginary parts 0
        assert_eq!(ln_q, expected);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_ln_finite_and_infinite() {
        // Test a quaternion with finite and infinite components

        use num_traits::Signed;
        let q = Quaternion::new(1.0, f32::INFINITY, -f32::INFINITY, -1.0);
        let ln_q = q.ln();
        let expected = Quaternion::new(
            f32::INFINITY,
            core::f32::consts::PI / 8.0f32.sqrt(),
            -core::f32::consts::PI / 8.0f32.sqrt(),
            0.0,
        );
        assert_eq!(ln_q.w, expected.w);
        assert!((ln_q.x - expected.x).abs() <= 4.0f32 * f32::EPSILON);
        assert!((ln_q.y - expected.y).abs() <= 4.0f32 * f32::EPSILON);
        assert_eq!(ln_q.z, 0.0);
        assert!(ln_q.z.is_negative());
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_sqrt_normal() {
        let q = Q64::new(1.0, 2.0, 3.0, 4.0);
        assert!(((q * q).sqrt() - q).norm() <= q.norm() * f64::EPSILON);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_sqrt_zero() {
        assert_eq!(Q32::ZERO.sqrt(), Q32::ZERO);
        let zero = Q32::new(-0.0, 0.0, -0.0, -0.0);
        assert!(zero.sqrt().w.is_sign_positive());
        assert!(zero.sqrt().x.is_sign_positive());
        assert!(zero.sqrt().y.is_sign_negative());
        assert!(zero.sqrt().z.is_sign_negative());
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_sqrt_negative_real() {
        let q = Q64::new(-4.0, -0.0, 0.0, 0.0);
        let sqrt_q = q.sqrt();
        let expected = Q64::new(0.0, -2.0, 0.0, 0.0);
        assert_eq!(sqrt_q, expected);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_sqrt_nan() {
        let q = Q32::new(f32::NAN, 0.0, 0.0, 0.0);
        let sqrt_q = q.sqrt();
        assert!(sqrt_q.w.is_nan());
        assert!(sqrt_q.x.is_nan());
        assert!(sqrt_q.y.is_nan());
        assert!(sqrt_q.z.is_nan());
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_sqrt_infinity() {
        let q = Q64::new(f64::INFINITY, -0.0, -0.0, 0.0);
        let sqrt_q = q.sqrt();
        assert!(sqrt_q.w.is_infinite());
        assert!(sqrt_q.x.is_zero());
        assert!(sqrt_q.y.is_zero());
        assert!(sqrt_q.z.is_zero());
        assert!(sqrt_q.w.is_sign_positive());
        assert!(sqrt_q.x.is_sign_negative());
        assert!(sqrt_q.y.is_sign_negative());
        assert!(sqrt_q.z.is_sign_positive());

        let q = Q32::new(0.0, 0.0, -f32::INFINITY, -0.0);
        let sqrt_q = q.sqrt();
        assert!(sqrt_q.w.is_infinite());
        assert!(sqrt_q.x.is_zero());
        assert!(sqrt_q.y.is_infinite());
        assert!(sqrt_q.z.is_zero());
        assert!(sqrt_q.w.is_sign_positive());
        assert!(sqrt_q.x.is_sign_positive());
        assert!(sqrt_q.y.is_sign_negative());
        assert!(sqrt_q.z.is_sign_negative());
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_sqrt_negative_infinity_real() {
        let q = Quaternion::new(-f64::INFINITY, 0.0, -1.0, 0.0);
        let sqrt_q = q.sqrt();
        assert!(sqrt_q.w.is_zero());
        assert!(sqrt_q.x.is_infinite());
        assert!(sqrt_q.y.is_zero());
        assert!(sqrt_q.z.is_zero());
        assert!(sqrt_q.w.is_sign_positive());
        assert!(sqrt_q.x.is_sign_positive());
        assert!(sqrt_q.y.is_sign_negative());
        assert!(sqrt_q.z.is_sign_positive());
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_sqrt_commutativity_with_conjugate() {
        let q = Q32::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(q.conj().sqrt(), q.sqrt().conj());
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_sqrt_subnormal_values() {
        let subnormal = f64::MIN_POSITIVE / 2.0;
        let q = Quaternion::new(subnormal, subnormal, subnormal, subnormal);
        let sqrt_q = q.sqrt();
        let norm_sqr = sqrt_q.norm_sqr();
        assert!((norm_sqr - f64::MIN_POSITIVE).abs() <= 4.0 * subnormal * f64::EPSILON);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_sqrt_mixed_infinities() {
        let q = Q32::new(
            -f32::INFINITY,
            -f32::INFINITY,
            f32::INFINITY,
            -f32::INFINITY,
        );
        let sqrt_q = q.sqrt();
        assert_eq!(sqrt_q.w, f32::INFINITY);
        assert_eq!(sqrt_q.x, -f32::INFINITY);
        assert_eq!(sqrt_q.y, f32::INFINITY);
        assert_eq!(sqrt_q.z, -f32::INFINITY);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_sqrt_positive_real() {
        let q = Q64::new(4.0, 0.0, 0.0, 0.0);
        let sqrt_q = q.sqrt();
        let expected = Q64::new(2.0, 0.0, 0.0, 0.0);
        assert_eq!(sqrt_q, expected);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_sqrt_purely_imaginary() {
        let q = Q32::new(0.0, 3.0, 4.0, 0.0);
        let sqrt_q = q.sqrt();
        assert!(sqrt_q.w > 0.0);
        assert!((sqrt_q * sqrt_q - q).norm() <= 2.0 * q.norm() * f32::EPSILON);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_sqrt_negative_imaginary() {
        let q = Q64::new(0.0, -3.0, -4.0, 0.0);
        let sqrt_q = q.sqrt();
        assert!(sqrt_q.w > 0.0);
        assert!((sqrt_q * sqrt_q - q).norm() <= 16.0 * q.norm() * f64::EPSILON);
    }

    /// Computes the hash value of `val` using the default hasher.
    #[cfg(feature = "std")]
    fn compute_hash(val: impl Hash) -> u64 {
        let mut hasher = DefaultHasher::new();
        val.hash(&mut hasher);
        hasher.finish()
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_serde_quaternion() {
        // Create a sample quaternion
        let q = Q32::new(1.0, 2.0, 3.0, 4.0);

        // Serialize the quaternion to a JSON string
        let serialized = serde_json::to_string(&q).expect("Failed to serialize quaternion");

        // Deserialize the JSON string back into a quaternion
        let deserialized: Quaternion<f32> =
            serde_json::from_str(&serialized).expect("Failed to deserialize quaternion");

        // Assert that the deserialized quaternion is equal to the original
        assert_eq!(q, deserialized);
    }

    // We test if the hash value of a unit quaternion is equal to the hash
    // value of the inner quaternion. This is required because `UnitQuaternion`
    // implements both `Hash` and `Borrow<Quaternion>`.
    #[cfg(feature = "std")]
    #[test]
    fn test_hash_of_unit_quaternion_equals_hash_of_inner_quaternion() {
        assert_eq!(
            compute_hash(UnitQuaternion::<u32>::ONE),
            compute_hash(Quaternion::<u32>::ONE)
        );
        assert_eq!(
            compute_hash(UnitQuaternion::<i32>::I),
            compute_hash(Quaternion::<i32>::I)
        );
        assert_eq!(
            compute_hash(UnitQuaternion::<isize>::J),
            compute_hash(Quaternion::<isize>::J)
        );
        assert_eq!(
            compute_hash(UnitQuaternion::<i128>::K),
            compute_hash(Quaternion::<i128>::K)
        );
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_serde_euler_angles() {
        // Create a sample angles
        let angles = EulerAngles {
            roll: 1.0,
            pitch: 2.0,
            yaw: 3.0,
        };

        // Serialize the angles to a JSON string
        let serialized = serde_json::to_string(&angles).expect("Failed to serialize angles");

        // Deserialize the JSON string back into angles
        let deserialized: EulerAngles<f64> =
            serde_json::from_str(&serialized).expect("Failed to deserialize angles");

        // Assert that the deserialized angles are equal to the original
        assert_eq!(angles, deserialized);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_from_euler_angles() {
        assert!(
            (UQ32::from_euler_angles(core::f32::consts::PI, 0.0, 0.0).into_quaternion() - Q32::I)
                .norm()
                < f32::EPSILON
        );
        assert!(
            (UQ64::from_euler_angles(0.0, core::f64::consts::PI, 0.0).into_quaternion() - Q64::J)
                .norm()
                < f64::EPSILON
        );
        assert!(
            (UQ32::from_euler_angles(0.0, 0.0, core::f32::consts::PI).into_quaternion() - Q32::K)
                .norm()
                < f32::EPSILON
        );
        assert!(
            (UQ64::from_euler_angles(1.0, 2.0, 3.0).into_quaternion()
                - (UQ64::from_euler_angles(0.0, 0.0, 3.0)
                    * UQ64::from_euler_angles(0.0, 2.0, 0.0)
                    * UQ64::from_euler_angles(1.0, 0.0, 0.0))
                .into_quaternion())
            .norm()
                < 4.0 * f64::EPSILON
        );
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_to_euler_angles() {
        let test_data = [
            Q64::new(1.0, 0.0, 0.0, 0.0),
            Q64::new(0.0, 1.0, 0.0, 0.0),
            Q64::new(0.0, 0.0, 1.0, 0.0),
            Q64::new(0.0, 0.0, 0.0, 1.0),
            Q64::new(1.0, 1.0, 1.0, 1.0),
            Q64::new(1.0, -2.0, 3.0, -4.0),
            Q64::new(4.0, 3.0, 2.0, 1.0),
        ];
        for q in test_data.into_iter().map(|q| q.normalize().unwrap()) {
            let EulerAngles { roll, pitch, yaw } = q.to_euler_angles();
            let p = UQ64::from_euler_angles(roll, pitch, yaw);
            assert!((p - q).norm() < f64::EPSILON);
        }
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_from_rotation_vector() {
        assert!(
            (UQ32::from_rotation_vector(&[core::f32::consts::PI, 0.0, 0.0]) - Q32::I).norm()
                < f32::EPSILON
        );
        assert!(
            (UQ64::from_rotation_vector(&[0.0, core::f64::consts::PI, 0.0]) - Q64::J).norm()
                < f64::EPSILON
        );
        assert!(
            (UQ32::from_rotation_vector(&[0.0, 0.0, core::f32::consts::PI]) - Q32::K).norm()
                < f32::EPSILON
        );
        let x = 2.0 * core::f64::consts::FRAC_PI_3 * (1.0f64 / 3.0).sqrt();
        assert!(
            (UQ64::from_rotation_vector(&[x, x, x]) - Q64::new(0.5, 0.5, 0.5, 0.5)).norm()
                < 4.0 * f64::EPSILON
        );
        assert!(
            (UQ64::from_rotation_vector(&[-x, x, -x]) - Q64::new(0.5, -0.5, 0.5, -0.5)).norm()
                < 4.0 * f64::EPSILON
        );
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_to_rotation_vector_zero_rotation() {
        // Quaternion representing no rotation (identity quaternion)
        assert_eq!(UQ32::ONE.to_rotation_vector(), [0.0, 0.0, 0.0]);
        assert_eq!(UQ64::ONE.to_rotation_vector(), [0.0, 0.0, 0.0]);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_to_rotation_vector_90_degree_rotation_x_axis() {
        // Quaternion representing a 90-degree rotation around the x-axis
        let q = Q32::new(1.0, 1.0, 0.0, 0.0).normalize().unwrap();
        let rotation_vector = q.to_rotation_vector();
        assert!((rotation_vector[0] - core::f32::consts::FRAC_PI_2).abs() < f32::EPSILON);
        assert!((rotation_vector[1]).abs() < f32::EPSILON);
        assert!((rotation_vector[2]).abs() < f32::EPSILON);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_to_rotation_vector_180_degree_rotation_y_axis() {
        // Quaternion representing a 180-degree rotation around the x-axis
        let q = UQ64::J;
        let rotation_vector = q.to_rotation_vector();
        assert!((rotation_vector[0]).abs() < f64::EPSILON);
        assert!((rotation_vector[1] - core::f64::consts::PI).abs() < f64::EPSILON);
        assert!((rotation_vector[2]).abs() < f64::EPSILON);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_to_rotation_vector_180_degree_rotation_arbitrary_axis() {
        // Quaternion representing a 180-degree rotation around an arbitrary axis
        let q = Q32::new(0.0, 1.0, 1.0, 1.0).normalize().unwrap();
        let rotation_vector = q.to_rotation_vector();
        let expected = [
            core::f32::consts::PI / (1.0f32 + 1.0 + 1.0).sqrt(),
            core::f32::consts::PI / (1.0f32 + 1.0 + 1.0).sqrt(),
            core::f32::consts::PI / (1.0f32 + 1.0 + 1.0).sqrt(),
        ];
        assert!((rotation_vector[0] - expected[0]).abs() < 4.0 * f32::EPSILON);
        assert!((rotation_vector[1] - expected[1]).abs() < 4.0 * f32::EPSILON);
        assert!((rotation_vector[2] - expected[2]).abs() < 4.0 * f32::EPSILON);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_to_rotation_vector_small_rotation() {
        // Quaternion representing a small rotation
        let angle = 1e-6f32;
        let q = Q32::new((angle / 2.0).cos(), (angle / 2.0).sin(), 0.0, 0.0)
            .normalize()
            .unwrap();
        let rotation_vector = q.to_rotation_vector();
        assert!((rotation_vector[0] - angle).abs() < f32::EPSILON);
        assert!((rotation_vector[1]).abs() < f32::EPSILON);
        assert!((rotation_vector[2]).abs() < f32::EPSILON);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_to_rotation_vector_general_case() {
        // Quaternion representing a general rotation
        // Here we first compute the rotation vector and then
        // check if `from_rotation_vector` restores the original
        // quaternion appropriately.
        for q in [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, -1.0],
            [1.0, 0.0, 2.0, 5.0],
            [1.0, 0.0, 1.0e-10, 2.0e-10],
            [-1.0, 0.0, 0.0, 0.0],
            [1.0, f64::EPSILON, 0.0, 0.0],
            [1.0, 0.0, 0.0, f64::MIN_POSITIVE],
            [
                -1.0,
                3.0 * f64::MIN_POSITIVE,
                2.0 * f64::MIN_POSITIVE,
                f64::MIN_POSITIVE,
            ],
        ]
        .into_iter()
        .map(|[w, x, y, z]| Q64::new(w, x, y, z).normalize().unwrap())
        {
            let rot = q.to_rotation_vector();
            let p = UQ64::from_rotation_vector(&rot);
            assert!((p - q).norm() <= 6.0 * f64::EPSILON);
        }
    }

    #[test]
    fn test_rotation_matrix_identity() {
        let q = UQ64::ONE;
        let rot_matrix = q.to_rotation_matrix3x3();
        let expected = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        assert_eq!(rot_matrix, expected);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_rotation_matrix_90_degrees_x() {
        let q = Q64::new(1.0, 1.0, 0.0, 0.0).normalize().unwrap();
        let rot_matrix = q.to_rotation_matrix3x3();
        let expected = [
            1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, //
            0.0, -1.0, 0.0,
        ];
        for (r, e) in rot_matrix.iter().zip(expected) {
            assert!((r - e).abs() <= f64::EPSILON);
        }
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_rotation_matrix_90_degrees_y() {
        let q = Q64::new(1.0, 0.0, 1.0, 0.0).normalize().unwrap();
        let rot_matrix = q.to_rotation_matrix3x3();
        let expected = [
            0.0, 0.0, -1.0, //
            0.0, 1.0, 0.0, //
            1.0, 0.0, 0.0,
        ];
        for (r, e) in rot_matrix.iter().zip(expected) {
            assert!((r - e).abs() <= f64::EPSILON);
        }
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_rotation_matrix_90_degrees_z() {
        let q = Q64::new(1.0, 0.0, 0.0, 1.0).normalize().unwrap();
        let rot_matrix = q.to_rotation_matrix3x3();
        let expected = [
            0.0, 1.0, 0.0, //
            -1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0,
        ];
        for (r, e) in rot_matrix.iter().zip(expected) {
            assert!((r - e).abs() <= f64::EPSILON);
        }
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_rotation_matrix_120_degrees_xyz() {
        let q = Q64::new(1.0, 1.0, 1.0, 1.0).normalize().unwrap();
        let rot_matrix = q.to_rotation_matrix3x3();
        let expected = [
            0.0, 1.0, 0.0, //
            0.0, 0.0, 1.0, //
            1.0, 0.0, 0.0,
        ];
        for (r, e) in rot_matrix.iter().zip(expected) {
            assert!((r - e).abs() <= f64::EPSILON);
        }
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_rotation_matrix_general() {
        let q = Q64::new(-1.0, 2.0, -3.0, 4.0).normalize().unwrap();
        let rot_matrix = q.to_rotation_matrix3x3();
        let [x1, y1, z1] = q.rotate_vector([1.0, 0.0, 0.0]);
        let [x2, y2, z2] = q.rotate_vector([0.0, 1.0, 0.0]);
        let [x3, y3, z3] = q.rotate_vector([0.0, 0.0, 1.0]);
        let expected = [
            x1, x2, x3, //
            y1, y2, y3, //
            z1, z2, z3,
        ];
        for (r, e) in rot_matrix.iter().zip(expected) {
            assert!((r - e).abs() <= f64::EPSILON);
        }
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_identity_matrix() {
        let identity: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let q = UQ32::from_rotation_matrix3x3(&identity);
        let expected = UQ32::ONE;
        assert_eq!(q, expected);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_rotation_x() {
        let angle = core::f32::consts::PI / 5.0;
        let rotation_x: [[f32; 3]; 3] = [
            [1.0, 0.0, 0.0],
            [0.0, angle.cos(), angle.sin()],
            [0.0, -angle.sin(), angle.cos()],
        ];
        let q = UQ32::from_rotation_matrix3x3(&rotation_x);
        let expected = UQ32::from_rotation_vector(&[angle, 0.0, 0.0]);
        assert!((q - expected).norm() <= f32::EPSILON);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_rotation_y() {
        let angle = 4.0 * core::f32::consts::PI / 5.0;
        let rotation_y: [[f32; 3]; 3] = [
            [angle.cos(), 0.0, -angle.sin()],
            [0.0, 1.0, 0.0],
            [angle.sin(), 0.0, angle.cos()],
        ];
        let q = UQ32::from_rotation_matrix3x3(&rotation_y);
        let expected = UQ32::from_rotation_vector(&[0.0, angle, 0.0]);
        assert!((q - expected).norm() <= f32::EPSILON);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_rotation_z() {
        let angle = 3.0 * core::f64::consts::PI / 5.0;
        let rotation_z: [[f64; 3]; 3] = [
            [angle.cos(), angle.sin(), 0.0],
            [-angle.sin(), angle.cos(), 0.0],
            [0.0, 0.0, 1.0],
        ];
        let q = UQ64::from_rotation_matrix3x3(&rotation_z);
        let expected = UQ64::from_rotation_vector(&[0.0, 0.0, angle]);
        assert!((q - expected).norm() <= f64::EPSILON);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_arbitrary_rotation() {
        let arbitrary_rotation: [[f32; 3]; 3] = [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]];
        let q = UnitQuaternion::from_rotation_matrix3x3(&arbitrary_rotation);
        let expected = Q32::new(1.0, 1.0, 1.0, 1.0).normalize().unwrap();
        assert!((q - expected).norm() <= f32::EPSILON);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_flat_array() {
        let angle = core::f32::consts::PI / 2.0;
        let rotation_z: [f32; 9] = [0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let q = UQ32::from_rotation_matrix3x3(&rotation_z);
        let expected = UQ32::from_rotation_vector(&[0.0, 0.0, angle]);
        assert!((q - expected).norm() <= f32::EPSILON);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_from_rotation_to_rotation() {
        let mat = [
            0.36f64, 0.864, -0.352, 0.48, 0.152, 0.864, 0.8, -0.48, -0.36,
        ];
        let q = UnitQuaternion::from_rotation_matrix3x3(&mat);
        let restored_mat = q.to_rotation_matrix3x3();
        for (x, e) in restored_mat.iter().zip(mat) {
            assert!((x - e).abs() <= 4.0 * f64::EPSILON);
        }
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_zero_vector_a() {
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let q = UnitQuaternion::from_two_vectors(&a, &b);
        assert_eq!(q, UnitQuaternion::one());
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_zero_vector_b() {
        let a = [1.0, 0.0, 0.0];
        let b = [0.0, 0.0, 0.0];
        let q = UnitQuaternion::from_two_vectors(&a, &b);
        assert_eq!(q, UnitQuaternion::one());
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_parallel_vectors() {
        let a = [1.0, 0.0, 0.0];
        let b = [2.0, 0.0, 0.0];
        let q = UnitQuaternion::from_two_vectors(&a, &b);
        assert_eq!(q, UnitQuaternion::one());
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_opposite_vectors() {
        let a = [1.0, 0.0, 0.0];
        let b = [-1.0, 0.0, 0.0];
        let q = UnitQuaternion::from_two_vectors(&a, &b);
        assert_eq!(q.as_quaternion().w, 0.0);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_perpendicular_vectors() {
        let a = [1.0f32, 0.0, 0.0];
        let b = [0.0f32, 1.0, 0.0];
        let q = UQ32::from_two_vectors(&a, &b);
        let expected = Q32::new(1.0, 0.0, 0.0, 1.0).normalize().unwrap();
        assert!((q - expected).norm() <= 2.0 * f32::EPSILON);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_non_normalized_vectors() {
        let a = [0.0, 3.0, 0.0];
        let b = [0.0, 5.0, 5.0];
        let q = UQ64::from_two_vectors(&a, &b);
        let expected = Q64::new(1.0, core::f64::consts::FRAC_PI_8.tan(), 0.0, 0.0)
            .normalize()
            .unwrap();
        assert!((q - expected).norm() <= 2.0 * f64::EPSILON);

        let a = [0.0, 3.0, 0.0];
        let b = [0.0, -5.0, 5.0];
        let q = UQ64::from_two_vectors(&a, &b);
        let expected = Q64::new(1.0, (3.0 * core::f64::consts::FRAC_PI_8).tan(), 0.0, 0.0)
            .normalize()
            .unwrap();
        assert!((q - expected).norm() <= 2.0 * f64::EPSILON);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_same_vector() {
        let a = [1.0, 1.0, 1.0];
        let q = UnitQuaternion::from_two_vectors(&a, &a);
        assert_eq!(q, UnitQuaternion::one());
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_arbitrary_vectors() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let q = UQ64::from_two_vectors(&a, &b);
        let v = [-3.0, 6.0, -3.0]; // cross product
        let v_norm = 54.0f64.sqrt();
        let dir = [v[0] / v_norm, v[1] / v_norm, v[2] / v_norm];
        let cos_angle = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2])
            / ((a[0] * a[0] + a[1] * a[1] + a[2] * a[2])
                * (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]))
                .sqrt();
        let angle = cos_angle.acos();
        let expected =
            UQ64::from_rotation_vector(&[dir[0] * angle, dir[1] * angle, dir[2] * angle]);
        assert!((q - expected).norm() <= 2.0 * f64::EPSILON);
    }

    #[test]
    fn test_default_unit_quaternion() {
        assert_eq!(UQ32::default().into_quaternion(), Q32::ONE);
    }

    #[test]
    fn test_constant_one() {
        assert_eq!(UQ32::ONE.into_quaternion(), Q32::ONE);
        assert_eq!(
            UnitQuaternion::<i32>::ONE.into_quaternion(),
            Quaternion::<i32>::ONE
        );
    }

    #[test]
    fn test_constant_i() {
        assert_eq!(UQ32::I.into_quaternion(), Q32::I);
    }

    #[test]
    fn test_constant_j() {
        assert_eq!(UQ32::J.into_quaternion(), Q32::J);
    }

    #[test]
    fn test_constant_k() {
        assert_eq!(UQ32::K.into_quaternion(), Q32::K);
    }

    #[test]
    fn test_const_one() {
        assert_eq!(<UQ32 as ConstOne>::ONE.into_quaternion(), Q32::ONE);
    }

    #[test]
    fn test_one_trait() {
        assert_eq!(<UQ32 as One>::one().into_quaternion(), Q32::ONE);
        assert!(UQ64::ONE.is_one());
        assert!(!UQ64::I.is_one());
        assert!(!UQ64::J.is_one());
        assert!(!UQ64::K.is_one());
        let mut uq = UQ32::I;
        uq.set_one();
        assert!(uq.is_one());
    }

    #[test]
    fn test_unit_quaternion_i_func() {
        assert_eq!(UQ32::i().into_quaternion(), Q32::i());
    }

    #[test]
    fn test_unit_quaternion_j_func() {
        assert_eq!(UQ32::j().into_quaternion(), Q32::j());
    }

    #[test]
    fn test_unit_quaternion_k_func() {
        assert_eq!(UQ32::k().into_quaternion(), Q32::k());
    }

    #[test]
    fn test_unit_quaternion_conj() {
        assert_eq!(UQ32::ONE.conj(), UQ32::ONE);
        assert_eq!(UQ64::I.conj(), -UQ64::I);
        assert_eq!(UQ32::J.conj(), -UQ32::J);
        assert_eq!(UQ64::K.conj(), -UQ64::K);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_unit_quaternion_conj_with_normalize() {
        assert_eq!(
            Q32::new(1.0, 2.0, 3.0, 4.0).normalize().unwrap().conj(),
            Q32::new(1.0, -2.0, -3.0, -4.0).normalize().unwrap()
        )
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_unit_quaternion_inv_func() {
        assert_eq!(
            Q32::new(1.0, 2.0, 3.0, 4.0).normalize().unwrap().inv(),
            Q32::new(1.0, 2.0, 3.0, 4.0).normalize().unwrap().conj()
        )
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_unit_quaternion_inv_trait() {
        assert_eq!(
            <UQ32 as Inv>::inv(Q32::new(1.0, 2.0, 3.0, 4.0).normalize().unwrap()),
            Q32::new(1.0, 2.0, 3.0, 4.0).normalize().unwrap().conj()
        )
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_unit_quaternion_ref_inv_trait() {
        assert_eq!(
            <&UQ32 as Inv>::inv(&Q32::new(1.0, 2.0, 3.0, 4.0).normalize().unwrap()),
            Q32::new(1.0, 2.0, 3.0, 4.0).normalize().unwrap().conj()
        )
    }

    #[test]
    fn test_unit_quaternion_add() {
        assert_eq!(UQ32::I + UQ32::J, Q32::new(0.0, 1.0, 1.0, 0.0));
    }

    #[test]
    fn test_unit_quaternion_add_quaternion() {
        assert_eq!(UQ32::J + Q32::K, Q32::new(0.0, 0.0, 1.0, 1.0));
    }

    #[test]
    fn test_unit_quaternion_add_underlying() {
        assert_eq!(UQ32::J + 2.0f32, Q32::new(2.0, 0.0, 1.0, 0.0));
    }

    #[test]
    fn test_f32_add_unit_quaternion() {
        assert_eq!(3.0f32 + UQ32::K, Q32::new(3.0, 0.0, 0.0, 1.0));
    }

    #[test]
    fn test_f64_add_unit_quaternion() {
        assert_eq!(4.0f64 + UQ64::I, Q64::new(4.0, 1.0, 0.0, 0.0));
    }

    #[test]
    fn test_unit_quaternion_sub() {
        assert_eq!(UQ32::I - UQ32::J, Q32::new(0.0, 1.0, -1.0, 0.0));
    }

    #[test]
    fn test_unit_quaternion_sub_quaternion() {
        assert_eq!(UQ32::J - Q32::K, Q32::new(0.0, 0.0, 1.0, -1.0));
    }

    #[test]
    fn test_unit_quaternion_sub_underlying() {
        assert_eq!(UQ32::J - 2.0f32, Q32::new(-2.0, 0.0, 1.0, 0.0));
    }

    #[test]
    fn test_f32_sub_unit_quaternion() {
        assert_eq!(3.0f32 - UQ32::K, Q32::new(3.0, 0.0, 0.0, -1.0));
    }

    #[test]
    fn test_f64_sub_unit_quaternion() {
        assert_eq!(4.0f64 - UQ64::I, Q64::new(4.0, -1.0, 0.0, 0.0));
    }

    #[test]
    fn test_unit_quaternion_mul() {
        assert_eq!(UQ32::I * UQ32::J, UQ32::K);
    }

    #[test]
    fn test_unit_quaternion_mul_quaternion() {
        assert_eq!(UQ32::J * Q32::K, Q32::new(0.0, 1.0, 0.0, 0.0));
    }

    #[test]
    fn test_unit_quaternion_mul_underlying() {
        assert_eq!(UQ32::J * 2.0f32, Q32::new(0.0, 0.0, 2.0, 0.0));
    }

    #[test]
    fn test_f32_mul_unit_quaternion() {
        assert_eq!(3.0f32 * UQ32::K, Q32::new(0.0, 0.0, 0.0, 3.0));
    }

    #[test]
    fn test_f64_mul_unit_quaternion() {
        assert_eq!(4.0f64 * UQ64::I, Q64::new(0.0, 4.0, 0.0, 0.0));
    }

    #[test]
    fn test_unit_quaternion_div() {
        assert_eq!(UQ32::I / UQ32::J, -UQ32::K);
    }

    #[test]
    fn test_unit_quaternion_div_quaternion() {
        assert_eq!(UQ32::J / Q32::K, Q32::new(0.0, -1.0, 0.0, 0.0));
    }

    #[test]
    fn test_unit_quaternion_div_underlying() {
        assert_eq!(UQ32::J / 2.0f32, Q32::new(0.0, 0.0, 0.5, 0.0));
    }

    #[test]
    fn test_f32_div_unit_quaternion() {
        assert_eq!(3.0f32 / UQ32::K, Q32::new(0.0, 0.0, 0.0, -3.0));
    }

    #[test]
    fn test_f64_div_unit_quaternion() {
        assert_eq!(4.0f64 / UQ64::I, Q64::new(0.0, -4.0, 0.0, 0.0));
    }

    #[test]
    #[cfg(any(feature = "std", feature = "libm"))]
    #[allow(clippy::op_ref)]
    fn test_add_with_ref_unit_quaternion() {
        let lhs = Quaternion::new(1.0, 2.0, 3.0, 4.0).normalize().unwrap();
        let rhs = Quaternion::new(5.0, 6.0, 7.0, 8.0).normalize().unwrap();
        assert_eq!(lhs + rhs, &lhs + rhs);
        assert_eq!(lhs + rhs, lhs + &rhs);
        assert_eq!(lhs + rhs, &lhs + &rhs);
    }

    #[test]
    #[cfg(any(feature = "std", feature = "libm"))]
    #[allow(clippy::op_ref)]
    fn test_sub_with_ref_unit_quaternion() {
        let lhs = Quaternion::new(1.0, 2.0, 3.0, 4.0).normalize().unwrap();
        let rhs = Quaternion::new(5.0, 6.0, 7.0, 8.0).normalize().unwrap();
        assert_eq!(lhs - rhs, &lhs - rhs);
        assert_eq!(lhs - rhs, lhs - &rhs);
        assert_eq!(lhs - rhs, &lhs - &rhs);
    }

    #[test]
    #[cfg(any(feature = "std", feature = "libm"))]
    #[allow(clippy::op_ref)]
    fn test_mul_with_ref_unit_quaternion() {
        let lhs = Quaternion::new(1.0, 2.0, 3.0, 4.0).normalize().unwrap();
        let rhs = Quaternion::new(5.0, 6.0, 7.0, 8.0).normalize().unwrap();
        assert_eq!(lhs * rhs, &lhs * rhs);
        assert_eq!(lhs * rhs, lhs * &rhs);
        assert_eq!(lhs * rhs, &lhs * &rhs);
    }

    #[test]
    #[cfg(any(feature = "std", feature = "libm"))]
    #[allow(clippy::op_ref)]
    fn test_div_with_ref_unit_quaternion() {
        let lhs = Quaternion::new(1.0, 2.0, 3.0, 4.0).normalize().unwrap();
        let rhs = Quaternion::new(5.0, 6.0, 7.0, 8.0).normalize().unwrap();
        assert_eq!(lhs / rhs, &lhs / rhs);
        assert_eq!(lhs / rhs, lhs / &rhs);
        assert_eq!(lhs / rhs, &lhs / &rhs);
    }

    #[test]
    #[cfg(any(feature = "std", feature = "libm"))]
    #[allow(clippy::op_ref)]
    fn test_add_quaternion_with_ref_unit_quaternion() {
        let lhs = Quaternion::new(1.0, 2.0, 3.0, 4.0).normalize().unwrap();
        let rhs = Quaternion::new(5.0, 6.0, 7.0, 8.0);
        assert_eq!(lhs + rhs, &lhs + rhs);
        assert_eq!(lhs + rhs, lhs + &rhs);
        assert_eq!(lhs + rhs, &lhs + &rhs);
    }

    #[test]
    #[cfg(any(feature = "std", feature = "libm"))]
    #[allow(clippy::op_ref)]
    fn test_sub_quaternion_with_ref_unit_quaternion() {
        let lhs = Quaternion::new(1.0, 2.0, 3.0, 4.0).normalize().unwrap();
        let rhs = Quaternion::new(5.0, 6.0, 7.0, 8.0);
        assert_eq!(lhs - rhs, &lhs - rhs);
        assert_eq!(lhs - rhs, lhs - &rhs);
        assert_eq!(lhs - rhs, &lhs - &rhs);
    }

    #[test]
    #[cfg(any(feature = "std", feature = "libm"))]
    #[allow(clippy::op_ref)]
    fn test_mul_quaternion_with_ref_unit_quaternion() {
        let lhs = Quaternion::new(1.0, 2.0, 3.0, 4.0).normalize().unwrap();
        let rhs = Quaternion::new(5.0, 6.0, 7.0, 8.0);
        assert_eq!(lhs * rhs, &lhs * rhs);
        assert_eq!(lhs * rhs, lhs * &rhs);
        assert_eq!(lhs * rhs, &lhs * &rhs);
    }

    #[test]
    #[cfg(any(feature = "std", feature = "libm"))]
    #[allow(clippy::op_ref)]
    fn test_div_quaternion_with_ref_unit_quaternion() {
        let lhs = Quaternion::new(1.0, 2.0, 3.0, 4.0).normalize().unwrap();
        let rhs = Quaternion::new(5.0, 6.0, 7.0, 8.0);
        assert_eq!(lhs / rhs, &lhs / rhs);
        assert_eq!(lhs / rhs, lhs / &rhs);
        assert_eq!(lhs / rhs, &lhs / &rhs);
    }

    #[test]
    #[cfg(any(feature = "std", feature = "libm"))]
    #[allow(clippy::op_ref)]
    fn test_add_real_with_ref_unit_quaternion() {
        let lhs = Quaternion::new(1.0, 2.0, 3.0, 4.0).normalize().unwrap();
        let rhs = 5.0;
        assert_eq!(lhs + rhs, &lhs + rhs);
        assert_eq!(lhs + rhs, lhs + &rhs);
        assert_eq!(lhs + rhs, &lhs + &rhs);
    }

    #[test]
    #[cfg(any(feature = "std", feature = "libm"))]
    #[allow(clippy::op_ref)]
    fn test_sub_real_with_ref_unit_quaternion() {
        let lhs = Quaternion::new(1.0, 2.0, 3.0, 4.0).normalize().unwrap();
        let rhs = 5.0;
        assert_eq!(lhs - rhs, &lhs - rhs);
        assert_eq!(lhs - rhs, lhs - &rhs);
        assert_eq!(lhs - rhs, &lhs - &rhs);
    }

    #[test]
    #[cfg(any(feature = "std", feature = "libm"))]
    #[allow(clippy::op_ref)]
    fn test_mul_real_with_ref_unit_quaternion() {
        let lhs = Quaternion::new(1.0, 2.0, 3.0, 4.0).normalize().unwrap();
        let rhs = 5.0;
        assert_eq!(lhs * rhs, &lhs * rhs);
        assert_eq!(lhs * rhs, lhs * &rhs);
        assert_eq!(lhs * rhs, &lhs * &rhs);
    }

    #[test]
    #[cfg(any(feature = "std", feature = "libm"))]
    #[allow(clippy::op_ref)]
    fn test_div_real_with_ref_unit_quaternion() {
        let lhs = Quaternion::new(1.0, 2.0, 3.0, 4.0).normalize().unwrap();
        let rhs = 5.0;
        assert_eq!(lhs / rhs, &lhs / rhs);
        assert_eq!(lhs / rhs, lhs / &rhs);
        assert_eq!(lhs / rhs, &lhs / &rhs);
    }

    #[test]
    fn test_unit_quaternion_neg() {
        assert_eq!(
            (-UQ32::ONE).into_quaternion(),
            Q32::new(-1.0, 0.0, 0.0, 0.0)
        );
        assert_eq!((-UQ32::I).into_quaternion(), Q32::new(0.0, -1.0, 0.0, 0.0));
        assert_eq!((-UQ32::J).into_quaternion(), Q32::new(0.0, 0.0, -1.0, 0.0));
        assert_eq!((-UQ32::K).into_quaternion(), Q32::new(0.0, 0.0, 0.0, -1.0));
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_unit_quaternion_adjust_norm() {
        let mut q = UQ32::from_euler_angles(1.0, 0.5, 1.5);
        for _ in 0..25 {
            q = q * q;
        }
        assert!((q.into_quaternion().norm() - 1.0).abs() > 0.5);
        assert!((q.adjust_norm().into_quaternion().norm() - 1.0).abs() <= 2.0 * f32::EPSILON);
    }

    #[test]
    fn test_unit_quaternion_rotate_vector_units() {
        let v = [1.0, 2.0, 3.0];
        assert_eq!(UQ32::I.rotate_vector(v), [1.0, -2.0, -3.0]);
        assert_eq!(UQ32::J.rotate_vector(v), [-1.0, 2.0, -3.0]);
        assert_eq!(UQ32::K.rotate_vector(v), [-1.0, -2.0, 3.0]);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_unit_quaternion_rotate_vector_normalized() {
        let q = Q32::new(1.0, 1.0, 1.0, 1.0).normalize().unwrap();
        let v = [1.0, 2.0, 3.0];
        let result = q.rotate_vector(v);
        assert_eq!(result, [2.0, 3.0, 1.0]);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    fn generate_unit_quaternion_data() -> impl Iterator<Item = UQ32> {
        [
            UQ32::ONE,
            UQ32::I,
            UQ32::J,
            UQ32::K,
            Q32::new(1.0, 1.0, 1.0, 1.0).normalize().unwrap(),
            Q32::new(10.0, 1.0, 1.0, 1.0).normalize().unwrap(),
            Q32::new(1.0, 10.0, 1.0, 1.0).normalize().unwrap(),
            Q32::new(1.0, 1.0, 3.0, 4.0).normalize().unwrap(),
            Q32::new(1.0, -1.0, 3.0, -4.0).normalize().unwrap(),
        ]
        .into_iter()
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_slerp_t_zero() {
        for q1 in generate_unit_quaternion_data() {
            for q2 in generate_unit_quaternion_data() {
                let result = q1.slerp(&q2, 0.0);
                assert!((result - q1).norm() <= f32::EPSILON);
            }
        }
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_slerp_t_one() {
        use core::cmp::Ordering;

        for q1 in generate_unit_quaternion_data() {
            for q2 in generate_unit_quaternion_data() {
                let result = q1.slerp(&q2, 1.0);
                match q1.dot(q2).partial_cmp(&0.0) {
                    Some(Ordering::Greater) => assert!((result - q2).norm() <= f32::EPSILON),
                    Some(Ordering::Less) => assert!((result + q2).norm() <= f32::EPSILON),
                    _ => {}
                }
            }
        }
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_slerp_t_half() {
        use core::cmp::Ordering;

        for q1 in generate_unit_quaternion_data() {
            for q2 in generate_unit_quaternion_data() {
                let result = q1.slerp(&q2, 0.5);
                let dot_sign = match q1.dot(q2).partial_cmp(&0.0) {
                    Some(Ordering::Greater) => 1.0,
                    Some(Ordering::Less) => -1.0,
                    _ => continue, // uncertain due to rounding, better skip it
                };
                assert!((result - (q1 + dot_sign * q2).normalize().unwrap()).norm() <= f32::EPSILON)
            }
        }
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_slerp_small_angle() {
        let q1 = UQ32::ONE;
        let q2 = Q32::new(999_999.0, 1.0, 0.0, 0.0).normalize().unwrap();
        let t = 0.5;
        let result = q1.slerp(&q2, t);
        let expected = Q32::new(999_999.75, 0.5, 0.0, 0.0).normalize().unwrap();
        assert!((result - expected).norm() <= f32::EPSILON);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_sqrt_of_identity() {
        assert_eq!(UQ32::ONE.sqrt(), UQ32::ONE);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_sqrt_of_negative_identity() {
        let q = Q64::new(-1.0, 0.0, -0.0, -0.0).normalize().unwrap();
        assert_eq!(q.sqrt(), UQ64::I);
        assert!(q.sqrt().0.w.is_sign_positive());
        assert!(q.sqrt().0.y.is_sign_negative());
        assert!(q.sqrt().0.z.is_sign_negative());

        let q = Q64::new(-1.0, -0.0, 0.0, 0.0).normalize().unwrap();
        assert_eq!(q.sqrt(), -UQ64::I);
        assert!(q.sqrt().0.w.is_sign_positive());
        assert!(q.sqrt().0.y.is_sign_positive());
        assert!(q.sqrt().0.z.is_sign_positive());
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_sqrt_general_case() {
        let c = Q64::new(1.0, 2.0, -3.0, 4.0).normalize().unwrap();
        let q = c.sqrt();
        assert!((q * q - c).norm() <= f64::EPSILON);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_sqrt_with_negative_real_part() {
        let c = Q64::new(-4.0, 2.0, -3.0, 1.0).normalize().unwrap();
        let q = c.sqrt();
        assert!((q * q - c).norm() <= f64::EPSILON);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_sqrt_with_subnormal_imaginary_parts() {
        let min_positive = f64::MIN_POSITIVE;
        let q = UnitQuaternion(Quaternion::new(
            -1.0,
            min_positive,
            min_positive,
            min_positive,
        ));
        let result = q.sqrt();
        let expected = UnitQuaternion(Quaternion::new(
            min_positive * 0.75f64.sqrt(),
            (1.0f64 / 3.0).sqrt(),
            (1.0f64 / 3.0).sqrt(),
            (1.0f64 / 3.0).sqrt(),
        ));
        assert!((result.0.w - expected.0.w).abs() <= 2.0 * expected.0.w * f64::EPSILON);
        assert!((result.0.x - expected.0.x).abs() <= 2.0 * expected.0.x * f64::EPSILON);
        assert!((result.0.y - expected.0.y).abs() <= 2.0 * expected.0.y * f64::EPSILON);
        assert!((result.0.z - expected.0.z).abs() <= 2.0 * expected.0.z * f64::EPSILON);
    }

    #[cfg(all(feature = "serde", any(feature = "std", feature = "libm")))]
    #[test]
    fn test_serde_unit_quaternion() {
        // Create a sample quaternion
        let q = Q64::new(1.0, 2.0, 3.0, 4.0).normalize().unwrap();

        // Serialize the quaternion to a JSON string
        let serialized = serde_json::to_string(&q).expect("Failed to serialize quaternion");

        // Deserialize the JSON string back into a quaternion
        let deserialized: UQ64 =
            serde_json::from_str(&serialized).expect("Failed to deserialize quaternion");

        // Assert that the deserialized quaternion is equal to the original
        assert_eq!(q, deserialized);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_serde_unit_quaternion_k() {
        // Create a sample quaternion
        let q = UQ64::K;

        // Serialize the quaternion to a JSON string
        let serialized = serde_json::to_string(&q).expect("Failed to serialize quaternion");

        // Deserialize the JSON string back into a quaternion
        let deserialized: UQ64 =
            serde_json::from_str(&serialized).expect("Failed to deserialize quaternion");

        // Assert that the deserialized quaternion is equal to the original
        assert_eq!(q, deserialized);
    }
}
