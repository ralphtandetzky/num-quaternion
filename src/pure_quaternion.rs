#![cfg(feature = "unstable")]

use core::ops::{Add, Mul};

use num_traits::{ConstZero, Zero};

#[cfg(any(feature = "std", feature = "libm"))]
use num_traits::Float;

/// A pure quaternion, i.e. a quaternion with a real part of zero.
///
/// A pure quaternion is a quaternion of the form $bi + cj + dk$.
/// Computations with pure quaternions can be more efficient than with general
/// quaternions. Apart from that, pure quaternions are used to represent
/// 3D vectors and provide the compile-time guarantee that the real part is
/// zero.
///
/// The `PureQuaternion` struct is kept as similar as possible to the
/// [`Quaternion`] struct with respect to its API. It provides
///
///   - a constructor [`PureQuaternion::new`] to create a new pure quaternion,
///   - member data fields `x`, `y`, and `z` to access the coefficients of $i$,
///     $j$, and $k$, respectively,
///   - The types aliases [`PQ32`] and [`PQ64`] are provided for
///     `PureQuaternion<f32>` and `PureQuaternion<f64>`, respectively.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct PureQuaternion<T> {
    /// The coefficient of $i$.
    pub x: T,
    /// The coefficient of $j$.
    pub y: T,
    /// The coefficient of $k$.
    pub z: T,
}

/// Alias for a [`PureQuaternion<f32>`].
pub type PQ32 = PureQuaternion<f32>;
/// Alias for a [`PureQuaternion<f64>`].
pub type PQ64 = PureQuaternion<f64>;

impl<T> PureQuaternion<T> {
    /// Constructs a new pure quaternion.
    ///
    /// # Examples
    ///
    /// ```
    /// # use num_quaternion::PureQuaternion;
    /// let pq = PureQuaternion::new(1.0, 2.0, 3.0);
    /// ```
    #[inline]
    pub const fn new(x: T, y: T, z: T) -> Self {
        Self { x, y, z }
    }
}

impl<T> PureQuaternion<T>
where
    T: ConstZero,
{
    /// Constructs a new pure quaternion with all components set to zero.
    ///
    /// # Examples
    ///
    /// ```
    /// # use num_quaternion::PureQuaternion;
    /// let pq = PureQuaternion::ZERO;
    /// assert_eq!(pq, PureQuaternion::new(0.0, 0.0, 0.0));
    /// ```
    pub const ZERO: Self = Self::new(T::ZERO, T::ZERO, T::ZERO);
}

// impl<T> ConstZero for PureQuaternion<T>
// where
//     T: ConstZero,
// {
//     const ZERO: Self = Self::ZERO;
// }

// impl<T> Zero for PureQuaternion<T>
// where
//     T: Zero,
// {
//     #[inline]
//     fn zero() -> Self {
//         Self::new(T::zero(), T::zero(), T::zero())
//     }

//     #[inline]
//     fn is_zero(&self) -> bool {
//         self.x.is_zero() && self.y.is_zero() && self.z.is_zero()
//     }

//     #[inline]
//     fn set_zero(&mut self) {
//         self.x.set_zero();
//         self.y.set_zero();
//         self.z.set_zero();
//     }
// }

#[cfg(any(feature = "std", feature = "libm"))]
impl<T> PureQuaternion<T>
where
    T: Float,
{
    /// Returns a pure quaternion filled with `NaN` values.
    ///
    /// # Example
    ///
    /// ```
    /// # use num_quaternion::PQ32;
    /// let q = PQ32::nan();
    /// assert!(q.x.is_nan());
    /// assert!(q.y.is_nan());
    /// assert!(q.z.is_nan());
    /// ```
    #[inline]
    pub fn nan() -> Self {
        let nan = T::nan();
        Self::new(nan, nan, nan)
    }
}

impl<T> PureQuaternion<T>
where
    T: Clone + Mul<T, Output = T> + Add<T, Output = T>,
{
    /// Returns the square of the norm.
    ///
    /// The result is $x^2 + y^2 + z^2$ with some rounding errors.
    /// The rounding error is at most 2
    /// [ulps](https://en.wikipedia.org/wiki/Unit_in_the_last_place).
    ///
    /// This is guaranteed to be more efficient than [`norm`](Quaternion::norm()).
    /// Furthermore, `T` only needs to support addition and multiplication
    /// and therefore, this function works for more types than
    /// [`norm`](Quaternion::norm()).
    ///
    /// # Example
    ///
    /// ```
    /// # use num_quaternion::PureQuaternion;
    /// let q = PureQuaternion::new(1.0f32, 2.0, 3.0);
    /// assert_eq!(q.norm_sqr(), 14.0);
    /// ```
    #[inline]
    pub fn norm_sqr(&self) -> T {
        self.x.clone() * self.x.clone()
            + self.y.clone() * self.y.clone()
            + self.z.clone() * self.z.clone()
    }
}
