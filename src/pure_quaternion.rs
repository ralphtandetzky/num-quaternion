#![cfg(feature = "unstable")]

use core::ops::{Add, Mul, Neg};

use num_traits::{ConstOne, ConstZero, Inv, Num, One, Zero};

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

impl<T> ConstZero for PureQuaternion<T>
where
    T: ConstZero,
{
    const ZERO: Self = Self::ZERO;
}

impl<T> Zero for PureQuaternion<T>
where
    T: Zero,
{
    #[inline]
    fn zero() -> Self {
        Self::new(T::zero(), T::zero(), T::zero())
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.x.is_zero() && self.y.is_zero() && self.z.is_zero()
    }

    #[inline]
    fn set_zero(&mut self) {
        self.x.set_zero();
        self.y.set_zero();
        self.z.set_zero();
    }
}

impl<T> PureQuaternion<T>
where
    T: ConstZero + ConstOne,
{
    /// A constant `PureQuaternion` of value $i$.
    ///
    /// See also [`Quaternion::I`](crate::Quaternion::I), [`PureQuaternion::i`].
    ///
    /// # Example
    ///
    /// ```
    /// # use num_quaternion::PureQuaternion;
    /// let q = PureQuaternion::I;
    /// assert_eq!(q, PureQuaternion::new(1.0, 0.0, 0.0));
    /// ```
    pub const I: Self = Self::new(T::ONE, T::ZERO, T::ZERO);

    /// A constant `PureQuaternion` of value $j$.
    ///
    /// See also [`Quaternion::J`](crate::Quaternion::J), [`PureQuaternion::j`].
    ///
    /// # Example
    ///
    /// ```
    /// # use num_quaternion::PureQuaternion;
    /// let q = PureQuaternion::J;
    /// assert_eq!(q, PureQuaternion::new(0.0, 1.0, 0.0));
    /// ```
    pub const J: Self = Self::new(T::ZERO, T::ONE, T::ZERO);

    /// A constant `PureQuaternion` of value $k$.
    ///
    /// See also [`Quaternion::K`](crate::Quaternion::K), [`PureQuaternion::k`].
    ///
    /// # Example
    ///
    /// ```
    /// # use num_quaternion::PureQuaternion;
    /// let q = PureQuaternion::K;
    /// assert_eq!(q, PureQuaternion::new(0.0, 0.0, 1.0));
    /// ```
    pub const K: Self = Self::new(T::ZERO, T::ZERO, T::ONE);
}

impl<T> PureQuaternion<T>
where
    T: Zero + One,
{
    /// Returns the imaginary unit $i$.
    ///
    /// See also [`Quaternion::i`](crate::Quaternion::i), [`PureQuaternion::I`].
    ///
    /// # Example
    ///
    /// ```
    /// # use num_quaternion::PureQuaternion;
    /// let q = PureQuaternion::i();
    /// assert_eq!(q, PureQuaternion::new(1.0, 0.0, 0.0));
    /// ```
    #[inline]
    pub fn i() -> Self {
        Self::new(T::one(), T::zero(), T::zero())
    }

    /// Returns the imaginary unit $j$.
    ///
    /// See also [`Quaternion::j`](crate::Quaternion::j), [`PureQuaternion::J`].
    ///
    /// # Example
    ///
    /// ```
    /// # use num_quaternion::PureQuaternion;
    /// let q = PureQuaternion::j();
    /// assert_eq!(q, PureQuaternion::new(0.0, 1.0, 0.0));
    /// ```
    #[inline]
    pub fn j() -> Self {
        Self::new(T::zero(), T::one(), T::zero())
    }

    /// Returns the imaginary unit $k$.
    ///
    /// See also [`Quaternion::k`](crate::Quaternion::k), [`PureQuaternion::K`].
    ///
    /// # Example
    ///
    /// ```
    /// # use num_quaternion::PureQuaternion;
    /// let q = PureQuaternion::k();
    /// assert_eq!(q, PureQuaternion::new(0.0, 0.0, 1.0));
    /// ```
    #[inline]
    pub fn k() -> Self {
        Self::new(T::zero(), T::zero(), T::one())
    }
}

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

impl<T> PureQuaternion<T>
where
    T: Clone + Neg<Output = T>,
{
    /// Returns the conjugate of the pure quaternion, i. e. its negation.
    ///
    /// # Example
    ///
    /// ```
    /// # use num_quaternion::PureQuaternion;
    /// let q = PureQuaternion::new(1.0f32, 2.0, 3.0);
    /// assert_eq!(q.conj(), PureQuaternion::new(-1.0, -2.0, -3.0));
    /// ```
    #[inline]
    pub fn conj(&self) -> Self {
        Self::new(-self.x.clone(), -self.y.clone(), -self.z.clone())
    }
}

impl<T> PureQuaternion<T>
where
    for<'a> &'a Self: Inv<Output = PureQuaternion<T>>,
{
    /// Returns the inverse of the pure quaternion.
    ///
    /// # Example
    ///
    /// ```
    /// # use num_quaternion::PureQuaternion;
    /// let q = PureQuaternion::new(1.0f32, 2.0, 3.0);
    /// assert_eq!(q.inv(), PureQuaternion::new(
    ///     -1.0 / 14.0, -2.0 / 14.0, -3.0 / 14.0));
    /// ```
    #[inline]
    pub fn inv(&self) -> Self {
        Inv::inv(self)
    }
}

impl<T> Inv for &PureQuaternion<T>
where
    T: Clone + Neg<Output = T> + Num,
{
    type Output = PureQuaternion<T>;

    #[inline]
    fn inv(self) -> Self::Output {
        let norm_sqr = self.norm_sqr();
        PureQuaternion::new(
            -self.x.clone() / norm_sqr.clone(),
            -self.y.clone() / norm_sqr.clone(),
            -self.z.clone() / norm_sqr,
        )
    }
}

impl<T> Inv for PureQuaternion<T>
where
    for<'a> &'a Self: Inv<Output = PureQuaternion<T>>,
{
    type Output = PureQuaternion<T>;

    #[inline]
    fn inv(self) -> Self::Output {
        Inv::inv(&self)
    }
}

#[cfg(any(feature = "std", feature = "libm"))]
impl<T> PureQuaternion<T>
where
    T: Float,
{
    /// Calculates |self|.
    ///
    /// The result is $\sqrt{x^2+y^2+z^2}$ with some possible rounding
    /// errors. The total relative rounding error is at most two
    /// [ulps](https://en.wikipedia.org/wiki/Unit_in_the_last_place).
    ///
    /// If any of the components of the input quaternion is `NaN`, then `NaN`
    /// is returned. Otherwise, if any of the components is infinite, then
    /// a positive infinite value is returned.
    ///
    /// # Example
    ///
    /// ```
    /// # use num_quaternion::PureQuaternion;
    /// let q = PureQuaternion::new(1.0f32, 2.0, 3.0);
    /// assert_eq!(q.norm(), 14.0f32.sqrt());
    /// ```
    #[inline]
    pub fn norm(self) -> T {
        let one = T::one();
        let two = one + one;
        let s = T::min_positive_value();
        let norm_sqr = self.norm_sqr();
        if norm_sqr < T::infinity() {
            if norm_sqr >= s * two {
                norm_sqr.sqrt()
            } else if self.is_zero() {
                // Likely, the whole vector is zero. If so, we can return
                // zero directly and avoid expensive floating point math.
                T::zero()
            } else {
                // Otherwise, scale up, such that the norm will be in the
                // normal floating point range, then scale down the result.
                (self / s).fast_norm() * s
            }
        } else {
            // There are three possible cases:
            //   1. one of x, y, z is NaN,
            //   2. neither is `NaN`, but at least one of them is infinite, or
            //   3. all of them are finite.
            // In the first case, multiplying by s or dividing by it does not
            // change the that the result is `NaN`. The same applies in the
            // second case: the result remains infinite. In the third case,
            // multiplying by s makes sure that the square norm is a normal
            // floating point number. Dividing by it will rescale the result
            // to the correct magnitude.
            (self * s).fast_norm() / s
        }
    }
}

#[cfg(any(feature = "std", feature = "libm"))]
impl<T> PureQuaternion<T>
where
    T: Float,
{
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
    ///   with reduced relative precision), then the result is the square
    ///   root of that.
    ///
    /// In other words, this function can be imprecise for very large and very
    /// small floating point numbers, but it is generally faster than
    /// [`norm`](Self::norm), because it does not do any branching. So if you
    /// are interested in maximum speed of your code, feel free to use this
    /// function. If you need to be precise results for the whole range of the
    /// floating point type `T`, stay with [`norm`](Self::norm).
    ///
    /// # Example
    ///
    /// ```
    /// # use num_quaternion::PureQuaternion;
    /// let q = PureQuaternion::new(1.0f32, 2.0, 3.0);
    /// assert_eq!(q.fast_norm(), q.norm());
    /// ```
    #[inline]
    pub fn fast_norm(self) -> T {
        self.norm_sqr().sqrt()
    }
}
