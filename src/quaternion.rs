use {
    crate::UnitQuaternion,
    core::ops::{Add, Mul, Neg},
    num_traits::{ConstOne, ConstZero, Inv, Num, One, Zero},
};

#[cfg(any(feature = "std", feature = "libm"))]
use {
    core::num::FpCategory,
    num_traits::{Float, FloatConst},
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
/// norm with [`norm`](Quaternion::norm) or [`fast_norm`](Quaternion::fast_norm)
/// and its square with [`norm_sqr`](Quaternion::norm_sqr). Quaternion
/// conjugation is done by the member function [`conj`](Quaternion::conj).
/// You can normalize a quaternion by calling
/// [`normalize`](Quaternion::normalize), which returns a [`UnitQuaternion`].
///
/// Furthermore, the following functions are supported:
///
/// - [`dot`](Quaternion::dot): Computes the dot product of two quaternions.
/// - [`exp`](Quaternion::exp): Computes the exponential of a quaternion.
/// - [`expf`](Quaternion::expf): Raises a real value to a quaternion power.
/// - [`inv`](Quaternion::inv): Computes the multiplicative inverse.
/// - [`ln`](Quaternion::ln): Computes the natural logarithm of a quaternion.
/// - [`powf`](Quaternion::powf): Raises a quaternion to a real power.
/// - [`powi`](Quaternion::powi): Raises a quaternion to a signed integer power.
/// - [`powu`](Quaternion::powu): Raises a quaternion to an unsigned integer power.
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
    ///
    /// # Example
    ///
    /// ```
    /// # use num_quaternion::Quaternion;
    /// let q = Quaternion::new(1.0f32, 2.0, 3.0, 4.0);
    /// ```
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
    ///
    /// This is the additive identity element of the quaternion space.
    /// See also [`Quaternion::zero`].
    ///
    /// # Example
    ///
    /// ```
    /// # use num_quaternion::Quaternion;
    /// let q = Quaternion::ZERO;
    /// assert_eq!(q, Quaternion::new(0.0f32, 0.0, 0.0, 0.0));
    /// ```
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
        self.w.is_zero()
            && self.x.is_zero()
            && self.y.is_zero()
            && self.z.is_zero()
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
    ///
    /// # Example
    ///
    /// ```
    /// # use num_quaternion::Quaternion;
    /// let q = Quaternion::ONE;
    /// assert_eq!(q, Quaternion::new(1.0f32, 0.0, 0.0, 0.0));
    /// ```
    pub const ONE: Self = Self::new(T::ONE, T::ZERO, T::ZERO, T::ZERO);

    /// A constant `Quaternion` of value $i$.
    ///
    /// See also [`Quaternion::i`].
    ///
    /// # Example
    ///
    /// ```
    /// # use num_quaternion::Quaternion;
    /// let q = Quaternion::I;
    /// assert_eq!(q, Quaternion::new(0.0f32, 1.0, 0.0, 0.0));
    /// ```
    pub const I: Self = Self::new(T::ZERO, T::ONE, T::ZERO, T::ZERO);

    /// A constant `Quaternion` of value $j$.
    ///
    /// See also [`Quaternion::j`].
    ///
    /// # Example
    ///
    /// ```
    /// # use num_quaternion::Quaternion;
    /// let q = Quaternion::J;
    /// assert_eq!(q, Quaternion::new(0.0f32, 0.0, 1.0, 0.0));
    /// ```
    pub const J: Self = Self::new(T::ZERO, T::ZERO, T::ONE, T::ZERO);

    /// A constant `Quaternion` of value $k$.
    ///
    /// See also [`Quaternion::k`].
    ///
    /// # Example
    ///
    /// ```
    /// # use num_quaternion::Quaternion;
    /// let q = Quaternion::K;
    /// assert_eq!(q, Quaternion::new(0.0f32, 0.0, 0.0, 1.0));
    /// ```
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
        self.w.is_one()
            && self.x.is_zero()
            && self.y.is_zero()
            && self.z.is_zero()
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
    /// Returns the real unit $1$.
    ///
    /// See also [`Quaternion::ONE`].
    ///
    /// # Example
    ///
    /// ```
    /// # use num_quaternion::Quaternion;
    /// let q = Quaternion::one();
    /// assert_eq!(q, Quaternion::new(1.0f32, 0.0, 0.0, 0.0));
    /// ```
    #[inline]
    pub fn one() -> Self {
        Self::new(T::one(), T::zero(), T::zero(), T::zero())
    }

    /// Returns the imaginary unit $i$.
    ///
    /// See also [`Quaternion::I`].
    ///
    /// # Example
    ///
    /// ```
    /// # use num_quaternion::Quaternion;
    /// let q = Quaternion::i();
    /// assert_eq!(q, Quaternion::new(0.0f32, 1.0, 0.0, 0.0));
    /// ```
    #[inline]
    pub fn i() -> Self {
        Self::new(T::zero(), T::one(), T::zero(), T::zero())
    }

    /// Returns the imaginary unit $j$.
    ///
    /// See also [`Quaternion::J`].
    ///
    /// # Example
    ///
    /// ```
    /// # use num_quaternion::Quaternion;
    /// let q = Quaternion::j();
    /// assert_eq!(q, Quaternion::new(0.0f32, 0.0, 1.0, 0.0));
    /// ```
    #[inline]
    pub fn j() -> Self {
        Self::new(T::zero(), T::zero(), T::one(), T::zero())
    }

    /// Returns the imaginary unit $k$.
    ///
    /// See also [`Quaternion::K`].
    ///
    /// # Example
    ///
    /// ```
    /// # use num_quaternion::Quaternion;
    /// let q = Quaternion::k();
    /// assert_eq!(q, Quaternion::new(0.0f32, 0.0, 0.0, 1.0));
    /// ```
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
    /// Returns a quaternion filled with `NaN` values.
    ///
    /// # Example
    ///
    /// ```
    /// # use num_quaternion::Q32;
    /// let q = Q32::nan();
    /// assert!(q.w.is_nan());
    /// assert!(q.x.is_nan());
    /// assert!(q.y.is_nan());
    /// assert!(q.z.is_nan());
    /// ```
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
    ///
    /// # Example
    ///
    /// ```
    /// # use num_quaternion::Quaternion;
    /// let q = Quaternion::new(1.0f32, 2.0, 3.0, 4.0);
    /// assert_eq!(q.norm_sqr(), 30.0);
    /// ```
    #[inline]
    pub fn norm_sqr(&self) -> T {
        (self.w.clone() * self.w.clone() + self.y.clone() * self.y.clone())
            + (self.x.clone() * self.x.clone()
                + self.z.clone() * self.z.clone())
    }
}

impl<T> Quaternion<T>
where
    T: Clone + Neg<Output = T>,
{
    /// Returns the conjugate quaternion, i. e. the imaginary part is negated.
    ///
    /// # Example
    ///
    /// ```
    /// # use num_quaternion::Quaternion;
    /// let q = Quaternion::new(1.0f32, 2.0, 3.0, 4.0);
    /// assert_eq!(q.conj(), Quaternion::new(1.0, -2.0, -3.0, -4.0));
    /// ```
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
    ///
    /// # Example
    ///
    /// ```
    /// # use num_quaternion::Quaternion;
    /// let q = Quaternion::new(1.0f32, 2.0, 3.0, 4.0);
    /// assert_eq!(q.inv(), Quaternion::new(
    ///     1.0 / 30.0, -1.0 / 15.0, -0.1, -2.0 / 15.0));
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
    ///
    /// # Example
    ///
    /// ```
    /// # use num_quaternion::Quaternion;
    /// let q = Quaternion::new(1.0f32, 2.0, 3.0, 4.0);
    /// assert_eq!(q.norm(), 30.0f32.sqrt());
    /// ```
    #[inline]
    pub fn norm(self) -> T {
        let norm_sqr = self.norm_sqr();
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
}

#[cfg(any(feature = "std", feature = "libm"))]
impl<T> Quaternion<T>
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
    /// # use num_quaternion::Quaternion;
    /// let q = Quaternion::new(1.0f32, 2.0, 3.0, 4.0);
    /// assert_eq!(q.fast_norm(), q.norm());
    /// ```
    #[inline]
    pub fn fast_norm(self) -> T {
        self.norm_sqr().sqrt()
    }
}

#[cfg(any(feature = "std", feature = "libm"))]
impl<T> Quaternion<T>
where
    T: Float,
{
    /// Normalizes the quaternion to length $1$.
    ///
    /// The sign of the real part will be the same as the sign of the input.
    /// If the input quaternion
    ///
    /// * is zero, or
    /// * has infinite length, or
    /// * has a `NaN` value,
    ///
    /// then `None` will be returned.
    ///
    /// # Example
    ///
    /// ```
    /// # use num_quaternion::Quaternion;
    /// # fn test() -> Option<()> {
    /// let q = Quaternion::new(1.0f32, 2.0, 2.0, 4.0);
    /// assert_eq!(q.normalize()?.into_quaternion(),
    ///         Quaternion::new(0.2f32, 0.4, 0.4, 0.8));
    /// # Some(())
    /// # }
    /// # assert!(test().is_some());
    /// ```
    #[inline]
    pub fn normalize(self) -> Option<UnitQuaternion<T>> {
        UnitQuaternion::normalize(self)
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
        q.into_inner()
    }
}

impl<'a, T> From<&'a UnitQuaternion<T>> for &'a Quaternion<T> {
    #[inline]
    fn from(q: &'a UnitQuaternion<T>) -> Self {
        q.as_quaternion()
    }
}

impl<T> Quaternion<T>
where
    T: Add<T, Output = T> + Mul<T, Output = T>,
{
    /// Computes the dot product of two quaternions interpreted as
    /// 4D real vectors.
    ///
    /// # Example
    ///
    /// ```
    /// # use num_quaternion::Quaternion;
    /// let q1 = Quaternion::new(1.0f32, 2.0, 3.0, 4.0);
    /// let q2 = Quaternion::new(0.0f32, 0.0, 1.0, 1.0);
    /// let d = q1.dot(q2);
    /// assert_eq!(d, 7.0);
    /// ```
    #[inline]
    pub fn dot(self, other: Self) -> T {
        self.w * other.w
            + self.y * other.y
            + (self.x * other.x + self.z * other.z)
    }
}

impl<T> Quaternion<T>
where
    T: Num + Clone,
{
    /// Raises `self` to an unsigned integer power `n`, i. e. $q^n$.
    ///
    /// # Example
    ///
    /// ```
    /// # use num_quaternion::Quaternion;
    /// let q = Quaternion::new(1.0f32, 2.0, 3.0, 4.0);
    /// let q_sqr = q.powu(2);
    /// assert_eq!(q_sqr, q * q);
    /// ```
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
    ///
    /// # Example
    ///
    /// ```
    /// # use num_quaternion::Quaternion;
    /// let q = Quaternion::new(1.0f32, 2.0, 3.0, 4.0);
    /// let q_sqr = q.powi(2);
    /// assert_eq!(q_sqr, q * q);
    /// ```
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
    ///
    /// # Example
    ///
    /// ```
    /// # use num_quaternion::Quaternion;
    /// let q = Quaternion::new(1.0f32, 2.0, 3.0, 4.0);
    /// let exp_q = q.exp();
    /// ```
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
            Some(core::cmp::Ordering::Less) => {
                if result_norm.is_zero() {
                    return Self::zero();
                }
                // Case: 0 < result_norm < ∞

                // Compute the squared norm of the imaginary part
                let sqr_angle =
                    self.x * self.x + self.y * self.y + self.z * self.z;

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
                let sqr_angle =
                    self.x * self.x + self.y * self.y + self.z * self.z;
                if sqr_angle < T::PI() * T::PI() * quarter {
                    // Angle less than 90 degrees
                    Self::new(inf, map(self.x), map(self.y), map(self.z))
                } else if sqr_angle.is_finite() {
                    // Angle 90 degrees or more -> careful sign handling
                    let angle = sqr_angle.sqrt();
                    let angle_revolutions_fract =
                        (angle * T::FRAC_1_PI() * half).fract();
                    let cos_angle_signum =
                        (angle_revolutions_fract - half).abs() - quarter;
                    let sin_angle_signum =
                        one.copysign(half - angle_revolutions_fract);
                    Self::new(
                        inf.copysign(cos_angle_signum),
                        map(self.x) * sin_angle_signum,
                        map(self.y) * sin_angle_signum,
                        map(self.z) * sin_angle_signum,
                    )
                } else {
                    // Angle is super large or NaN
                    debug_assert!(
                        sqr_angle.is_infinite() || sqr_angle.is_nan()
                    );
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
}

#[cfg(any(feature = "std", feature = "libm"))]
impl<T> Quaternion<T>
where
    T: Float + FloatConst,
{
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
    ///
    /// # Example
    ///
    /// ```
    /// # use num_quaternion::Quaternion;
    /// let q = Quaternion::new(1.0f32, 2.0, 3.0, 4.0);
    /// let exp_q = q.expf(2.0);
    /// ```
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
}

#[cfg(any(feature = "std", feature = "libm"))]
impl<T> Quaternion<T>
where
    T: Float + FloatConst,
{
    /// Raises a quaternion (`self`) to a real value (`exponent`) power.
    ///
    /// Given a quaternion $q$ and a real value $t$, this function computes
    /// $q^t := e^{t \ln q}$. The function handles special cases as follows:
    ///
    /// - If $q = 0$ and $t > 0$, a zero quaternion is returned.
    /// - If $|q| = \infty$ and $t < 0$, a zero quaternion is returned.
    /// - If $|q| = \infty$ and $t > 0$ is not too large to prevent
    ///   numerical accuracy for the direction of the quaternion, then an
    ///   infinite quaternion without `NaN` components is returned. For larger
    ///   but finite values of $t$ this may still be hold, or alternatively a
    ///   quaternion filled with `NaN` values is returned.
    /// - If $q = +\infty$ and $t = +\infty$, then positive infinity is
    ///   returned.
    /// - If $|q| = \infty$, but $q \neq +\infty$ and $t = +\infty$, then `NaN`
    ///   is returned.
    /// - If $q = 0$ and $t < 0$, then positive infinity is returned.
    /// - If $q$ contains a `NaN` component, or if $t$ is `NaN`, a `NaN`
    ///   quaternion is returned.
    /// - If $|q| = \infty$ and $t = 0$, a `NaN` quaternion is returned.
    /// - If $q = 0$ and $t = 0$, a `NaN` quaternion is returned.
    ///
    /// For non-zero finite $q$, the following conventions for boundary values
    /// of $t$ are applied:
    ///
    /// - If $t = +\infty$ and $q, |q| \ge 1$ is not real or $q = 1$ or
    ///   $q \le -1$, a `NaN` quaternion is returned.
    /// - If $t = +\infty$ and $q > 1$, positive infinity is returned.
    /// - If $t = +\infty$ and $|q| < 1$, zero is returned.
    /// - If $t = -\infty$ and $|q| > 1$, zero is returned.
    /// - If $t = -\infty$ and $0 \le q < 1$, positive infinity is returned.
    /// - If $t = -\infty$ and $q, |q| \le 1$ is not real or $q = 1$ or
    ///   $-1 \le q < 0$, a `NaN` quaternion is returned.
    ///
    /// If the true result's norm is neither greater than the largest
    /// representable floating point value nor less than the smallest
    /// representable floating point value, and the direction of the output
    /// quaternion cannot be accurately determined, a `NaN` quaternion may or
    /// may not be returned to indicate inaccuracy. This can occur when
    /// $\|t \Im(\ln q)\|$ is on the order of $1/\varepsilon$, where
    /// $\varepsilon$ is the machine precision of the floating point type used.
    ///
    /// # Example
    ///
    /// ```
    /// # use num_quaternion::Quaternion;
    /// let q = Quaternion::new(1.0f32, 2.0, 3.0, 4.0);
    /// let exp_q = q.powf(2.0);
    /// ```
    #[inline]
    pub fn powf(self, exponent: T) -> Self {
        if exponent.is_finite() {
            // -∞ < t < +∞ ==> apply the general formula.
            (self.ln() * exponent).exp()
        } else if exponent > T::zero() {
            // t = +∞
            if self.x.is_zero() && self.y.is_zero() && self.z.is_zero() {
                // q is real --> handle special cases
                match self.w.partial_cmp(&T::one()) {
                    Some(core::cmp::Ordering::Greater) => T::infinity().into(),
                    Some(core::cmp::Ordering::Less) => T::zero().into(),
                    _ => Self::nan(),
                }
            } else if self.norm_sqr() < T::one() {
                // |q| < 1
                Self::zero()
            } else {
                // Otherwise, return NaN
                Self::nan()
            }
        } else if exponent < T::zero() {
            // t = -∞
            if self.x.is_zero() && self.y.is_zero() && self.z.is_zero() {
                // q is real --> handle special cases
                match self.w.partial_cmp(&T::one()) {
                    Some(core::cmp::Ordering::Greater) => T::zero().into(),
                    Some(core::cmp::Ordering::Less) => T::infinity().into(),
                    _ => Self::nan(),
                }
            } else if self.norm_sqr() > T::one() {
                // |q| > 1
                Self::zero()
            } else {
                // Otherwise, return NaN
                Self::nan()
            }
        } else {
            // t is NaN
            debug_assert!(exponent.is_nan());
            Self::nan()
        }
    }
}

#[cfg(any(feature = "std", feature = "libm"))]
impl<T> Quaternion<T>
where
    T: Float,
{
    /// Returns whether all components of the quaternion are finite.
    ///
    /// If a `Quaternion` has an infinite or `NaN` entry, the function returns
    /// `false`, otherwise `true`.
    ///
    /// # Example
    ///
    /// ```
    /// # use num_quaternion::Quaternion;
    /// let q = Quaternion::new(1.0f32, 2.0, 3.0, 4.0);
    /// assert!(q.is_finite());
    /// ```
    pub fn is_finite(&self) -> bool {
        self.w.is_finite()
            && self.x.is_finite()
            && self.y.is_finite()
            && self.z.is_finite()
    }

    /// Returns whether any component of the quaternion is `NaN`.
    ///
    /// # Example
    ///
    /// ```
    /// # use num_quaternion::Quaternion;
    /// let q = Quaternion::new(1.0f32, 2.0, 3.0, 4.0);
    /// assert!(!q.has_nan());
    /// ```
    pub fn has_nan(&self) -> bool {
        self.w.is_nan() || self.x.is_nan() || self.y.is_nan() || self.z.is_nan()
    }

    /// Returns whether all components of a quaternion are `NaN`.
    ///
    /// # Example
    ///
    /// ```
    /// # use num_quaternion::Q64;
    /// let q = Q64::nan();
    /// assert!(q.is_all_nan());
    /// ```
    pub fn is_all_nan(&self) -> bool {
        self.w.is_nan() && self.x.is_nan() && self.y.is_nan() && self.z.is_nan()
    }
}

#[cfg(any(feature = "std", feature = "libm"))]
impl<T> Quaternion<T>
where
    T: Float + FloatConst,
{
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
    /// - If $q = 0$, the result is $-\infty$. (The coefficients of $i$, $j$,
    ///   and $k$ are zero with the original signs copied.)
    /// - If the input has a `NaN` value, then the result is `NaN` in all
    ///   components.
    /// - Otherwise, if $q = w + xi + yj + zk$ where at least one of
    ///   $w, x, y, z$ is infinite, then the real part of the result is
    ///   $+\infty$ and the imaginary part is the imaginary part
    ///   of the logarithm of $f(w) + f(x)i + f(y)j + f(z)k$ where
    ///     - $f(+\infty) := 1$,
    ///     - $f(-\infty) :=-1$, and
    ///     - $f(s) := 0$ for finite values of $s$.
    ///
    /// # Example
    ///
    /// ```
    /// # use num_quaternion::Quaternion;
    /// let q = Quaternion::new(1.0f32, 2.0, 3.0, 4.0);
    /// let ln_q = q.ln();
    /// ```
    pub fn ln(self) -> Self {
        // The square norm of the imaginary part.
        let sqr_norm_im = self.x * self.x + self.y * self.y + self.z * self.z;
        // The square norm of `self`.
        let norm_sqr = self.w * self.w + sqr_norm_im;

        match norm_sqr.classify() {
            FpCategory::Normal => {
                // The normal case: First compute the real part of the result.
                let w =
                    norm_sqr.ln() * T::from(0.5).expect("Conversion failed");

                if sqr_norm_im <= self.w * self.w * T::epsilon() {
                    // We're close to or on the positive real axis
                    if self.w.is_sign_positive() {
                        // This approximation leaves a relative error of less
                        // than a floating point epsilon for the imaginary part
                        let x = self.x / self.w;
                        let y = self.y / self.w;
                        let z = self.z / self.w;
                        Self::new(w, x, y, z)
                    } else if self.x.is_zero()
                        && self.y.is_zero()
                        && self.z.is_zero()
                    {
                        // We're on the negative real axis.
                        Self::new(w, T::PI().copysign(self.x), self.y, self.z)
                    } else if sqr_norm_im.is_normal() {
                        // We're close the the negative real axis. Compute the
                        // norm of the imaginary part.
                        let norm_im = sqr_norm_im.sqrt();

                        // The angle of `self` to the positive real axis is
                        // pi minus the angle from the negative real axis.
                        // The angle from the negative real axis
                        // can be approximated by `norm_im / self.w.abs()`
                        // which is equal to `-norm_im / self.w`. This the
                        // angle from the positive real axis is
                        // `pi + norm_im / self.w`. We obtain the imaginary
                        // part of the result by multiplying this value by
                        // the imaginary part of the input normalized, or
                        // equivalently, by multiplying the imaginary part
                        // of the input by the following factor:
                        let f = T::PI() / norm_im + self.w.recip();

                        Self::new(w, f * self.x, f * self.y, f * self.z)
                    } else {
                        // The imaginary part is so small, that the norm of the
                        // resulting imaginary part differs from `pi` by way
                        // less than half an ulp. Therefore, it's sufficient to
                        // normalize the imaginary part and multiply it by
                        // `pi`.
                        let f = T::min_positive_value().sqrt();
                        let xf = self.x / f;
                        let yf = self.y / f;
                        let zf = self.z / f;
                        let sqr_sum = xf * xf + yf * yf + zf * zf;
                        let im_norm_div_f = sqr_sum.sqrt();
                        let pi_div_f = T::PI() / f;
                        // We could try to reduce the number of divisions by
                        // computing `pi_div_f / im_norm_div_f` and then
                        // multiplying the imaginary part by this value.
                        // However, this reduces numerical accuracy, if the
                        // pi times the norm of the imaginary part is
                        // subnormal. We could also introduce another branch
                        // here, but this would make the code more complex
                        // and extend the worst case latency. Therefore, we
                        // keep the divisions like that.
                        Self::new(
                            w,
                            self.x * pi_div_f / im_norm_div_f,
                            self.y * pi_div_f / im_norm_div_f,
                            self.z * pi_div_f / im_norm_div_f,
                        )
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
                Self::new(T::neg_infinity(), self.x, self.y, self.z)
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
                    let q =
                        Self::new(f(self.w), f(self.x), f(self.y), f(self.z));
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
}

#[cfg(any(feature = "std", feature = "libm"))]
impl<T> Quaternion<T>
where
    T: Float + FloatConst,
{
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
    ///
    /// # Example
    ///
    /// ```
    /// # use num_quaternion::Quaternion;
    /// let q = Quaternion::new(1.0f32, 2.0, 3.0, 4.0);
    /// let sqrt_q = q.sqrt();
    /// ```
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

                    Self::new(
                        wx2 * half,
                        self.x / wx2,
                        self.y / wx2,
                        self.z / wx2,
                    )
                } else {
                    // The first formula for the real part of the result may
                    //  not be robust if the sign of the input real part is
                    // negative.
                    let im_norm_sqr =
                        self.y * self.y + (self.x * self.x + self.z * self.z);
                    if im_norm_sqr >= T::min_positive_value() {
                        // Second formula for the real part of the result,
                        // which is robust for inputs with a negative real
                        // part.
                        let wx2 = (im_norm_sqr * two / (norm - self.w)).sqrt();

                        Self::new(
                            wx2 * half,
                            self.x / wx2,
                            self.y / wx2,
                            self.z / wx2,
                        )
                    } else if self.x.is_zero()
                        && self.y.is_zero()
                        && self.z.is_zero()
                    {
                        // The input is a negative real number.
                        Self::new(
                            zero,
                            (-self.w).sqrt().copysign(self.x),
                            self.y,
                            self.z,
                        )
                    } else {
                        // `im_norm_sqr` is subnormal. Compute the norm of the
                        // imaginary part by scaling up first.
                        let sx = s * self.x;
                        let sy = s * self.y;
                        let sz = s * self.z;
                        let im_norm =
                            (sy * sy + (sx * sx + sz * sz)).sqrt() / s;

                        // Compute the real part according to the second
                        // formula from above.
                        let w = im_norm / (half * (norm - self.w)).sqrt();

                        Self::new(w * half, self.x / w, self.y / w, self.z / w)
                    }
                }
            }
            FpCategory::Zero if self.is_zero() => {
                Self::new(zero, self.x, self.y, self.z)
            }
            FpCategory::Infinite => {
                if self.w == inf
                    || self.x.is_infinite()
                    || self.y.is_infinite()
                    || self.z.is_infinite()
                {
                    let f = |a: T| {
                        if a.is_infinite() {
                            a
                        } else {
                            zero.copysign(a)
                        }
                    };
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

#[cfg(test)]
mod tests {

    use {
        crate::{Quaternion, Q32, Q64, UQ32, UQ64},
        num_traits::{ConstOne, ConstZero, Inv, One, Zero},
    };

    #[cfg(any(feature = "std", feature = "libm"))]
    use num_traits::FloatConst;

    #[test]
    fn test_new() {
        // Test the new function
        let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(q.w, 1.0);
        assert_eq!(q.x, 2.0);
        assert_eq!(q.y, 3.0);
        assert_eq!(q.z, 4.0);
    }

    #[test]
    fn test_zero_constant() {
        // Test the zero function
        let q1 = Q32::ZERO;
        let q2 = Q32::default();
        assert_eq!(q1, q2);
    }

    #[test]
    fn test_const_zero_trait() {
        // Test the ConstZero trait
        let q3 = <Q64 as ConstZero>::ZERO;
        let q4 = Q64::default();
        assert_eq!(q3, q4);
    }

    #[test]
    fn test_zero_trait() {
        // Test the Zero trait's `zero` method
        let q1 = Q32::zero();
        let q2 = Q32::default();
        assert_eq!(q1, q2);
        assert!(q1.is_zero());
    }

    #[test]
    fn test_zero_trait_set_zero() {
        // Test the Zero trait's `set_zero` method
        let mut q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
        q.set_zero();
        assert!(q.is_zero());
    }

    #[test]
    fn test_one_constant() {
        // Test the `ONE` constant
        assert_eq!(Q32::ONE, Q32::new(1.0, 0.0, 0.0, 0.0))
    }

    #[test]
    fn test_i_constant() {
        // Test the `I` constant
        assert_eq!(Q64::I, Q64::new(0.0, 1.0, 0.0, 0.0))
    }

    #[test]
    fn test_j_constant() {
        // Test the `J` constant
        assert_eq!(Q32::J, Q32::new(0.0, 0.0, 1.0, 0.0))
    }

    #[test]
    fn test_k_constant() {
        // Test the `K` constant
        assert_eq!(Q64::K, Q64::new(0.0, 0.0, 0.0, 1.0))
    }

    #[test]
    fn test_const_one_trait() {
        // Test the ConstOne trait
        assert_eq!(<Q32 as ConstOne>::ONE, Q32::new(1.0, 0.0, 0.0, 0.0))
    }

    #[test]
    fn test_one_trait_one() {
        // Test the One trait's `one` method
        assert_eq!(<Q64 as One>::one(), Q64::ONE);
        assert!(Q64::ONE.is_one());
    }

    #[test]
    fn test_one_trait_set_one() {
        // Test the One trait's `set_one` method
        let mut q = Q32::new(2.0, 3.0, 4.0, 5.0);
        q.set_one();
        assert!(q.is_one());
    }

    #[test]
    fn test_one_func() {
        // Test the `one` function
        assert_eq!(Q32::one(), Q32::new(1.0, 0.0, 0.0, 0.0));
    }

    #[test]
    fn test_i_func() {
        // Test the `i` function
        assert_eq!(Q64::i(), Q64::new(0.0, 1.0, 0.0, 0.0));
    }

    #[test]
    fn test_j_func() {
        // Test the `j` function
        assert_eq!(Q64::j(), Q64::new(0.0, 0.0, 1.0, 0.0));
    }

    #[test]
    fn test_k_func() {
        // Test the `k` function
        assert_eq!(Q64::k(), Q64::new(0.0, 0.0, 0.0, 1.0));
    }

    #[test]
    fn test_norm_sqr() {
        // Test the norm_sqr function
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
        // Test the conj function
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
        // Test the inv function
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
        // Test the inv trait for references
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
        // Test the inv trait
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
        // Test the norm function for normal floating point values
        let q = Q64::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(q.norm(), 30.0f64.sqrt());
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_norm_zero_quaternion() {
        // Test the norm function for a zero quaternion
        let q = Q32::new(0.0, 0.0, 0.0, 0.0);
        assert_eq!(q.norm(), 0.0, "Norm of zero quaternion should be 0");
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_norm_subnormal_values() {
        // Test the norm function for subnormal floating point values
        let s = f64::MIN_POSITIVE * 0.25;
        let q = Q64::new(s, s, s, s);
        assert!(
            (q.norm() - 2.0 * s).abs() <= 2.0 * s * f64::EPSILON,
            "Norm of subnormal is computed correctly"
        );
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_norm_large_values() {
        // Test the norm function for large floating point values
        let s = f64::MAX * 0.50;
        let q = Q64::new(s, s, s, s);
        assert!(
            (q.norm() - 2.0 * s).abs() <= 2.0 * s * f64::EPSILON,
            "Norm of large values is computed correctly"
        );
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_norm_infinite_values() {
        // Test the norm function for infinite floating point values
        let inf = f32::INFINITY;
        assert_eq!(Q32::new(inf, 1.0, 1.0, 1.0).norm(), inf);
        assert_eq!(Q32::new(1.0, inf, 1.0, 1.0).norm(), inf);
        assert_eq!(Q32::new(1.0, 1.0, inf, 1.0).norm(), inf);
        assert_eq!(Q32::new(1.0, 1.0, 1.0, inf).norm(), inf);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_norm_nan_values() {
        // Test the norm function for NaN floating point values
        let nan = f32::NAN;
        assert!(Q32::new(nan, 1.0, 1.0, 1.0).norm().is_nan());
        assert!(Q32::new(1.0, nan, 1.0, 1.0).norm().is_nan());
        assert!(Q32::new(1.0, 1.0, nan, 1.0).norm().is_nan());
        assert!(Q32::new(1.0, 1.0, 1.0, nan).norm().is_nan());
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_fast_norm_normal_values() {
        // Test the fast_norm function for normal floating point values
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
        // Test the fast_norm function for a zero quaternion
        assert_eq!(
            Q32::zero().fast_norm(),
            0.0,
            "Fast norm of zero quaternion should be 0"
        );
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_fast_norm_infinite_values() {
        // Test the fast_norm function for infinite floating point values
        let inf = f32::INFINITY;
        assert_eq!(Q32::new(inf, 1.0, 1.0, 1.0).fast_norm(), inf);
        assert_eq!(Q32::new(1.0, inf, 1.0, 1.0).fast_norm(), inf);
        assert_eq!(Q32::new(1.0, 1.0, inf, 1.0).fast_norm(), inf);
        assert_eq!(Q32::new(1.0, 1.0, 1.0, inf).fast_norm(), inf);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_fast_norm_nan_values() {
        // Test the fast_norm function for NaN floating point values
        let nan = f32::NAN;
        assert!(Q32::new(nan, 1.0, 1.0, 1.0).fast_norm().is_nan());
        assert!(Q32::new(1.0, nan, 1.0, 1.0).fast_norm().is_nan());
        assert!(Q32::new(1.0, 1.0, nan, 1.0).fast_norm().is_nan());
        assert!(Q32::new(1.0, 1.0, 1.0, nan).fast_norm().is_nan());
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_fast_norm_for_norm_sqr_underflow() {
        // Test the fast_norm function for underflowing norm_sqr
        let s = f64::MIN_POSITIVE;
        let q = Q64::new(s, s, s, s);
        assert_eq!(q.fast_norm(), 0.0);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_fast_norm_for_norm_sqr_overflow() {
        // Test the fast_norm function for overflowing norm_sqr
        let s = f32::MAX / 16.0;
        let q = Q32::new(s, s, s, s);
        assert_eq!(q.fast_norm(), f32::INFINITY);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_normalize() {
        // Test the normalize function
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
    fn test_normalize_zero() {
        // Test the normalize function for zero
        assert_eq!(Q64::ZERO.normalize(), None);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_normalize_infinity() {
        // Test the normalize function for infinite quaternions
        assert_eq!(Q64::new(f64::INFINITY, 0.0, 0.0, 0.0).normalize(), None);
        assert_eq!(
            Q64::new(0.0, f64::NEG_INFINITY, 0.0, 0.0).normalize(),
            None
        );
        assert_eq!(
            Q64::new(0.0, 0.0, f64::NEG_INFINITY, 0.0).normalize(),
            None
        );
        assert_eq!(Q64::new(0.0, 0.0, 0.0, f64::INFINITY).normalize(), None);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_normalize_nan() {
        // Test the normalize function for NaN quaternions
        assert_eq!(Q64::new(f64::NAN, 0.0, 0.0, 0.0).normalize(), None);
        assert_eq!(Q64::new(0.0, f64::NAN, 0.0, 0.0).normalize(), None);
        assert_eq!(Q64::new(0.0, 0.0, f64::NAN, 0.0).normalize(), None);
        assert_eq!(Q64::new(0.0, 0.0, 0.0, f64::NAN).normalize(), None);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_normalize_infinity_and_nan() {
        // Test the normalize function for quaternions with infinite and NaN
        // values
        assert_eq!(
            Q64::new(f64::INFINITY, f64::NAN, 1.0, 0.0).normalize(),
            None
        );
        assert_eq!(
            Q64::new(1.0, 0.0, f64::INFINITY, f64::NAN).normalize(),
            None
        );
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_normalize_subnormal() {
        // Test the normalize function for subnormal quaternions
        let s = f64::MIN_POSITIVE / 4.0;
        let q = Q64::new(s, s, s, s);
        assert_eq!(
            q.normalize().unwrap().into_inner(),
            Q64::new(0.5, 0.5, 0.5, 0.5)
        );
    }

    #[test]
    fn test_from_underlying_type_val() {
        // Test the From trait for values
        assert_eq!(Q64::from(-5.0), Q64::new(-5.0, 0.0, 0.0, 0.0));
        assert_eq!(Into::<Q32>::into(42.0), Q32::new(42.0, 0.0, 0.0, 0.0));
    }

    #[allow(clippy::needless_borrows_for_generic_args)]
    #[test]
    fn test_from_underlying_type_ref() {
        // Test the From trait for references
        assert_eq!(Q64::from(&-5.0), Q64::new(-5.0, 0.0, 0.0, 0.0));
        assert_eq!(Into::<Q32>::into(&42.0), Q32::new(42.0, 0.0, 0.0, 0.0));
    }

    #[test]
    fn test_from_unit_quaternion() {
        // Test the From trait for unit quaternions
        assert_eq!(Q32::from(UQ32::ONE), Q32::ONE);
        assert_eq!(Q64::from(UQ64::I), Q64::I);
        assert_eq!(Q32::from(UQ32::J), Q32::J);
        assert_eq!(Q64::from(UQ64::K), Q64::K);
    }

    #[allow(clippy::needless_borrows_for_generic_args)]
    #[test]
    fn test_from_unit_quaternion_ref() {
        // Test the From trait for references to unit quaternions
        assert_eq!(<&Q32 as From<&UQ32>>::from(&UQ32::ONE), &Q32::ONE);
        assert_eq!(<&Q64 as From<&UQ64>>::from(&UQ64::I), &Q64::I);
        assert_eq!(<&Q32 as From<&UQ32>>::from(&UQ32::J), &Q32::J);
        assert_eq!(<&Q64 as From<&UQ64>>::from(&UQ64::K), &Q64::K);
    }

    #[test]
    fn test_powu() {
        // Test the power function for unsigned integer exponents
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
        // Test the power function for signed integer exponents
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
                    (q.powi(-e) - expected.inv()).norm_sqr()
                        / expected.norm_sqr()
                        < f32::EPSILON
                );
                expected *= q;
            }
        }
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_exp_zero_quaternion() {
        // Test the exponential function for the zero quaternion
        assert_eq!(Q32::ZERO.exp(), Q32::ONE);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_exp_real_part_only() {
        // Test the exponential function for quaternions with real part only
        assert_eq!(Q32::ONE.exp(), core::f32::consts::E.into())
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_exp_imaginary_part_only() {
        // Test the exponential function for quaternions with imaginary part only
        assert!(
            (Q64::I.exp() - Q64::new(1.0f64.cos(), 1.0f64.sin(), 0.0, 0.0))
                .norm()
                <= f64::EPSILON
        );
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_exp_complex_quaternion() {
        // Test the exponential function for a complex quaternion
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
        // Test the exponential function for quaternions with negative real part
        let q = Q64::new(-1000.0, 0.0, f64::INFINITY, f64::NAN);
        let exp_q = q.exp();
        assert_eq!(exp_q, Q64::zero());
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_exp_nan_input() {
        // Test the exponential function for quaternions with NaN input
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
        // Test the exponential function for quaternions with large imaginary norm
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
        // Test the exponential function for quaternions with infinite real part
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
        // Test the exponential function for quaternions with infinite
        // imaginary part
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
        // Test the exponential function for quaternions with small norm of the imaginary part
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
        // Test the exponential function for quaternions with an angle greater
        // than 90 degrees
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
        // Test the exponential function for quaternions with an angle greater
        // than 180 degrees
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
        // Test the exponential function for a quaternion with a zero base and a
        // positive real part
        assert_eq!(Q64::new(1.0, 0.0, 0.0, 0.0).expf(0.0), Q64::ZERO);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_expf_zero_base_negative_real_quaternion() {
        // Test the exponential function for a quaternion with a zero base and a
        // negative real part
        assert_eq!(
            Q32::new(-1.0, 0.0, 0.0, 0.0).expf(0.0),
            f32::INFINITY.into()
        );
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_expf_infinity_base_positive_real_quaternion() {
        // Test the exponential function for a quaternion with an infinite base
        // and a positive real part
        let inf = f64::INFINITY;
        assert_eq!(Q64::new(1.0, 0.0, 0.0, 0.0).expf(inf), inf.into());
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_expf_infinity_base_negative_real_quaternion() {
        // Test the exponential function for a quaternion with an infinite base
        // and a negative real part
        assert_eq!(
            Q32::new(-1.0, 0.0, 0.0, 0.0).expf(f32::INFINITY),
            0.0f32.into()
        );
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_expf_negative_base() {
        // Test the expf power function for negative exponent and positive base
        let q = Q64::new(1.0, 0.0, 0.0, 0.0).expf(-1.0);
        assert!(q.w.is_nan());
        assert!(q.x.is_nan());
        assert!(q.y.is_nan());
        assert!(q.z.is_nan());
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_expf_nan_base() {
        // Test the expf power function for a NaN base
        let q = Q32::new(1.0, 0.0, 0.0, 0.0).expf(f32::NAN);
        assert!(q.w.is_nan());
        assert!(q.x.is_nan());
        assert!(q.y.is_nan());
        assert!(q.z.is_nan());
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_expf_finite_positive_base() {
        // Test the expf power function for a finite positive base
        let q = Q64::new(1.0, 2.0, 3.0, 4.0);
        let base = 2.0;
        let result = q.expf(base);
        let expected = (q * base.ln()).exp();
        assert!((result - expected).norm() <= expected.norm() * f64::EPSILON);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_expf_nan_quaternion_component() {
        // Test the expf power function for a base with a NaN component
        let q = Q64::new(f64::NAN, 1.0, 1.0, 1.0).expf(3.0);
        assert!(q.w.is_nan());
        assert!(q.x.is_nan());
        assert!(q.y.is_nan());
        assert!(q.z.is_nan());
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_expf_infinity_quaternion_component() {
        // Test the expf power function for a base with an infinite component
        let q = Q32::new(1.0, f32::INFINITY, 1.0, 1.0).expf(2.0);
        assert!(q.w.is_nan());
        assert!(q.x.is_nan());
        assert!(q.y.is_nan());
        assert!(q.z.is_nan());
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_expf_infinite_real_component_with_t_greater_than_1() {
        // Test the expf power function for an infinite real base with a
        // finite real exponent greater than `1`
        let inf = f64::INFINITY;
        assert!(!Q64::new(inf, 0.0, 0.0, 0.0).expf(5.0).is_finite());
        assert!(!Q64::new(inf, 0.0, 0.0, 0.0).expf(5.0).has_nan());
        assert!(!Q64::new(inf, 1.0, 2.0, 3.0).expf(42.0).is_finite());
        assert!(!Q64::new(inf, 1.0, 2.0, 3.0).expf(42.0).has_nan());
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_expf_neg_infinite_real_component_with_t_between_0_and_1() {
        // Test the expf power function for a negative infinite real base with
        // an exponent between 0 and 1
        assert!(!Q32::new(f32::NEG_INFINITY, 0.0, 0.0, 0.0)
            .expf(0.5)
            .is_finite());
        assert!(!Q32::new(f32::NEG_INFINITY, 0.0, 0.0, 0.0)
            .expf(0.5)
            .has_nan());
        assert!(!Q32::new(f32::NEG_INFINITY, 1.0, 2.0, 3.0)
            .expf(0.75)
            .is_finite());
        assert!(!Q32::new(f32::NEG_INFINITY, 1.0, 2.0, 3.0)
            .expf(0.75)
            .has_nan());
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_powf_zero_q_positive_t() {
        // Test the power function for a zero quaternion and a positive
        // exponent
        let q = Quaternion::zero();
        let t = 1.0;
        let result = q.powf(t);
        assert_eq!(result, Quaternion::zero());
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_powf_infinite_q_negative_t() {
        // Test the power function for an infinite quaternion and a negative
        // exponent
        let q = Quaternion::new(f64::INFINITY, 0.0, 0.0, 0.0);
        let t = -1.0;
        let result = q.powf(t);
        assert_eq!(result, Quaternion::zero());
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_powf_infinite_q_positive_t_not_too_large() {
        // Test the power function for an infinite quaternion and a positive
        // exponent
        let q = Quaternion::new(f64::INFINITY, 0.0, 0.0, 0.0);
        let t = 1.0;
        let result = q.powf(t);
        assert_eq!(result, f64::INFINITY.into());
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_powf_infinite_q_large_positive_t() {
        // Test the power function for an infinite quaternion and a very large
        // positive exponent
        let q = Quaternion::new(0.0, f64::INFINITY, 0.0, 0.0);
        let t = f64::MAX;
        let result = q.powf(t);
        assert!(result.w.is_nan());
        assert!(result.x.is_nan());
        assert!(result.y.is_nan());
        assert!(result.z.is_nan());
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_powf_infinite_q_infinite_t() {
        // Test the power function for an infinite quaternion and an infinite
        // exponent
        let q = Quaternion::new(f64::INFINITY, 0.0, 0.0, 0.0);
        let t = f64::INFINITY;
        let result = q.powf(t);
        assert_eq!(result, f64::INFINITY.into());
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_powf_infinite_q_infinite_t_not_positive() {
        // Test the power function for an infinite non-positive quaternion
        // and an infinite positive exponent
        let q = Quaternion::new(f64::INFINITY, 1.0, 0.0, 0.0);
        let t = f64::INFINITY;
        let result = q.powf(t);
        assert!(result.w.is_nan());
        assert!(result.x.is_nan());
        assert!(result.y.is_nan());
        assert!(result.z.is_nan());
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_powf_zero_q_negative_t() {
        // Test that the power function for a zero quaternion and a negative
        // exponent returns an infinite quaternion
        let q = Quaternion::zero();
        let t = -1.0;
        let result = q.powf(t);
        assert_eq!(result, f64::INFINITY.into());
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_powf_nan_q_or_t() {
        // Test the power function for a quaternion or exponent with a NaN
        // component.
        let q = Quaternion::new(0.0, 0.0, f64::NAN, 0.0);
        let t = 1.0;
        let result = q.powf(t);
        assert!(result.w.is_nan());
        assert!(result.x.is_nan());
        assert!(result.y.is_nan());
        assert!(result.z.is_nan());

        let q = Quaternion::one();
        let t = f64::NAN;
        let result = q.powf(t);
        assert!(result.w.is_nan());
        assert!(result.x.is_nan());
        assert!(result.y.is_nan());
        assert!(result.z.is_nan());
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_powf_infinite_q_zero_t() {
        // Test the power function for an infinite quaternion and a zero
        // exponent
        let q = Quaternion::new(f64::INFINITY, 0.0, 0.0, 0.0);
        let t = 0.0;
        let result = q.powf(t);
        assert!(result.w.is_nan());
        assert!(result.x.is_nan());
        assert!(result.y.is_nan());
        assert!(result.z.is_nan());
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_powf_zero_q_zero_t() {
        // Test the power function for a zero quaternion and a zero exponent
        let q = Q32::zero();
        let t = 0.0;
        let result = q.powf(t);
        assert!(result.w.is_nan());
        assert!(result.x.is_nan());
        assert!(result.y.is_nan());
        assert!(result.z.is_nan());
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_powf_non_zero_q_positive_infinite_t() {
        // Test the power function for a non-zero finite quaternion and a
        // positive infinite exponent
        let q = Quaternion::new(2.0, 0.0, 0.0, 0.0);
        let t = f64::INFINITY;
        let result = q.powf(t);
        assert_eq!(result, f64::INFINITY.into());

        let q = Quaternion::new(0.5, 0.0, 0.0, 0.0);
        let result = q.powf(t);
        assert_eq!(result, Quaternion::zero());

        let q = Quaternion::new(0.25, 0.25, 0.25, 0.25);
        let result = q.powf(t);
        assert_eq!(result, Quaternion::zero());

        let q = Quaternion::new(1.0, 0.0, 0.0, 0.0);
        let result = q.powf(t);
        assert!(result.w.is_nan());
        assert!(result.x.is_nan());
        assert!(result.y.is_nan());
        assert!(result.z.is_nan());
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_powf_non_zero_q_negative_infinite_t() {
        // Test the power function for a non-zero finite quaternion and a
        // negative infinite exponent
        let q = Quaternion::new(2.0, 0.0, 0.0, 0.0);
        let t = f64::NEG_INFINITY;
        let result = q.powf(t);
        assert_eq!(result, Quaternion::zero());

        let q = Quaternion::new(1.0, 0.0, 0.0, 1.0);
        let result = q.powf(t);
        assert_eq!(result, Quaternion::zero());

        let q = Quaternion::new(0.5, 0.0, 0.0, 0.0);
        let result = q.powf(t);
        assert_eq!(result, f64::INFINITY.into());

        let q = Quaternion::new(0.25, 0.25, 0.25, 0.25);
        let result = q.powf(t);
        assert!(result.is_all_nan());

        let q = Quaternion::new(1.0, 0.0, 0.0, 0.0);
        let result = q.powf(t);
        assert!(result.w.is_nan());
        assert!(result.x.is_nan());
        assert!(result.y.is_nan());
        assert!(result.z.is_nan());
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_is_finite_for_finite_values() {
        // Test the is_finite method for finite values
        let q = Q64::new(1.0, 2.0, 3.0, 4.0);
        assert!(q.is_finite());
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_is_finite_for_zero() {
        // Test the is_finite method for the zero quaternion
        let q = Q32::zero();
        assert!(q.is_finite());
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_is_finite_for_infinite_values() {
        // Test the is_finite method for infinite values
        let inf = f32::INFINITY;
        assert!(!Q32::new(inf, 1.0, 1.0, 1.0).is_finite());
        assert!(!Q32::new(1.0, inf, 1.0, 1.0).is_finite());
        assert!(!Q32::new(1.0, 1.0, inf, 1.0).is_finite());
        assert!(!Q32::new(1.0, 1.0, 1.0, inf).is_finite());
        assert!(!Q32::new(-inf, 1.0, 1.0, 1.0).is_finite());
        assert!(!Q32::new(1.0, -inf, 1.0, 1.0).is_finite());
        assert!(!Q32::new(1.0, 1.0, -inf, 1.0).is_finite());
        assert!(!Q32::new(1.0, 1.0, 1.0, -inf).is_finite());
        assert!(!Q32::new(inf, -inf, inf, -inf).is_finite());
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_is_finite_for_nan_values() {
        // Test the is_finite method for nan values
        let nan = f64::NAN;
        assert!(!Q64::new(nan, 1.0, 1.0, 1.0).is_finite());
        assert!(!Q64::new(1.0, nan, 1.0, 1.0).is_finite());
        assert!(!Q64::new(1.0, 1.0, nan, 1.0).is_finite());
        assert!(!Q64::new(1.0, 1.0, 1.0, nan).is_finite());
        assert!(!Q64::new(-nan, 1.0, 1.0, 1.0).is_finite());
        assert!(!Q64::new(1.0, -nan, 1.0, 1.0).is_finite());
        assert!(!Q64::new(1.0, 1.0, -nan, 1.0).is_finite());
        assert!(!Q64::new(1.0, 1.0, 1.0, -nan).is_finite());
        assert!(!Q64::new(nan, -nan, nan, -nan).is_finite());
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_has_nan() {
        // Test the has_nan method for nan values
        let nan = f64::NAN;
        let inf = f64::INFINITY;
        assert!(!Q64::new(0.0, 0.0, 0.0, 0.0).has_nan());
        assert!(!Q64::new(1.0, 1.0, 1.0, 1.0).has_nan());
        assert!(!Q64::new(inf, inf, inf, inf).has_nan());
        assert!(Q64::new(nan, 1.0, 1.0, 1.0).has_nan());
        assert!(Q64::new(1.0, nan, 1.0, 1.0).has_nan());
        assert!(Q64::new(1.0, 1.0, nan, 1.0).has_nan());
        assert!(Q64::new(1.0, 1.0, 1.0, nan).has_nan());
        assert!(Q64::new(-nan, 1.0, 1.0, 1.0).has_nan());
        assert!(Q64::new(1.0, -nan, 1.0, 1.0).has_nan());
        assert!(Q64::new(1.0, 1.0, -nan, 1.0).has_nan());
        assert!(Q64::new(1.0, 1.0, 1.0, -nan).has_nan());
        assert!(Q64::new(nan, -nan, nan, -nan).has_nan());
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_is_all_nan() {
        // Test the is_all_nan method for nan values
        let nan = f64::NAN;
        let inf = f64::INFINITY;
        assert!(!Q64::new(0.0, 0.0, 0.0, 0.0).is_all_nan());
        assert!(!Q64::new(1.0, 1.0, 1.0, 1.0).is_all_nan());
        assert!(!Q64::new(inf, inf, inf, inf).is_all_nan());
        assert!(!Q64::new(nan, 1.0, 1.0, 1.0).is_all_nan());
        assert!(!Q64::new(1.0, nan, 1.0, 1.0).is_all_nan());
        assert!(!Q64::new(1.0, 1.0, nan, 1.0).is_all_nan());
        assert!(!Q64::new(1.0, 1.0, 1.0, nan).is_all_nan());
        assert!(!Q64::new(-nan, 1.0, 1.0, 1.0).is_all_nan());
        assert!(!Q64::new(1.0, -nan, 1.0, 1.0).is_all_nan());
        assert!(!Q64::new(1.0, 1.0, -nan, 1.0).is_all_nan());
        assert!(!Q64::new(1.0, 1.0, 1.0, -nan).is_all_nan());
        assert!(Q64::new(nan, nan, nan, nan).is_all_nan());
        assert!(Q64::new(-nan, -nan, -nan, -nan).is_all_nan());
        assert!(Q64::new(nan, -nan, nan, -nan).is_all_nan());
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
        assert!(
            (ln_q.x.hypot(ln_q.y.hypot(ln_q.z)) - 29.0f64.sqrt().atan())
                <= 4.0 * f64::EPSILON
        );
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
        assert!(
            (ln_q - expected).norm() <= core::f32::consts::PI * f32::EPSILON
        );
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
        let expected = Q64::new(f64::NEG_INFINITY, 0.0, 0.0, 0.0);
        assert_eq!(ln_q.w, expected.w);
        assert_eq!(ln_q.x, expected.x);
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
    fn test_ln_negative_real_part_tiny_imaginary_part() {
        // Test a quaternion with a tiny imaginary part

        use core::f32;
        let q = Q32::new(-2.0, 346.0 * f32::EPSILON, 0.0, 0.0);
        let ln_q = q.ln();
        let expected =
            Q32::new(2.0f32.ln(), f32::consts::PI + q.x / q.w, 0.0, 0.0);
        assert!((ln_q - expected).norm() <= 8.0 * f32::EPSILON);

        let q = Q32::new(-3.0, f32::MIN_POSITIVE / 64.0, 0.0, 0.0);
        let ln_q = q.ln();
        let expected = Q32::new(3.0f32.ln(), f32::consts::PI, 0.0, 0.0);
        assert_eq!(ln_q, expected);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_ln_tiny_real_part_and_tiny_imaginary_part() {
        // Test a quaternion with a tiny real and imaginary part

        use core::f32;
        let w = f32::MIN_POSITIVE.sqrt();
        let q = Q32::new(w, 0.0, w / 2.0, 0.0);
        let ln_q = q.ln();
        let expected = Q32::new(
            (1.25 * f32::MIN_POSITIVE).ln() / 2.0,
            0.0,
            0.5f32.atan(),
            0.0,
        );
        assert_eq!(ln_q, expected);

        let w = f32::MIN_POSITIVE;
        let q = Q32::new(w, w, w, w);
        let ln_q = q.ln();
        let expected = Q32::new(
            (2.0 * f32::MIN_POSITIVE).ln(),
            f32::consts::PI / 27.0f32.sqrt(),
            f32::consts::PI / 27.0f32.sqrt(),
            f32::consts::PI / 27.0f32.sqrt(),
        );
        assert!(
            (ln_q - expected).norm() <= expected.norm() * 2.0 * f32::EPSILON
        );
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_ln_very_large_inputs() {
        // Test the logarithm of very large inputs

        use core::f64;
        let q = Q64::new(f64::MAX, 0.0, 0.0, f64::MAX);
        let ln_q = q.ln();
        let expected = Q64::new(
            f64::MAX.ln() + 2.0f64.ln() / 2.0,
            0.0,
            0.0,
            f64::consts::PI / 4.0,
        );
        assert!(
            (ln_q - expected).norm() <= expected.norm() * 2.0 * f64::EPSILON
        );
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_sqrt_normal() {
        // Test the square root of a normal quaternion
        let q = Q64::new(1.0, 2.0, 3.0, 4.0);
        assert!(((q * q).sqrt() - q).norm() <= q.norm() * f64::EPSILON);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_sqrt_zero() {
        // Test the square root of the zero quaternion
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
        // Test the square root of a negative real quaternion
        let q = Q64::new(-4.0, -0.0, 0.0, 0.0);
        let sqrt_q = q.sqrt();
        let expected = Q64::new(0.0, -2.0, 0.0, 0.0);
        assert_eq!(sqrt_q, expected);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_sqrt_nan() {
        // Test the square root of a quaternion with NaN
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
        // Test the square root of a quaternion with infinite components
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
        // Test the square root of a quaternion with a negative infinite real
        // part
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
        // Test the commutativity of the square root with the conjugate
        let q = Q32::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(q.conj().sqrt(), q.sqrt().conj());
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_sqrt_subnormal_values() {
        // Test the square root of subnormal values
        let subnormal = f64::MIN_POSITIVE / 2.0;
        let q = Quaternion::new(subnormal, subnormal, subnormal, subnormal);
        let sqrt_q = q.sqrt();
        let norm_sqr = sqrt_q.norm_sqr();
        assert!(
            (norm_sqr - f64::MIN_POSITIVE).abs()
                <= 4.0 * subnormal * f64::EPSILON
        );
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_sqrt_mixed_infinities() {
        // Test the square root of a quaternion with all infinite components
        // with different signs
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
        // Test the square root of a positive real quaternion
        let q = Q64::new(4.0, 0.0, 0.0, 0.0);
        let sqrt_q = q.sqrt();
        let expected = Q64::new(2.0, 0.0, 0.0, 0.0);
        assert_eq!(sqrt_q, expected);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_sqrt_purely_imaginary() {
        // Test the square root of a purely imaginary quaternion
        let q = Q32::new(0.0, 3.0, 4.0, 0.0);
        let sqrt_q = q.sqrt();
        assert!(sqrt_q.w > 0.0);
        assert!((sqrt_q * sqrt_q - q).norm() <= 2.0 * q.norm() * f32::EPSILON);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_sqrt_negative_imaginary() {
        // Test the square root of a pure quaternion with negative imaginary
        // parts
        let q = Q64::new(0.0, -3.0, -4.0, 0.0);
        let sqrt_q = q.sqrt();
        assert!(sqrt_q.w > 0.0);
        assert!((sqrt_q * sqrt_q - q).norm() <= 16.0 * q.norm() * f64::EPSILON);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_sqrt_negative_real_part_subnormal_imaginary_part() {
        // Test the square root of a quaternion with a negative real part and
        // subnormal imaginary parts
        let q = Q32::new(-1.0, f32::MIN_POSITIVE / 64.0, 0.0, 0.0);
        let sqrt_q = q.sqrt();
        let expected = Q32::new(q.x / 2.0, 1.0, 0.0, 0.0);
        assert_eq!(sqrt_q, expected);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_sqrt_for_overflowing_norm_sqr_of_input() {
        // Test the square root of a quaternion with an overflowing norm_sqr
        let n = f64::MAX / 2.0;
        let q = Q64::new(-n, n, n, n);
        let sqrt_q = q.sqrt();
        let sqrt_n = f64::MAX.sqrt() / 2.0;
        let expected = Q64::new(sqrt_n, sqrt_n, sqrt_n, sqrt_n);
        assert!((sqrt_q - expected).norm() <= expected.norm() * f64::EPSILON);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_serde_quaternion() {
        // Create a sample quaternion
        let q = Q32::new(1.0, 2.0, 3.0, 4.0);

        // Serialize the quaternion to a JSON string
        let serialized =
            serde_json::to_string(&q).expect("Failed to serialize quaternion");

        // Deserialize the JSON string back into a quaternion
        let deserialized: Quaternion<f32> = serde_json::from_str(&serialized)
            .expect("Failed to deserialize quaternion");

        // Assert that the deserialized quaternion is equal to the original
        assert_eq!(q, deserialized);
    }
}
