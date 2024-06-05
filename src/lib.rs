//! Quaternions for Rust.

#![deny(missing_docs)]
#![no_std]

#[cfg(feature = "std")]
extern crate std;

use core::{
    borrow::Borrow,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};
use num_traits::{ConstOne, ConstZero, Inv, Num, One, Zero};

#[cfg(any(feature = "std", feature = "libm"))]
use {core::num, num_traits::float::Float};

/// Quaternion type.
///
/// We follow the naming conventions from
/// [Wikipedia](https://en.wikipedia.org/wiki/Quaternion) for quaternions.
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
    /// See also [`Quaternion::one()`].
    pub const ONE: Self = Self::new(T::ONE, T::ZERO, T::ZERO, T::ZERO);

    /// A constant `Quaternion` of value $i$.
    ///
    /// See also [`Quaternion::i()`].
    pub const I: Self = Self::new(T::ZERO, T::ONE, T::ZERO, T::ZERO);

    /// A constant `Quaternion` of value $j$.
    ///
    /// See also [`Quaternion::j()`].
    pub const J: Self = Self::new(T::ZERO, T::ZERO, T::ONE, T::ZERO);

    /// A constant `Quaternion` of value $k$.
    ///
    /// See also [`Quaternion::k()`].
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
    /// This is guaranteed to be more efficient than [`norm()`](Quaternion::norm()).
    /// Furthermore, `T` only needs to support addition and multiplication
    /// and therefore, this function works for more types than
    /// [`norm()`](Quaternion::norm()).
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
    /// errors. The rounding error is at most 1.5
    /// [ulps](https://en.wikipedia.org/wiki/Unit_in_the_last_place).
    #[inline]
    pub fn norm(self) -> T {
        // TODO: Optimize this function.
        self.w.hypot(self.x).hypot(self.y.hypot(self.z))
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
            core::num::FpCategory::Normal | core::num::FpCategory::Subnormal => {
                Some(UnitQuaternion(self / norm))
            }
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

/// A quaternion with norm $1$.
///
/// Unit quaternions form a non-commutative group that can be conveniently used
/// for rotating 3D vectors. A 3D vector can be interpreted as a pure
/// quaternion (a quaternion with real part zero). Such a pure quaternion
/// $v$ can be rotated in 3D space by computing $q^{-1}\cdot v\cdot q$ for a
/// unit quaternion $q$. The resulting product is again a pure quaternion which
/// is $v$ rotated around the axis given by the imaginary part of $q$. The
/// method [`rotate_vector()`](UnitQuaternion::rotate_vector) performs this
/// operation efficiently. The angle of rotation is double the angle between
/// $1$ and $q$ interpreted as 4D vectors.
///
/// Multiplying two unit quaternions yields again unit quaternion in theory.
/// However, due to limited machine precision, rounding errors accumulate
/// in practice and the resulting norm may deviate from $1$ more and more.
/// Thus, when you multiply a unit quaternions many times, then you need to
/// adjust the norm. This can be done by calling the function
/// [`adjust_norm()`](UnitQuaternion::adjust_norm).
///
/// See also [`Quaternion`].
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

    /// Converts the UnitQuaternion to roll, pitch, and yaw angles.
    pub fn to_euler_angles(&self) -> EulerAngles<T> {
        let &Self(Quaternion { w, x, y, z }) = self;

        let two = T::from(2.0).unwrap();
        let one = T::from(1.0).unwrap();
        let epsilon = T::epsilon();

        // Compute the sin of the pitch angle
        let sin_pitch = two * (w * y - z * x);

        // Check for gimbal lock, which occurs when sin_pitch is close to 1 or -1
        if sin_pitch.abs() >= one - epsilon {
            // Gimbal lock case
            let half_pi = T::from(std::f64::consts::FRAC_PI_2).unwrap();
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

    /// Returns a quaternion from a vector which is parallel to the rotation
    /// axis and whose norm is the rotation angle.
    pub fn from_rotation_vector(v: &[T; 3]) -> Self {
        let sqr_norm = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
        let two = T::one() + T::one();
        match sqr_norm.classify() {
            num::FpCategory::Normal => {
                // TODO: Optimize this further for norms that are not above pi.
                let norm = sqr_norm.sqrt();
                let (sine, cosine) = (norm / two).sin_cos();
                let f = sine / norm;
                Self(Quaternion::new(cosine, v[0] * f, v[1] * f, v[2] * f))
            }
            num::FpCategory::Zero | num::FpCategory::Subnormal => Self(Quaternion::new(
                // This formula could be used for norm <= epsilon generally,
                // where epsilon is the floating point epsilon.
                T::one(),
                v[0] / two,
                v[1] / two,
                v[2] / two,
            )),
            num::FpCategory::Nan | num::FpCategory::Infinite => Self(Quaternion::nan()),
        }
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
    /// See also [`UnitQuaternion::one()`], [`Quaternion::ONE`].
    pub const ONE: Self = Self(Quaternion::ONE);

    /// A constant `UnitQuaternion` of value $i$.
    ///
    /// See also [`UnitQuaternion::i()`], [`Quaternion::I`].
    pub const I: Self = Self(Quaternion::I);

    /// A constant `UnitQuaternion` of value $j$.
    ///
    /// See also [`UnitQuaternion::j()`], [`Quaternion::J`].
    pub const J: Self = Self(Quaternion::J);

    /// A constant `UnitQuaternion` of value $k$.
    ///
    /// See also [`UnitQuaternion::k()`], [`Quaternion::K`].
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
    /// See also [`UnitQuaternion::I`], [`Quaternion::i()`].
    #[inline]
    pub fn i() -> Self {
        Self(Quaternion::i())
    }

    /// Returns the imaginary unit $j$.
    ///
    /// See also [`UnitQuaternion::J`], [`Quaternion::j()`].
    #[inline]
    pub fn j() -> Self {
        Self(Quaternion::j())
    }

    /// Returns the imaginary unit $k$.
    ///
    /// See also [`UnitQuaternion::K`], [`Quaternion::k()`].
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

impl<T> Neg for UnitQuaternion<T>
where
    T: Neg<Output = T>,
{
    type Output = UnitQuaternion<T>;

    fn neg(self) -> Self::Output {
        Self(-self.0)
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
    use num_traits::Inv;
    use num_traits::One;
    use num_traits::Zero;

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
    fn test_norm() {
        assert_eq!(Q64::ONE.norm(), 1.0);
        assert_eq!(Q32::I.norm(), 1.0);
        assert_eq!(Q64::J.norm(), 1.0);
        assert_eq!(Q32::K.norm(), 1.0);
        assert_eq!(Q64::new(9.0, 12.0, -12.0, -16.0).norm(), 25.0);
        assert_eq!(Q32::new(-1.0, -1.0, 1.0, -1.0).norm(), 2.0);
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

    /// Computes the hash value of `val` using the default hasher.
    #[cfg(feature = "std")]
    fn compute_hash(val: impl Hash) -> u64 {
        let mut hasher = DefaultHasher::new();
        val.hash(&mut hasher);
        hasher.finish()
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
            Q32::new(1.0, 0.0, 0.0, 0.0),
            Q32::new(0.0, 1.0, 0.0, 0.0),
            Q32::new(0.0, 0.0, 1.0, 0.0),
            Q32::new(0.0, 0.0, 0.0, 1.0),
            Q32::new(1.0, 1.0, 1.0, 1.0),
            Q32::new(1.0, -2.0, 3.0, -4.0),
            Q32::new(4.0, 3.0, 2.0, 1.0),
        ];
        for q in test_data.into_iter().map(|q| q.normalize().unwrap()) {
            let EulerAngles { roll, pitch, yaw } = q.to_euler_angles();
            let p = UQ32::from_euler_angles(roll, pitch, yaw);
            assert!((p - q).norm() < core::f32::EPSILON);
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
        assert!((q.adjust_norm().into_quaternion().norm() - 1.0).abs() <= 2.0 * core::f32::EPSILON);
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
}
