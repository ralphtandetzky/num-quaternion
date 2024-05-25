//! Quaternions for Rust.

#![deny(missing_docs)]
#![no_std]

#[cfg(feature = "std")]
extern crate std;

use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use num_traits::{ConstOne, ConstZero, Inv, Num, One, Zero};

#[cfg(any(feature = "std", feature = "libm"))]
use num_traits::float::Float;

/// Quaternion type.
///
/// We follow the naming conventions from
/// [Wikipedia](https://en.wikipedia.org/wiki/Quaternion) for quaternions.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct Quaternion<T> {
    /// Real part of the quaternion.
    pub a: T,
    /// The coefficient of $i$.
    pub b: T,
    /// The coefficient of $j$.
    pub c: T,
    /// The coefficient of $k$.
    pub d: T,
}

/// Alias for a [`Quaternion<f32>`].
pub type Q32 = Quaternion<f32>;
/// Alias for a [`Quaternion<f64>`].
pub type Q64 = Quaternion<f64>;

impl<T> Quaternion<T> {
    /// Create a new quaternion $a + bi + cj + dk$.
    pub const fn new(a: T, b: T, c: T, d: T) -> Self {
        Self { a, b, c, d }
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
    fn zero() -> Self {
        Self::new(Zero::zero(), Zero::zero(), Zero::zero(), Zero::zero())
    }

    fn is_zero(&self) -> bool {
        self.a.is_zero() && self.b.is_zero() && self.c.is_zero() && self.d.is_zero()
    }

    fn set_zero(&mut self) {
        self.a.set_zero();
        self.b.set_zero();
        self.c.set_zero();
        self.d.set_zero();
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
    fn one() -> Self {
        Self::new(One::one(), Zero::zero(), Zero::zero(), Zero::zero())
    }

    fn is_one(&self) -> bool {
        self.a.is_one() && self.b.is_zero() && self.c.is_zero() && self.d.is_zero()
    }

    fn set_one(&mut self) {
        self.a.set_one();
        self.b.set_zero();
        self.c.set_zero();
        self.d.set_zero();
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

impl<T> Quaternion<T>
where
    T: Clone + Mul<T, Output = T> + Add<T, Output = T>,
{
    /// Returns the square of the norm.
    ///
    /// The result is $a^2 + b^2 + c^2 + d^2$ with some rounding errors.
    /// The rounding error is at most 2
    /// [ulps](https://en.wikipedia.org/wiki/Unit_in_the_last_place).
    ///
    /// This is guaranteed to be more efficient than [`norm()`](Quaternion::norm()).
    /// Furthermore, `T` only needs to support addition and multiplication
    /// and therefore, this function works for more types than
    /// [`norm()`](Quaternion::norm()).
    #[inline]
    pub fn norm_sqr(&self) -> T {
        (self.a.clone() * self.a.clone() + self.c.clone() * self.c.clone())
            + (self.b.clone() * self.b.clone() + self.d.clone() * self.d.clone())
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
            self.a.clone(),
            -self.b.clone(),
            -self.c.clone(),
            -self.d.clone(),
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
            self.a.clone() / norm_sqr.clone(),
            -self.b.clone() / norm_sqr.clone(),
            -self.c.clone() / norm_sqr.clone(),
            -self.d.clone() / norm_sqr,
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
    /// The result is $\sqrt{a^2+b^2+c^2+d^2}$ with some possible rounding
    /// errors. The rounding error is at most 1.5
    /// [ulps](https://en.wikipedia.org/wiki/Unit_in_the_last_place).
    #[inline]
    pub fn norm(self) -> T {
        // TODO: Optimize this function.
        self.a.hypot(self.b).hypot(self.c.hypot(self.d))
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

impl<T> Add<Quaternion<T>> for Quaternion<T>
where
    T: Add<T, Output = T>,
{
    type Output = Quaternion<T>;

    fn add(self, rhs: Quaternion<T>) -> Self::Output {
        Self::new(
            self.a + rhs.a,
            self.b + rhs.b,
            self.c + rhs.c,
            self.d + rhs.d,
        )
    }
}

impl<T> Add<T> for Quaternion<T>
where
    T: Add<T, Output = T>,
{
    type Output = Quaternion<T>;

    fn add(self, rhs: T) -> Self::Output {
        Self::new(self.a + rhs, self.b, self.c, self.d)
    }
}

impl<T> Sub<Quaternion<T>> for Quaternion<T>
where
    T: Sub<T, Output = T>,
{
    type Output = Quaternion<T>;

    fn sub(self, rhs: Quaternion<T>) -> Self::Output {
        Self::new(
            self.a - rhs.a,
            self.b - rhs.b,
            self.c - rhs.c,
            self.d - rhs.d,
        )
    }
}

impl<T> Sub<T> for Quaternion<T>
where
    T: Sub<T, Output = T>,
{
    type Output = Quaternion<T>;

    fn sub(self, rhs: T) -> Self::Output {
        Self::new(self.a - rhs, self.b, self.c, self.d)
    }
}

impl<T> Mul<Quaternion<T>> for Quaternion<T>
where
    T: Add<T, Output = T> + Sub<T, Output = T> + Mul<T, Output = T> + Clone,
{
    type Output = Quaternion<T>;

    fn mul(self, rhs: Quaternion<T>) -> Self::Output {
        let a = self.a.clone() * rhs.a.clone()
            - self.b.clone() * rhs.b.clone()
            - self.c.clone() * rhs.c.clone()
            - self.d.clone() * rhs.d.clone();
        let b = self.a.clone() * rhs.b.clone()
            + self.b.clone() * rhs.a.clone()
            + self.c.clone() * rhs.d.clone()
            - self.d.clone() * rhs.c.clone();
        let c = self.a.clone() * rhs.c.clone() - self.b.clone() * rhs.d.clone()
            + self.c.clone() * rhs.a.clone()
            + self.d.clone() * rhs.b.clone();
        let d = self.a * rhs.d + self.b * rhs.c - self.c * rhs.b + self.d * rhs.a;
        Self::new(a, b, c, d)
    }
}

impl<T> Mul<T> for Quaternion<T>
where
    T: Mul<T, Output = T> + Clone,
{
    type Output = Quaternion<T>;

    fn mul(self, rhs: T) -> Self::Output {
        Self::new(
            self.a * rhs.clone(),
            self.b * rhs.clone(),
            self.c * rhs.clone(),
            self.d * rhs,
        )
    }
}

impl<T> Div<Quaternion<T>> for Quaternion<T>
where
    T: Num + Clone + Neg<Output = T>,
{
    type Output = Quaternion<T>;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Quaternion<T>) -> Self::Output {
        self * rhs.inv()
    }
}

impl<T> Div<T> for Quaternion<T>
where
    T: Div<T, Output = T> + Clone,
{
    type Output = Quaternion<T>;

    fn div(self, rhs: T) -> Self::Output {
        Self::new(
            self.a / rhs.clone(),
            self.b / rhs.clone(),
            self.c / rhs.clone(),
            self.d / rhs,
        )
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

            fn add(self, mut rhs: Quaternion<$real>) -> Self::Output {
                rhs.a += self;
                rhs
            }
        }

        impl Sub<Quaternion<$real>> for $real {
            type Output = Quaternion<$real>;

            fn sub(self, rhs: Quaternion<$real>) -> Self::Output {
                let zero = <$real>::zero();
                Self::Output::new(self - rhs.a, zero - rhs.b, zero - rhs.c, zero - rhs.d)
            }
        }

        impl Mul<Quaternion<$real>> for $real {
            type Output = Quaternion<$real>;

            fn mul(self, rhs: Quaternion<$real>) -> Self::Output {
                Self::Output::new(self * rhs.a, self * rhs.b, self * rhs.c, self * rhs.d)
            }
        }
    )*
    };
}

impl_ops_lhs_real!(usize, u8, u16, u32, u64, u128, isize, i8, i16, i32, i64, i128, f32, f64);

impl Div<Q32> for f32 {
    type Output = Q32;

    fn div(mut self, rhs: Q32) -> Self::Output {
        self /= rhs.norm_sqr();
        Self::Output::new(self * rhs.a, self * -rhs.b, self * -rhs.c, self * -rhs.d)
    }
}

impl Div<Q64> for f64 {
    type Output = Q64;

    fn div(mut self, rhs: Q64) -> Self::Output {
        self /= rhs.norm_sqr();
        Self::Output::new(self * rhs.a, self * -rhs.b, self * -rhs.c, self * -rhs.d)
    }
}

impl<T> Neg for Quaternion<T>
where
    T: Neg<Output = T>,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::new(-self.a, -self.b, -self.c, -self.d)
    }
}

impl<T> Quaternion<T>
where
    T: Num + Clone,
{
    /// Raises `self` to an unsigned integer power `n`, i. e. $q^n$.
    #[inline]
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

#[cfg(test)]
mod tests {
    use num_traits::ConstOne;
    use num_traits::ConstZero;
    use num_traits::Inv;
    use num_traits::One;
    use num_traits::Zero;

    use crate::Quaternion;
    use crate::Q32;
    use crate::Q64;

    #[test]
    fn test_new() {
        let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(q.a, 1.0);
        assert_eq!(q.b, 2.0);
        assert_eq!(q.c, 3.0);
        assert_eq!(q.d, 4.0);
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
}
