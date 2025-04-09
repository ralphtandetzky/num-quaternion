use {
    crate::{Quaternion, UnitQuaternion, Q32, Q64, UQ32, UQ64},
    core::ops::{
        Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign,
    },
    num_traits::{Inv, Num, Zero},
};

#[cfg(feature = "unstable")]
use crate::PureQuaternion;

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
    fn add(self, rhs: UnitQuaternion<T>) -> Self {
        self + rhs.into_inner()
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
    fn sub(self, rhs: UnitQuaternion<T>) -> Self {
        self - rhs.into_inner()
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
        let d =
            self.w * rhs.z + self.x * rhs.y - self.y * rhs.x + self.z * rhs.w;
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
        self * rhs.into_inner()
    }
}

impl<T> Div<Quaternion<T>> for Quaternion<T>
where
    T: Num + Clone + Neg<Output = T>,
{
    type Output = Quaternion<T>;

    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline]
    fn div(self, rhs: Quaternion<T>) -> Self {
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
        self * rhs.into_inner().conj()
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
                Quaternion::new(self - rhs.w, zero - rhs.x, zero - rhs.y, zero - rhs.z)
            }
        }

        impl Mul<Quaternion<$real>> for $real {
            type Output = Quaternion<$real>;

            #[inline]
            fn mul(self, rhs: Quaternion<$real>) -> Self::Output {
                Quaternion::new(self * rhs.w, self * rhs.x, self * rhs.y, self * rhs.z)
            }
        }
    )*
    };
}

impl_ops_lhs_real!(
    usize, u8, u16, u32, u64, u128, isize, i8, i16, i32, i64, i128, f32, f64
);

impl Div<Q32> for f32 {
    type Output = Q32;

    #[inline]
    fn div(mut self, rhs: Q32) -> Self::Output {
        self /= rhs.norm_sqr();
        Quaternion::new(
            self * rhs.w,
            self * -rhs.x,
            self * -rhs.y,
            self * -rhs.z,
        )
    }
}

impl Div<Q64> for f64 {
    type Output = Q64;

    #[inline]
    fn div(mut self, rhs: Q64) -> Self::Output {
        self /= rhs.norm_sqr();
        Quaternion::new(
            self * rhs.w,
            self * -rhs.x,
            self * -rhs.y,
            self * -rhs.z,
        )
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

impl<T> Add<UnitQuaternion<T>> for UnitQuaternion<T>
where
    Quaternion<T>: Add<Output = Quaternion<T>>,
{
    type Output = Quaternion<T>;

    #[inline]
    fn add(self, rhs: UnitQuaternion<T>) -> Self::Output {
        self.into_inner() + rhs.into_inner()
    }
}

impl<T> Add<Quaternion<T>> for UnitQuaternion<T>
where
    Quaternion<T>: Add<Output = Quaternion<T>>,
{
    type Output = Quaternion<T>;

    #[inline]
    fn add(self, rhs: Quaternion<T>) -> Self::Output {
        self.into_inner() + rhs
    }
}

impl<T> Add<T> for UnitQuaternion<T>
where
    Quaternion<T>: Add<T, Output = Quaternion<T>>,
{
    type Output = Quaternion<T>;

    #[inline]
    fn add(self, rhs: T) -> Self::Output {
        self.into_inner() + rhs
    }
}

impl Add<UQ32> for f32 {
    type Output = Q32;

    #[inline]
    fn add(self, rhs: UQ32) -> Self::Output {
        self + rhs.into_inner()
    }
}

impl Add<UQ64> for f64 {
    type Output = Q64;

    #[inline]
    fn add(self, rhs: UQ64) -> Self::Output {
        self + rhs.into_inner()
    }
}

impl<T> Sub<UnitQuaternion<T>> for UnitQuaternion<T>
where
    Quaternion<T>: Sub<Output = Quaternion<T>>,
{
    type Output = Quaternion<T>;

    #[inline]
    fn sub(self, rhs: UnitQuaternion<T>) -> Self::Output {
        self.into_inner() - rhs.into_inner()
    }
}

impl<T> Sub<Quaternion<T>> for UnitQuaternion<T>
where
    Quaternion<T>: Sub<Output = Quaternion<T>>,
{
    type Output = Quaternion<T>;

    #[inline]
    fn sub(self, rhs: Quaternion<T>) -> Self::Output {
        self.into_inner() - rhs
    }
}

impl<T> Sub<T> for UnitQuaternion<T>
where
    Quaternion<T>: Sub<T, Output = Quaternion<T>>,
{
    type Output = Quaternion<T>;

    #[inline]
    fn sub(self, rhs: T) -> Self::Output {
        self.into_inner() - rhs
    }
}

impl Sub<UQ32> for f32 {
    type Output = Q32;

    #[inline]
    fn sub(self, rhs: UQ32) -> Self::Output {
        self - rhs.into_inner()
    }
}

impl Sub<UQ64> for f64 {
    type Output = Q64;

    #[inline]
    fn sub(self, rhs: UQ64) -> Self::Output {
        self - rhs.into_inner()
    }
}

impl<T> Mul<Quaternion<T>> for UnitQuaternion<T>
where
    Quaternion<T>: Mul<Output = Quaternion<T>>,
{
    type Output = Quaternion<T>;

    #[inline]
    fn mul(self, rhs: Quaternion<T>) -> Self::Output {
        self.into_inner() * rhs
    }
}

impl<T> Mul<T> for UnitQuaternion<T>
where
    Quaternion<T>: Mul<T, Output = Quaternion<T>>,
{
    type Output = Quaternion<T>;

    #[inline]
    fn mul(self, rhs: T) -> Self::Output {
        self.into_inner() * rhs
    }
}

impl Mul<UQ32> for f32 {
    type Output = Q32;

    #[inline]
    fn mul(self, rhs: UQ32) -> Self::Output {
        self * rhs.into_inner()
    }
}

impl Mul<UQ64> for f64 {
    type Output = Q64;

    #[inline]
    fn mul(self, rhs: UQ64) -> Self::Output {
        self * rhs.into_inner()
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
        self.into_inner() / rhs
    }
}

impl<T> Div<T> for UnitQuaternion<T>
where
    Quaternion<T>: Div<T, Output = Quaternion<T>>,
{
    type Output = Quaternion<T>;

    #[inline]
    fn div(self, rhs: T) -> Self::Output {
        self.into_inner() / rhs
    }
}

impl Div<UQ32> for f32 {
    type Output = Q32;

    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline]
    fn div(self, rhs: UQ32) -> Self::Output {
        self * rhs.inv().into_inner()
    }
}

impl Div<UQ64> for f64 {
    type Output = Q64;

    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline]
    fn div(self, rhs: UQ64) -> Self::Output {
        self * rhs.inv().into_inner()
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

#[cfg(feature = "unstable")]
impl<T> Add<PureQuaternion<T>> for PureQuaternion<T>
where
    T: Add<T, Output = T>,
{
    type Output = PureQuaternion<T>;

    #[inline]
    fn add(self, rhs: PureQuaternion<T>) -> Self::Output {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

#[cfg(feature = "unstable")]
impl<T> Sub<PureQuaternion<T>> for PureQuaternion<T>
where
    T: Sub<T, Output = T>,
{
    type Output = PureQuaternion<T>;

    #[inline]
    fn sub(self, rhs: PureQuaternion<T>) -> Self::Output {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

#[cfg(feature = "unstable")]
impl<T> Mul<PureQuaternion<T>> for PureQuaternion<T>
where
    T: Neg<Output = T>
        + Add<T, Output = T>
        + Sub<T, Output = T>
        + Mul<T, Output = T>
        + Clone,
{
    type Output = Quaternion<T>;

    #[inline]
    fn mul(self, rhs: PureQuaternion<T>) -> Self::Output {
        Quaternion::new(
            -(self.x.clone() * rhs.x.clone()
                + self.y.clone() * rhs.y.clone()
                + self.z.clone() * rhs.z.clone()),
            self.y.clone() * rhs.z.clone() - self.z.clone() * rhs.y.clone(),
            self.z.clone() * rhs.x.clone() - self.x.clone() * rhs.z.clone(),
            self.x.clone() * rhs.y.clone() - self.y.clone() * rhs.x.clone(),
        )
    }
}

#[cfg(feature = "unstable")]
impl<T> Mul<T> for PureQuaternion<T>
where
    T: Mul<T, Output = T> + Clone,
{
    type Output = PureQuaternion<T>;

    #[inline]
    fn mul(self, rhs: T) -> Self::Output {
        Self::new(self.x * rhs.clone(), self.y * rhs.clone(), self.z * rhs)
    }
}

#[cfg(feature = "unstable")]
impl<T> Div<T> for PureQuaternion<T>
where
    T: Div<T, Output = T> + Clone,
{
    type Output = PureQuaternion<T>;

    #[inline]
    fn div(self, rhs: T) -> Self::Output {
        Self::new(self.x / rhs.clone(), self.y / rhs.clone(), self.z / rhs)
    }
}

#[cfg(test)]
mod tests {

    use crate::{Quaternion, Q32, Q64, UQ32, UQ64};

    #[test]
    fn test_add_quaternion() {
        // Test the addition of quaternions
        assert_eq!(Q32::ONE + Q32::J, Q32::new(1.0, 0.0, 1.0, 0.0));
        assert_eq!(
            Q64::new(1.0, 2.0, 3.0, 4.0) + Q64::new(1.0, 3.0, 10.0, -5.0),
            Q64::new(2.0, 5.0, 13.0, -1.0)
        );
    }

    #[test]
    fn test_add_real() {
        // Test the addition of a real number to a quaternion
        assert_eq!(Q32::I + 1.0, Q32::new(1.0, 1.0, 0.0, 0.0));
        assert_eq!(
            Q32::new(1.0, 2.0, 3.0, 4.0) + 42.0,
            Q32::new(43.0, 2.0, 3.0, 4.0)
        );
    }

    #[test]
    fn test_sub_quaternion() {
        // Test the subtraction of quaternions
        assert_eq!(Q32::ONE - Q32::J, Q32::new(1.0, 0.0, -1.0, 0.0));
        assert_eq!(
            Q64::new(1.0, 2.0, 3.0, 4.0) - Q64::new(1.0, 3.0, 10.0, -5.0),
            Q64::new(0.0, -1.0, -7.0, 9.0)
        );
    }

    #[test]
    fn test_sub_real() {
        // Test the subtraction of a real number from a quaternion
        assert_eq!(Q32::I - 1.0, Q32::new(-1.0, 1.0, 0.0, 0.0));
        assert_eq!(
            Q32::new(1.0, 2.0, 3.0, 4.0) - 42.0,
            Q32::new(-41.0, 2.0, 3.0, 4.0)
        );
    }

    #[test]
    fn test_mul_quaternion() {
        // Test the multiplication of quaternions
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
        // Test the multiplication of a quaternion by a real number
        assert_eq!(Q32::I * 1.0, Q32::I);
        assert_eq!(
            Q32::new(1.0, 2.0, 3.0, 4.0) * 42.0,
            Q32::new(42.0, 84.0, 126.0, 168.0)
        );
    }

    #[test]
    fn test_mul_quaternion_by_unit_quaternion() {
        // Test the multiplication of a quaternion by a unit quaternion
        assert_eq!(Q32::I * UQ32::J, Q32::K);
        assert_eq!(Q64::J * UQ64::K, Q64::I);
        assert_eq!(
            Q32::new(1.0, 2.0, 3.0, 4.0) * UQ32::K,
            Q32::new(-4.0, 3.0, -2.0, 1.0)
        );
    }

    #[test]
    fn test_div_quaternion() {
        // Test the division of quaternions
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
        // Test the division of a quaternion by a real number
        assert_eq!(Q32::I * 1.0, Q32::I);
        assert_eq!(
            Q32::new(1.0, 2.0, 3.0, 4.0) / 4.0,
            Q32::new(0.25, 0.5, 0.75, 1.0)
        );
    }

    #[test]
    fn test_div_quaternion_by_unit_quaternion() {
        // Test the division of a quaternion by a unit quaternion
        assert_eq!(Q32::I / UQ32::J, -Q32::K);
        assert_eq!(Q64::J / UQ64::K, -Q64::I);
        assert_eq!(
            Q32::new(1.0, 2.0, 3.0, 4.0) / UQ32::K,
            Q32::new(4.0, -3.0, 2.0, -1.0)
        );
    }

    #[test]
    fn test_add_assign() {
        // Test the addition assignment operator
        let mut q = Q32::new(1.0, 2.0, 3.0, 4.0);
        q += 4.0;
        assert_eq!(q, Quaternion::new(5.0, 2.0, 3.0, 4.0));
    }

    #[test]
    fn test_sub_assign() {
        // Test the subtraction assignment operator
        let mut q = Q64::new(1.0, 2.0, 3.0, 4.0);
        q -= Q64::new(4.0, 8.0, 6.0, 1.0);
        assert_eq!(q, Quaternion::new(-3.0, -6.0, -3.0, 3.0));
    }

    #[test]
    fn test_mul_assign() {
        // Test the multiplication assignment operator
        let mut q = Q32::new(1.0, 2.0, 3.0, 4.0);
        q *= Q32::I;
        assert_eq!(q, Quaternion::new(-2.0, 1.0, 4.0, -3.0));
    }

    #[test]
    fn test_div_assign() {
        // Test the division assignment operator
        let mut q = Quaternion::new(1.0f32, 2.0f32, 3.0f32, 4.0f32);
        q /= 4.0f32;
        assert_eq!(q, Quaternion::new(0.25f32, 0.5f32, 0.75f32, 1.0f32));
    }

    #[test]
    #[allow(clippy::op_ref)]
    fn test_add_with_ref() {
        // Test the addition operator with references
        let lhs = Quaternion::new(1.0, 2.0, 3.0, 4.0);
        let rhs = Quaternion::new(5.0, 6.0, 7.0, 8.0);
        assert_eq!(lhs + rhs, &lhs + rhs);
        assert_eq!(lhs + rhs, lhs + &rhs);
        assert_eq!(lhs + rhs, &lhs + &rhs);
    }

    #[test]
    #[allow(clippy::op_ref)]
    fn test_sub_with_ref() {
        // Test the subtraction operator with references
        let lhs = Quaternion::new(1.0, 2.0, 3.0, 4.0);
        let rhs = Quaternion::new(5.0, 6.0, 7.0, 8.0);
        assert_eq!(lhs - rhs, &lhs - rhs);
        assert_eq!(lhs - rhs, lhs - &rhs);
        assert_eq!(lhs - rhs, &lhs - &rhs);
    }

    #[test]
    #[allow(clippy::op_ref)]
    fn test_mul_with_ref() {
        // Test the multiplication operator with references
        let lhs = Quaternion::new(1.0, 2.0, 3.0, 4.0);
        let rhs = Quaternion::new(5.0, 6.0, 7.0, 8.0);
        assert_eq!(lhs * rhs, &lhs * rhs);
        assert_eq!(lhs * rhs, lhs * &rhs);
        assert_eq!(lhs * rhs, &lhs * &rhs);
    }

    #[test]
    #[allow(clippy::op_ref)]
    fn test_div_with_ref() {
        // Test the division operator with references
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
        // Test the addition of a quaternion with a unit quaternion with
        // references
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
        // Test the subtraction of a quaternion with a unit quaternion with
        // references
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
        // Test the multiplication of a quaternion with a unit quaternion with
        // references
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
        // Test the division of a quaternion by a unit quaternion with
        // references
        let lhs = Quaternion::new(1.0, 2.0, 3.0, 4.0);
        let rhs = Quaternion::new(5.0, 6.0, 7.0, 8.0).normalize().unwrap();
        assert_eq!(lhs / rhs, &lhs / rhs);
        assert_eq!(lhs / rhs, lhs / &rhs);
        assert_eq!(lhs / rhs, &lhs / &rhs);
    }

    #[test]
    #[allow(clippy::op_ref)]
    fn test_add_real_with_ref() {
        // Test the addition of a real number to a quaternion with
        // references
        let lhs = Quaternion::new(1.0, 2.0, 3.0, 4.0);
        let rhs = 5.0;
        assert_eq!(lhs + rhs, &lhs + rhs);
        assert_eq!(lhs + rhs, lhs + &rhs);
        assert_eq!(lhs + rhs, &lhs + &rhs);
    }

    #[test]
    #[allow(clippy::op_ref)]
    fn test_sub_real_with_ref() {
        // Test the subtraction of a real number from a quaternion with
        // references
        let lhs = Quaternion::new(1.0, 2.0, 3.0, 4.0);
        let rhs = 5.0;
        assert_eq!(lhs - rhs, &lhs - rhs);
        assert_eq!(lhs - rhs, lhs - &rhs);
        assert_eq!(lhs - rhs, &lhs - &rhs);
    }

    #[test]
    #[allow(clippy::op_ref)]
    fn test_mul_real_with_ref() {
        // Test the multiplication of a quaternion by a real number with
        // references
        let lhs = Quaternion::new(1.0, 2.0, 3.0, 4.0);
        let rhs = 5.0;
        assert_eq!(lhs * rhs, &lhs * rhs);
        assert_eq!(lhs * rhs, lhs * &rhs);
        assert_eq!(lhs * rhs, &lhs * &rhs);
    }

    #[test]
    #[allow(clippy::op_ref)]
    fn test_div_real_with_ref() {
        // Test the division of a quaternion by a real number with references
        let lhs = Quaternion::new(1.0, 2.0, 3.0, 4.0);
        let rhs = 5.0;
        assert_eq!(lhs / rhs, &lhs / rhs);
        assert_eq!(lhs / rhs, lhs / &rhs);
        assert_eq!(lhs / rhs, &lhs / &rhs);
    }

    #[test]
    fn test_add_lhs_real() {
        // Test the addition of a real number to a quaternion
        assert_eq!(42.0 + Quaternion::I, Quaternion::new(42.0, 1.0, 0.0, 0.0));
        assert_eq!(
            1 + Quaternion::new(2, 4, 6, 8),
            Quaternion::new(3, 4, 6, 8)
        );
    }

    #[test]
    fn test_sub_lhs_real() {
        // Test the subtraction of a real number from a quaternion
        assert_eq!(42.0 - Quaternion::I, Quaternion::new(42.0, -1.0, 0.0, 0.0));
        assert_eq!(
            1 - Quaternion::new(2, 4, 6, 8),
            Quaternion::new(-1, -4, -6, -8)
        );
    }

    #[test]
    fn test_mul_lhs_real() {
        // Test the multiplication of a quaternion by a real number
        assert_eq!(42.0 * Quaternion::I, Quaternion::new(0.0, 42.0, 0.0, 0.0));
        assert_eq!(
            2 * Quaternion::new(1, 2, 3, 4),
            Quaternion::new(2, 4, 6, 8)
        );
    }

    #[test]
    fn test_div_lhs_real() {
        // Test the division of a quaternion by a real number{
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
        // Test the negation operator
        assert_eq!(
            -Q64::new(1.0, -2.0, 3.0, -4.0),
            Q64::new(-1.0, 2.0, -3.0, 4.0)
        );
    }

    #[test]
    fn test_unit_quaternion_add() {
        // Test the addition of unit quaternions
        assert_eq!(UQ32::I + UQ32::J, Q32::new(0.0, 1.0, 1.0, 0.0));
    }

    #[test]
    fn test_unit_quaternion_add_quaternion() {
        // Test the addition of unit quaternions and quaternions
        assert_eq!(UQ32::J + Q32::K, Q32::new(0.0, 0.0, 1.0, 1.0));
    }

    #[test]
    fn test_unit_quaternion_add_underlying() {
        // Test the addition of unit quaternions and underlying types
        assert_eq!(UQ32::J + 2.0f32, Q32::new(2.0, 0.0, 1.0, 0.0));
    }

    #[test]
    fn test_f32_add_unit_quaternion() {
        // Test the addition of `f32` values and `f32` unit quaternions
        assert_eq!(3.0f32 + UQ32::K, Q32::new(3.0, 0.0, 0.0, 1.0));
    }

    #[test]
    fn test_f64_add_unit_quaternion() {
        // Test the addition of `f64` values and `f64` unit quaternions
        assert_eq!(4.0f64 + UQ64::I, Q64::new(4.0, 1.0, 0.0, 0.0));
    }

    #[test]
    fn test_unit_quaternion_sub() {
        // Test the subtraction of unit quaternions
        assert_eq!(UQ32::I - UQ32::J, Q32::new(0.0, 1.0, -1.0, 0.0));
    }

    #[test]
    fn test_unit_quaternion_sub_quaternion() {
        // Test the subtraction of unit quaternions and quaternions
        assert_eq!(UQ32::J - Q32::K, Q32::new(0.0, 0.0, 1.0, -1.0));
    }

    #[test]
    fn test_unit_quaternion_sub_underlying() {
        // Test the subtraction of unit quaternions and underlying types
        assert_eq!(UQ32::J - 2.0f32, Q32::new(-2.0, 0.0, 1.0, 0.0));
    }

    #[test]
    fn test_f32_sub_unit_quaternion() {
        // Test the subtraction of `f32` values and `f32` unit quaternions
        assert_eq!(3.0f32 - UQ32::K, Q32::new(3.0, 0.0, 0.0, -1.0));
    }

    #[test]
    fn test_f64_sub_unit_quaternion() {
        // Test the subtraction of `f64` values and `f64` unit quaternions
        assert_eq!(4.0f64 - UQ64::I, Q64::new(4.0, -1.0, 0.0, 0.0));
    }

    #[test]
    fn test_unit_quaternion_mul() {
        // Test the multiplication of unit quaternions
        assert_eq!(UQ32::I * UQ32::J, UQ32::K);
    }

    #[test]
    fn test_unit_quaternion_mul_quaternion() {
        // Test the multiplication of unit quaternions and quaternions
        assert_eq!(UQ32::J * Q32::K, Q32::new(0.0, 1.0, 0.0, 0.0));
    }

    #[test]
    fn test_unit_quaternion_mul_underlying() {
        // Test the multiplication of unit quaternions and underlying types
        assert_eq!(UQ32::J * 2.0f32, Q32::new(0.0, 0.0, 2.0, 0.0));
    }

    #[test]
    fn test_f32_mul_unit_quaternion() {
        // Test the multiplication of `f32` values and `f32` unit quaternions
        assert_eq!(3.0f32 * UQ32::K, Q32::new(0.0, 0.0, 0.0, 3.0));
    }

    #[test]
    fn test_f64_mul_unit_quaternion() {
        // Test the multiplication of `f64` values and `f64` unit quaternions
        assert_eq!(4.0f64 * UQ64::I, Q64::new(0.0, 4.0, 0.0, 0.0));
    }

    #[test]
    fn test_unit_quaternion_div() {
        // Test the division of unit quaternions
        assert_eq!(UQ32::I / UQ32::J, -UQ32::K);
    }

    #[test]
    fn test_unit_quaternion_div_quaternion() {
        // Test the division of unit quaternions and quaternions
        assert_eq!(UQ32::J / Q32::K, Q32::new(0.0, -1.0, 0.0, 0.0));
    }

    #[test]
    fn test_unit_quaternion_div_underlying() {
        // Test the division of unit quaternions and underlying types
        assert_eq!(UQ32::J / 2.0f32, Q32::new(0.0, 0.0, 0.5, 0.0));
    }

    #[test]
    fn test_f32_div_unit_quaternion() {
        // Test the division of `f32` values and `f32` unit quaternions
        assert_eq!(3.0f32 / UQ32::K, Q32::new(0.0, 0.0, 0.0, -3.0));
    }

    #[test]
    fn test_f64_div_unit_quaternion() {
        // Test the division of `f64` values and `f64` unit quaternions
        assert_eq!(4.0f64 / UQ64::I, Q64::new(0.0, -4.0, 0.0, 0.0));
    }

    #[test]
    #[cfg(any(feature = "std", feature = "libm"))]
    #[allow(clippy::op_ref)]
    fn test_add_with_ref_unit_quaternion() {
        // Test the addition of unit quaternions with references
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
        // Test the subtraction of unit quaternions with references
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
        // Test the multiplication of unit quaternions with references
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
        // Test the division of unit quaternions with references
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
        // Test the addition of unit quaternions and quaternions with references
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
        // Test the subtraction of unit quaternions and quaternions with references
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
        // Test the multiplication of unit quaternions and quaternions with references
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
        // Test the division of unit quaternions and quaternions with references
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
        // Test the addition of unit quaternions and real numbers with references
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
        // Test the subtraction of unit quaternions and real numbers with references
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
        // Test the multiplication of unit quaternions and real numbers with references
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
        // Test the division of unit quaternions and real numbers with references
        let lhs = Quaternion::new(1.0, 2.0, 3.0, 4.0).normalize().unwrap();
        let rhs = 5.0;
        assert_eq!(lhs / rhs, &lhs / rhs);
        assert_eq!(lhs / rhs, lhs / &rhs);
        assert_eq!(lhs / rhs, &lhs / &rhs);
    }
}
