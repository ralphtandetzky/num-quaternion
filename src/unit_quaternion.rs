use {
    crate::Quaternion,
    core::{
        borrow::Borrow,
        ops::{Add, Mul, Neg, Sub},
    },
    num_traits::{ConstOne, ConstZero, Inv, Num, One, Zero},
};

#[cfg(feature = "unstable")]
#[cfg(any(feature = "std", feature = "libm"))]
use crate::PureQuaternion;

#[cfg(any(feature = "std", feature = "libm"))]
use {
    core::num::FpCategory,
    num_traits::{Float, FloatConst},
};

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
/// [`Quaternion::normalize`](Quaternion::normalize) method. Alternatively, you
/// can use [`from_euler_angles`](UnitQuaternion::from_euler_angles),
/// [`from_rotation_vector`](UnitQuaternion::from_rotation_vector), or
/// [`from_rotation_matrix3x3`](UnitQuaternion::from_rotation_matrix3x3) to
/// obtain one. The inverse functions
/// [`to_euler_angles`](UnitQuaternion::to_euler_angles),
/// [`to_rotation_vector`](UnitQuaternion::to_rotation_vector), and
/// [`to_rotation_matrix3x3`](UnitQuaternion::to_rotation_matrix3x3) are also
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

#[cfg(feature = "unstable")]
#[cfg(any(feature = "std", feature = "libm"))]
impl<T> UnitQuaternion<T> {
    pub(crate) fn new(w: T, x: T, y: T, z: T) -> Self {
        Self(Quaternion::new(w, x, y, z))
    }
}

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
    ///
    /// # Example
    ///
    /// ```
    /// # use num_quaternion::UnitQuaternion;
    /// let uq = UnitQuaternion::from_euler_angles(1.5, 1.0, 3.0);
    /// ```
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
    ///
    /// # Example
    ///
    /// ```
    /// # use num_quaternion::UnitQuaternion;
    /// # use num_quaternion::EulerAngles;
    /// let angles = EulerAngles { roll: 1.5, pitch: 1.0, yaw: 3.0 };
    /// let uq = UnitQuaternion::from_euler_angles_struct(angles);
    /// ```
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
    ///
    /// # Example
    ///
    /// ```
    /// # use num_quaternion::UnitQuaternion;
    /// let uq = UnitQuaternion::from_euler_angles(1.5, 1.0, 3.0);
    /// let angles = uq.to_euler_angles();
    /// ```
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
            if sin_pitch >= one - epsilon {
                let pitch = half_pi; // 90 degrees
                let roll = T::zero();
                let yaw = -two * T::atan2(x, w);
                EulerAngles { roll, pitch, yaw }
            } else {
                let pitch = -half_pi; // -90 degrees
                let roll = T::zero();
                let yaw = two * T::atan2(x, w);
                EulerAngles { roll, pitch, yaw }
            }
        } else {
            // General case
            let pitch = sin_pitch.asin();
            let roll =
                T::atan2(two * (w * x + y * z), one - two * (x * x + y * y));
            let yaw =
                T::atan2(two * (w * z + x * y), one - two * (y * y + z * z));
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
    ///
    /// The results of this function may not be accurate, if the input has a
    /// very large norm. If the input vector is not finite (i. e. it contains
    /// an infinite or `NaN` component), then the result is filled with `NaN`.
    ///
    /// # Example
    ///
    /// ```
    /// # use num_quaternion::UnitQuaternion;
    /// let v = [1.0, 0.0, 0.0];
    /// let uq = UnitQuaternion::from_rotation_vector(&v);
    /// ```
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
    ///
    /// # Example
    ///
    /// ```
    /// # use num_quaternion::UnitQuaternion;
    /// let uq = UnitQuaternion::from_euler_angles(1.5, 1.0, 3.0);
    /// let v = uq.to_rotation_vector();
    /// ```
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
    ///
    /// # Example
    ///
    /// ```
    /// # use num_quaternion::UQ64;
    /// let uq = UQ64::I;
    /// let matrix = uq.to_rotation_matrix3x3();
    /// ```
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
            xy.clone() - wz.clone(),
            xz.clone() + wy.clone(),
            xy + wz,
            T::one() - xx.clone() - zz,
            yz.clone() - wx.clone(),
            xz - wy,
            yz + wx,
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
    ///
    /// # Example
    ///
    /// ```
    /// # use num_quaternion::UnitQuaternion;
    /// let matrix = [[0.0, 1.0, 0.0],
    ///               [0.0, 0.0, 1.0],
    ///               [1.0, 0.0, 0.0]];
    /// let uq = UnitQuaternion::from_rotation_matrix3x3(&matrix);
    /// ```
    pub fn from_rotation_matrix3x3(
        mat: &impl ReadMat3x3<T>,
    ) -> UnitQuaternion<T> {
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
            let qx = (*mat.at(2, 1) - *mat.at(1, 2)) / s;
            let qy = (*mat.at(0, 2) - *mat.at(2, 0)) / s;
            let qz = (*mat.at(1, 0) - *mat.at(0, 1)) / s;
            Self(Quaternion::new(qw, qx, qy, qz))
        } else if (m00 > m11) && (m00 > m22) {
            let s = (one + *m00 - *m11 - *m22).sqrt() * two; // s=4*qx
            let qw = (*mat.at(2, 1) - *mat.at(1, 2)) / s;
            let qx = quarter * s;
            let qy = (*mat.at(0, 1) + *mat.at(1, 0)) / s;
            let qz = (*mat.at(0, 2) + *mat.at(2, 0)) / s;
            if *mat.at(2, 1) >= *mat.at(1, 2) {
                Self(Quaternion::new(qw, qx, qy, qz))
            } else {
                Self(Quaternion::new(-qw, -qx, -qy, -qz))
            }
        } else if m11 > m22 {
            let s = (one + *m11 - *m00 - *m22).sqrt() * two; // s=4*qy
            let qw = (*mat.at(0, 2) - *mat.at(2, 0)) / s;
            let qx = (*mat.at(0, 1) + *mat.at(1, 0)) / s;
            let qy = quarter * s;
            let qz = (*mat.at(1, 2) + *mat.at(2, 1)) / s;
            if *mat.at(0, 2) >= *mat.at(2, 0) {
                Self(Quaternion::new(qw, qx, qy, qz))
            } else {
                Self(Quaternion::new(-qw, -qx, -qy, -qz))
            }
        } else {
            let s = (one + *m22 - *m00 - *m11).sqrt() * two; // s=4*qz
            let qw = (*mat.at(1, 0) - *mat.at(0, 1)) / s;
            let qx = (*mat.at(0, 2) + *mat.at(2, 0)) / s;
            let qy = (*mat.at(1, 2) + *mat.at(2, 1)) / s;
            let qz = quarter * s;
            if *mat.at(1, 0) >= *mat.at(0, 1) {
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
    ///
    /// # Example
    ///
    /// ```
    /// # use num_quaternion::UQ64;
    /// let a = [1.0, 0.0, 0.0];
    /// let b = [0.0, 1.0, 0.0];
    /// let uq = UQ64::from_two_vectors(&a, &b);
    /// let angles = uq.to_euler_angles();
    /// assert!((angles.yaw - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
    /// assert!((angles.pitch).abs() < 1e-10);
    /// assert!((angles.roll).abs() < 1e-10);
    /// ```
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

#[cfg(any(feature = "std", feature = "libm"))]
impl<T> UnitQuaternion<T>
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
    /// Note, that it may be more natural to use the method
    /// [`Quaternion::normalize`] instead of this function. Both functions are
    /// equivalent, but the method is more idiomatic in most cases.
    ///
    /// # Example
    ///
    /// ```
    /// # use num_quaternion::UnitQuaternion;
    /// let q = UnitQuaternion::from_euler_angles(1.5, 1.0, 3.0);
    /// let normalized = UnitQuaternion::normalize(q.into_inner());
    /// assert_eq!(normalized, Some(q));
    /// ```
    #[inline]
    pub fn normalize(q: Quaternion<T>) -> Option<Self> {
        let norm = q.norm();
        match norm.classify() {
            FpCategory::Normal | FpCategory::Subnormal => {
                Some(UnitQuaternion(q / norm))
            }
            _ => None,
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
    /// See also [`UnitQuaternion::one`], [`Quaternion::ONE`].
    ///
    /// # Example
    ///
    /// ```
    /// # use num_quaternion::UQ32;
    /// assert_eq!(UQ32::ONE, UQ32::one());
    /// ```
    pub const ONE: Self = Self(Quaternion::ONE);

    /// A constant `UnitQuaternion` of value $i$.
    ///
    /// See also [`UnitQuaternion::i`], [`Quaternion::I`].
    ///
    /// # Example
    ///
    /// ```
    /// # use num_quaternion::UQ32;
    /// assert_eq!(UQ32::I, UQ32::i());
    /// ```
    pub const I: Self = Self(Quaternion::I);

    /// A constant `UnitQuaternion` of value $j$.
    ///
    /// See also [`UnitQuaternion::j`], [`Quaternion::J`].
    ///
    /// # Example
    ///
    /// ```
    /// # use num_quaternion::UQ32;
    /// assert_eq!(UQ32::J, UQ32::j());
    /// ```
    pub const J: Self = Self(Quaternion::J);

    /// A constant `UnitQuaternion` of value $k$.
    ///
    /// See also [`UnitQuaternion::k`], [`Quaternion::K`].
    ///
    /// # Example
    ///
    /// ```
    /// # use num_quaternion::UQ32;
    /// assert_eq!(UQ32::K, UQ32::k());
    /// ```
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
    /// Returns the identity quaternion $1$.
    ///
    /// See also [`UnitQuaternion::ONE`], [`Quaternion::ONE`].
    ///
    /// # Example
    ///
    /// ```
    /// # use num_quaternion::{Q32, UQ32};
    /// assert_eq!(UQ32::one().into_inner(), Q32::one());
    /// ```
    #[inline]
    pub fn one() -> Self {
        Self(Quaternion::new(T::one(), T::zero(), T::zero(), T::zero()))
    }

    /// Returns the imaginary unit $i$.
    ///
    /// See also [`UnitQuaternion::I`], [`Quaternion::i`].
    ///
    /// # Example
    ///
    /// ```
    /// # use num_quaternion::{Q32, UQ32};
    /// assert_eq!(UQ32::i().into_inner(), Q32::new(0.0, 1.0, 0.0, 0.0));
    /// ```
    #[inline]
    pub fn i() -> Self {
        Self(Quaternion::i())
    }

    /// Returns the imaginary unit $j$.
    ///
    /// See also [`UnitQuaternion::J`], [`Quaternion::j`].
    ///
    /// # Example
    ///
    /// ```
    /// # use num_quaternion::{Q32, UQ32};
    /// assert_eq!(UQ32::j().into_inner(), Q32::new(0.0, 0.0, 1.0, 0.0));
    /// ```
    #[inline]
    pub fn j() -> Self {
        Self(Quaternion::j())
    }

    /// Returns the imaginary unit $k$.
    ///
    /// See also [`UnitQuaternion::K`], [`Quaternion::k`].
    ///
    /// # Example
    ///
    /// ```
    /// # use num_quaternion::{Q32, UQ32};
    /// assert_eq!(UQ32::k().into_inner(), Q32::new(0.0, 0.0, 0.0, 1.0));
    /// ```
    #[inline]
    pub fn k() -> Self {
        Self(Quaternion::k())
    }
}

impl<T> UnitQuaternion<T> {
    /// Returns the inner quaternion.
    ///
    /// This function does the same as
    /// [`into_inner`](UnitQuaternion::into_inner). Client code can decide
    /// which function to use based on the naming preference and context.
    ///
    /// # Example
    ///
    /// ```
    /// # use num_quaternion::{Q64, UQ64};
    /// let uq = UQ64::I;
    /// let q = uq.into_quaternion();
    /// assert_eq!(q, Q64::I);
    /// ```
    #[inline]
    pub fn into_quaternion(self) -> Quaternion<T> {
        self.0
    }

    /// Returns the inner quaternion.
    ///
    /// This function does the same as
    /// [`into_quaternion`](UnitQuaternion::into_quaternion). Client code can
    /// decide which function to use based on the naming preference and
    /// context.
    ///
    /// # Example
    ///
    /// ```
    /// # use num_quaternion::{Q64, UQ64};
    /// let uq = UQ64::I;
    /// let q = uq.into_inner();
    /// assert_eq!(q, Q64::I);
    /// ```
    #[inline]
    pub fn into_inner(self) -> Quaternion<T> {
        self.into_quaternion()
    }

    /// Returns a reference to the inner quaternion.
    ///
    /// # Example
    ///
    /// ```
    /// # use num_quaternion::{Q64, UQ64};
    /// let uq = UQ64::I;
    /// let q = uq.as_quaternion();
    /// assert_eq!(q, &Q64::I);
    /// ```
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
    /// Returns the conjugate quaternion, i. e. the imaginary part is negated.
    ///
    /// # Example
    ///
    /// ```
    /// # use num_quaternion::UQ64;
    /// let uq = UQ64::I;
    /// let conj = uq.conj();
    /// assert_eq!(conj, -uq);
    /// ```
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
    ///
    /// # Example
    ///
    /// ```
    /// # use num_quaternion::UQ32;
    /// let uq = UQ32::I;
    /// let inv = uq.inv();
    /// assert_eq!(inv, -UQ32::I);
    /// ```
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

impl<T> Mul<UnitQuaternion<T>> for UnitQuaternion<T>
where
    Quaternion<T>: Mul<Output = Quaternion<T>>,
{
    type Output = UnitQuaternion<T>;

    #[inline]
    fn mul(self, rhs: UnitQuaternion<T>) -> Self::Output {
        Self(self.into_inner() * rhs.into_inner())
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

impl<T> UnitQuaternion<T>
where
    T: Add<T, Output = T> + Mul<T, Output = T>,
{
    /// Computes the dot product of two unit quaternions interpreted as
    /// 4D real vectors.
    ///
    /// # Example
    ///
    /// ```
    /// # use num_quaternion::UQ32;
    /// let uq1 = UQ32::I;
    /// let uq2 = UQ32::J;
    /// let dot = uq1.dot(uq2);
    /// assert_eq!(dot, 0.0);
    /// ```
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
    ///
    /// # Panics
    ///
    /// Panics if the norm of the quaternion is too inaccurate to be
    /// renormalized.
    ///
    /// # Example
    ///
    /// ```
    /// # use num_quaternion::UnitQuaternion;
    /// let uq = UnitQuaternion::from_euler_angles(1.5, 1.0, 3.0);
    /// let adjusted = uq.adjust_norm();
    /// assert!((adjusted - uq).norm() < 1e-10);
    /// ```
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
    ///
    /// # Example
    ///
    /// ```
    /// # use num_quaternion::UQ64;
    /// let uq = UQ64::I;
    /// let rotated = uq.rotate_vector([1.0, 2.0, 3.0]);
    /// assert_eq!(rotated, [1.0, -2.0, -3.0]);
    /// ```
    pub fn rotate_vector(self, vector: [T; 3]) -> [T; 3] {
        let q = self.into_quaternion();
        let [vx, vy, vz] = vector;
        let v_q_inv = Quaternion::<T>::new(
            vx.clone() * q.x.clone()
                + vy.clone() * q.y.clone()
                + vz.clone() * q.z.clone(),
            vx.clone() * q.w.clone() - vy.clone() * q.z.clone()
                + vz.clone() * q.y.clone(),
            vx.clone() * q.z.clone() + vy.clone() * q.w.clone()
                - vz.clone() * q.x.clone(),
            vy * q.x.clone() - vx * q.y.clone() + vz * q.w.clone(),
        );
        let result = q * v_q_inv;
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
    ///
    /// # Example
    ///
    /// ```
    /// # use num_quaternion::UnitQuaternion;
    /// let uq1 = UnitQuaternion::from_euler_angles(1.5, 1.0, 3.0);
    /// let uq2 = UnitQuaternion::from_euler_angles(0.5, 2.0, 1.0);
    /// let uq = uq1.slerp(&uq2, 0.5);
    /// ```
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
    ///
    /// # Example
    ///
    /// ```
    /// # use num_quaternion::UnitQuaternion;
    /// let uq = UnitQuaternion::from_euler_angles(1.5, 1.0, 3.0);
    /// let sqrt = uq.sqrt();
    /// assert!((sqrt * sqrt - uq).norm() < 1e-10);
    /// ```
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

            UnitQuaternion(Quaternion::new(
                wx2 * half,
                c.x / wx2,
                c.y / wx2,
                c.z / wx2,
            ))
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
                UnitQuaternion(Quaternion::new(
                    wx2 * half,
                    c.x / wx2,
                    c.y / wx2,
                    c.z / wx2,
                ))
            } else if c.x.is_zero() && c.y.is_zero() && c.z.is_zero() {
                // Special case: input is -1. The result is `±i` with the same
                // signs for the imaginary parts as the input.
                UnitQuaternion(Quaternion::new(
                    zero,
                    one.copysign(c.x),
                    c.y,
                    c.z,
                ))
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

#[cfg(feature = "unstable")]
#[cfg(any(feature = "std", feature = "libm"))]
impl<T> UnitQuaternion<T>
where
    T: Float + FloatConst,
{
    /// Computes the natural logarithm of a unit quaternion.
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
    /// - If the input has a `NaN` value, then the result is `NaN` in all
    ///   components.
    ///
    /// # Example
    ///
    /// ```
    /// # use num_quaternion::Quaternion;
    /// let q = Quaternion::new(1.0f32, 2.0, 3.0, 4.0).normalize().unwrap();
    /// let ln_q = q.ln();
    /// ```
    pub fn ln(self) -> PureQuaternion<T> {
        // The square norm of the imaginary part.
        let sqr_norm_im =
            self.0.x * self.0.x + self.0.y * self.0.y + self.0.z * self.0.z;

        if sqr_norm_im <= T::epsilon() {
            // We're close to or on the positive real axis
            if self.0.w.is_sign_positive() {
                // This approximation leaves a relative error of less
                // than a floating point epsilon for the imaginary part
                PureQuaternion::new(self.0.x, self.0.y, self.0.z)
            } else if self.0.x.is_zero()
                && self.0.y.is_zero()
                && self.0.z.is_zero()
            {
                // We're on the negative real axis.
                PureQuaternion::new(
                    T::PI().copysign(self.0.x),
                    self.0.y,
                    self.0.z,
                )
            } else if sqr_norm_im.is_normal() {
                // We're close to the negative real axis. Compute the
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
                let f = T::PI() / norm_im + self.0.w.recip();

                PureQuaternion::new(self.0.x, self.0.y, self.0.z) * f
            } else {
                // The imaginary part is so small, that the norm of the
                // resulting imaginary part differs from `pi` by way
                // less than half an ulp. Therefore, it's sufficient to
                // normalize the imaginary part and multiply it by
                // `pi`.
                let f = T::min_positive_value().sqrt() * T::epsilon();
                let xf = self.0.x / f;
                let yf = self.0.y / f;
                let zf = self.0.z / f;
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
                PureQuaternion::new(
                    self.0.x * pi_div_f / im_norm_div_f,
                    self.0.y * pi_div_f / im_norm_div_f,
                    self.0.z * pi_div_f / im_norm_div_f,
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
                let f = T::min_positive_value().sqrt() * T::epsilon();
                let xf = self.0.x / f;
                let yf = self.0.y / f;
                let zf = self.0.z / f;
                let sqr_sum = xf * xf + yf * yf + zf * zf;
                sqr_sum.sqrt() * f
            };
            let angle = norm_im.atan2(self.0.w);
            let x = self.0.x * angle / norm_im;
            let y = self.0.y * angle / norm_im;
            let z = self.0.z * angle / norm_im;
            PureQuaternion::new(x, y, z)
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

#[cfg(all(feature = "rand", any(feature = "std", feature = "libm")))]
impl<T> rand::distr::Distribution<UnitQuaternion<T>>
    for rand::distr::StandardUniform
where
    T: Float,
    rand_distr::StandardNormal: rand_distr::Distribution<T>,
{
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> UnitQuaternion<T> {
        loop {
            let s = rand_distr::StandardNormal;
            let w = s.sample(rng);
            let x = s.sample(rng);
            let y = s.sample(rng);
            let z = s.sample(rng);
            let q = Quaternion::new(w, x, y, z);
            if let Some(q) = q.normalize() {
                return q;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use {
        crate::{Quaternion, UnitQuaternion, Q32, UQ32, UQ64},
        core::borrow::Borrow,
        num_traits::{ConstOne, One},
    };

    #[cfg(feature = "std")]
    use {
        core::hash::{Hash, Hasher},
        std::collections::hash_map::DefaultHasher,
    };

    #[cfg(any(feature = "std", feature = "libm"))]
    use {crate::Q64, num_traits::Inv};

    #[cfg(any(feature = "std", feature = "libm", feature = "serde"))]
    use crate::EulerAngles;

    #[cfg(any(feature = "std", feature = "libm"))]
    #[cfg(feature = "unstable")]
    use crate::PureQuaternion;

    /// Computes the hash value of `val` using the default hasher.
    #[cfg(feature = "std")]
    fn compute_hash(val: impl Hash) -> u64 {
        let mut hasher = DefaultHasher::new();
        val.hash(&mut hasher);
        hasher.finish()
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_hash_of_unit_quaternion_equals_hash_of_inner_quaternion() {
        // We test if the hash value of a unit quaternion is equal to the hash
        // value of the inner quaternion. This is required because
        // `UnitQuaternion` implements both `Hash` and `Borrow<Quaternion>`.
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
        let serialized =
            serde_json::to_string(&angles).expect("Failed to serialize angles");

        // Deserialize the JSON string back into angles
        let deserialized: EulerAngles<f64> = serde_json::from_str(&serialized)
            .expect("Failed to deserialize angles");

        // Assert that the deserialized angles are equal to the original
        assert_eq!(angles, deserialized);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_from_euler_angles() {
        // Test the conversion from Euler angles to quaternions
        assert!(
            (UQ32::from_euler_angles(core::f32::consts::PI, 0.0, 0.0)
                .into_quaternion()
                - Q32::I)
                .norm()
                < f32::EPSILON
        );
        assert!(
            (UQ64::from_euler_angles(0.0, core::f64::consts::PI, 0.0)
                .into_quaternion()
                - Q64::J)
                .norm()
                < f64::EPSILON
        );
        assert!(
            (UQ32::from_euler_angles(0.0, 0.0, core::f32::consts::PI)
                .into_quaternion()
                - Q32::K)
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

    #[cfg(all(
        feature = "unstable",
        feature = "rand",
        any(feature = "std", feature = "libm")
    ))]
    #[test]
    fn test_from_euler_angles_struct() {
        // Test the conversion from Euler angles to quaternions using the
        // function `from_euler_angles_struct`.
        let angles = EulerAngles {
            roll: 1.0,
            pitch: 2.0,
            yaw: 3.0,
        };
        let q = UQ64::from_euler_angles_struct(angles);
        let expected = UQ64::from_euler_angles(1.0, 2.0, 3.0);
        assert_eq!(q, expected);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_to_euler_angles() {
        // Test the conversion from quaternions to Euler angles
        let test_data = [
            Q64::new(1.0, 0.0, 0.0, 0.0),   // identity
            Q64::new(0.0, 1.0, 0.0, 0.0),   // 180 degree x axis
            Q64::new(0.0, 0.0, 1.0, 0.0),   // 180 degree y axis
            Q64::new(0.0, 0.0, 0.0, 1.0),   // 180 degree z axis
            Q64::new(1.0, 1.0, 1.0, 1.0),   // 120 degree xyz
            Q64::new(1.0, -2.0, 3.0, -4.0), // arbitrary
            Q64::new(4.0, 3.0, 2.0, 1.0),   // arbitrary
            Q64::new(1.0, 0.0, 1.0, 0.0),   // gimbal lock 1
            Q64::new(1.0, 1.0, 1.0, -1.0),  // gimbal lock 2
            Q64::new(1.0, 0.0, -1.0, 0.0),  // gimbal lock 3
            Q64::new(1.0, 1.0, -1.0, 1.0),  // gimbal lock 4
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
        // Test the conversion from rotation vectors to quaternions
        assert!(
            (UQ32::from_rotation_vector(&[core::f32::consts::PI, 0.0, 0.0])
                - Q32::I)
                .norm()
                < f32::EPSILON
        );
        assert!(
            (UQ64::from_rotation_vector(&[0.0, core::f64::consts::PI, 0.0])
                - Q64::J)
                .norm()
                < f64::EPSILON
        );
        assert!(
            (UQ32::from_rotation_vector(&[0.0, 0.0, core::f32::consts::PI])
                - Q32::K)
                .norm()
                < f32::EPSILON
        );
        let x = 2.0 * core::f64::consts::FRAC_PI_3 * (1.0f64 / 3.0).sqrt();
        assert!(
            (UQ64::from_rotation_vector(&[x, x, x])
                - Q64::new(0.5, 0.5, 0.5, 0.5))
            .norm()
                < 4.0 * f64::EPSILON
        );
        assert!(
            (UQ64::from_rotation_vector(&[-x, x, -x])
                - Q64::new(0.5, -0.5, 0.5, -0.5))
            .norm()
                < 4.0 * f64::EPSILON
        );
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_from_rotation_vector_infinite() {
        // Test `from_rotation_vector` for a vector with infinite components.
        let inf = f32::INFINITY;
        assert!(UQ32::from_rotation_vector(&[inf, 0.0, 0.0])
            .into_inner()
            .is_all_nan());
        assert!(UQ32::from_rotation_vector(&[inf, inf, inf])
            .into_inner()
            .is_all_nan());
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_from_rotation_vector_nan_input() {
        // Test `from_rotation_vector` for a vector with infinite components.
        let nan = f64::NAN;
        assert!(UQ64::from_rotation_vector(&[nan, 0.0, 0.0])
            .into_inner()
            .is_all_nan());
        assert!(UQ64::from_rotation_vector(&[nan, nan, nan])
            .into_inner()
            .is_all_nan());
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
        assert!(
            (rotation_vector[0] - core::f32::consts::FRAC_PI_2).abs()
                < f32::EPSILON
        );
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
        assert!(
            (rotation_vector[1] - core::f64::consts::PI).abs() < f64::EPSILON
        );
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
        let min_pos = f64::MIN_POSITIVE;
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
            [1.0, 0.0, 0.0, min_pos],
            [-1.0, 3.0 * min_pos, 2.0 * min_pos, min_pos],
            [1.0, 0.1, 0.0, 0.0],
            [-1.0, 0.0, 0.1, 0.0],
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
        // Test the rotation matrix of the identity quaternion
        let q = UQ64::ONE;
        let rot_matrix = q.to_rotation_matrix3x3();
        let expected = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        assert_eq!(rot_matrix, expected);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_rotation_matrix_90_degrees_x() {
        // Test the rotation matrix of a 90-degree rotation around the x-axis
        let q = Q64::new(1.0, 1.0, 0.0, 0.0).normalize().unwrap();
        let rot_matrix = q.to_rotation_matrix3x3();
        let expected = [
            1.0, 0.0, 0.0, //
            0.0, 0.0, -1.0, //
            0.0, 1.0, 0.0,
        ];
        for (r, e) in rot_matrix.iter().zip(expected) {
            assert!((r - e).abs() <= f64::EPSILON);
        }
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_rotation_matrix_90_degrees_y() {
        // Test the rotation matrix of a 90-degree rotation around the y-axis
        let q = Q64::new(1.0, 0.0, 1.0, 0.0).normalize().unwrap();
        let rot_matrix = q.to_rotation_matrix3x3();
        let expected = [
            0.0, 0.0, 1.0, //
            0.0, 1.0, 0.0, //
            -1.0, 0.0, 0.0,
        ];
        for (r, e) in rot_matrix.iter().zip(expected) {
            assert!((r - e).abs() <= f64::EPSILON);
        }
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_rotation_matrix_90_degrees_z() {
        // Test the rotation matrix of a 90-degree rotation around the z-axis
        let q = Q64::new(1.0, 0.0, 0.0, 1.0).normalize().unwrap();
        let rot_matrix = q.to_rotation_matrix3x3();
        let expected = [
            0.0, -1.0, 0.0, //
            1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0,
        ];
        for (r, e) in rot_matrix.iter().zip(expected) {
            assert!((r - e).abs() <= f64::EPSILON);
        }
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_rotation_matrix_120_degrees_xyz() {
        // Test the rotation matrix of a 120-degree rotation which rotates the
        // x-axis onto the y-axis, the y-axis onto the z-axis, and the z-axis
        // onto the x-axis
        let q = Q64::new(1.0, 1.0, 1.0, 1.0).normalize().unwrap();
        let rot_matrix = q.to_rotation_matrix3x3();
        let expected = [
            0.0, 0.0, 1.0, //
            1.0, 0.0, 0.0, //
            0.0, 1.0, 0.0,
        ];
        for (r, e) in rot_matrix.iter().zip(expected) {
            assert!((r - e).abs() <= f64::EPSILON);
        }
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_rotation_matrix_general() {
        // Test the rotation matrix of a general rotation
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
        // Test the quaternion corresponding to the identity matrix
        let identity: [[f32; 3]; 3] =
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let q = UQ32::from_rotation_matrix3x3(&identity);
        let expected = UQ32::ONE;
        assert_eq!(q, expected);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_rotation_x() {
        // Test the quaternion corresponding to a rotation around the x-axis
        let angle = core::f32::consts::PI / 5.0;
        let rotation_x: [[f32; 3]; 3] = [
            [1.0, 0.0, 0.0],
            [0.0, angle.cos(), -angle.sin()],
            [0.0, angle.sin(), angle.cos()],
        ];
        let q = UQ32::from_rotation_matrix3x3(&rotation_x);
        let expected = UQ32::from_rotation_vector(&[angle, 0.0, 0.0]);
        assert!((q - expected).norm() <= f32::EPSILON);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_rotation_y() {
        // Test the quaternion corresponding to a rotation around the y-axis
        let angle = 4.0 * core::f32::consts::PI / 5.0;
        let rotation_y: [[f32; 3]; 3] = [
            [angle.cos(), 0.0, angle.sin()],
            [0.0, 1.0, 0.0],
            [-angle.sin(), 0.0, angle.cos()],
        ];
        let q = UQ32::from_rotation_matrix3x3(&rotation_y);
        let expected = UQ32::from_rotation_vector(&[0.0, angle, 0.0]);
        assert!((q - expected).norm() <= f32::EPSILON);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_rotation_z() {
        // Test the quaternion corresponding to a rotation around the z-axis
        let angle = 3.0 * core::f64::consts::PI / 5.0;
        let rotation_z: [[f64; 3]; 3] = [
            [angle.cos(), -angle.sin(), 0.0],
            [angle.sin(), angle.cos(), 0.0],
            [0.0, 0.0, 1.0],
        ];
        let q = UQ64::from_rotation_matrix3x3(&rotation_z);
        let expected = UQ64::from_rotation_vector(&[0.0, 0.0, angle]);
        assert!((q - expected).norm() <= f64::EPSILON);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_arbitrary_rotation() {
        // Test the quaternion corresponding to an arbitrary rotation matrix
        let arbitrary_rotation: [[f32; 3]; 3] =
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let q = UnitQuaternion::from_rotation_matrix3x3(&arbitrary_rotation);
        let expected = Q32::new(1.0, 1.0, 1.0, 1.0).normalize().unwrap();
        assert!((q - expected).norm() <= f32::EPSILON);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_flat_array() {
        // Test the conversion between a flat array and a quaternion
        let angle = core::f32::consts::PI / 2.0;
        let rotation_z: [f32; 9] = [
            0.0, -1.0, 0.0, //
            1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0,
        ];
        let q = UQ32::from_rotation_matrix3x3(&rotation_z);
        let expected = UQ32::from_rotation_vector(&[0.0, 0.0, angle]);
        assert!((q - expected).norm() <= f32::EPSILON);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_from_rotation_to_rotation() {
        // Test the conversion from a rotation matrix to a quaternion and back
        let mat = [
            0.36f64, 0.864, -0.352, 0.48, 0.152, 0.864, 0.8, -0.48, -0.36,
        ];
        let q = UnitQuaternion::from_rotation_matrix3x3(&mat);
        let restored_mat = q.to_rotation_matrix3x3();
        for (x, e) in restored_mat.iter().zip(mat) {
            assert!((x - e).abs() <= 4.0 * f64::EPSILON);
        }
    }

    #[cfg(all(feature = "rand", any(feature = "std", feature = "libm")))]
    #[test]
    fn test_to_rotation_from_rotation() {
        // Test the conversion from a unit quaternion to a 3x3 matrix and back
        // in a randomized way.
        use rand::Rng;
        let mut rng = make_seeded_rng();
        for _ in 0..100000 {
            let q = rng.random::<UQ32>();
            let mat = q.to_rotation_matrix3x3();
            let restored_q = UQ32::from_rotation_matrix3x3(&mat);
            assert!(restored_q.0.w >= 0.0);
            let expected = if q.0.w >= 0.0 { q } else { -q };
            if (restored_q - expected).norm() > 4.0 * f32::EPSILON {
                assert!((restored_q - expected).norm() <= 8.0 * f32::EPSILON);
            }
        }
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_zero_vector_a() {
        // Test `from_two_vectors` for the case where the first vector is the
        // zero vector
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let q = UnitQuaternion::from_two_vectors(&a, &b);
        assert_eq!(q, UnitQuaternion::one());
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_zero_vector_b() {
        // Test `from_two_vectors` for the case where the second vector is the
        // zero vector
        let a = [1.0, 0.0, 0.0];
        let b = [0.0, 0.0, 0.0];
        let q = UnitQuaternion::from_two_vectors(&a, &b);
        assert_eq!(q, UnitQuaternion::one());
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_parallel_vectors() {
        // Test `from_two_vectors` for the case where the vectors are parallel
        let a = [1.0, 0.0, 0.0];
        let b = [2.0, 0.0, 0.0];
        let q = UnitQuaternion::from_two_vectors(&a, &b);
        assert_eq!(q, UnitQuaternion::one());
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_opposite_vectors() {
        // Test `from_two_vectors` for the case where the vectors are opposite
        let a = [1.0, 0.0, 0.0];
        let b = [-1.0, 0.0, 0.0];
        let q = UnitQuaternion::from_two_vectors(&a, &b);
        assert_eq!(q.as_quaternion().w, 0.0);
    }

    #[cfg(all(feature = "rand", any(feature = "std", feature = "libm")))]
    #[test]
    fn test_opposite_vectors_randomized() {
        // Test `from_two_vectors` for the case where the vectors are opposite
        // in a randomized way.
        use rand::Rng;
        let mut rng = make_seeded_rng();
        let mut gen_coord = move || rng.random::<f32>() * 2.0 - 1.0;
        for _ in 0..10000 {
            let a = [gen_coord(), gen_coord(), gen_coord()];
            let b = [-a[0], -a[1], -a[2]];
            let q = UnitQuaternion::from_two_vectors(&a, &b);
            assert!(q.as_quaternion().w.abs() <= f32::EPSILON);
        }
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_perpendicular_vectors() {
        // Test `from_two_vectors` for the case where the vectors are
        // perpendicular
        let a = [1.0f32, 0.0, 0.0];
        let b = [0.0f32, 1.0, 0.0];
        let q = UQ32::from_two_vectors(&a, &b);
        let expected = Q32::new(1.0, 0.0, 0.0, 1.0).normalize().unwrap();
        assert!((q - expected).norm() <= 2.0 * f32::EPSILON);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_non_normalized_vectors() {
        // Test `from_two_vectors` for the case where the vectors are not
        // normalized
        let a = [0.0, 3.0, 0.0];
        let b = [0.0, 5.0, 5.0];
        let q = UQ64::from_two_vectors(&a, &b);
        let expected =
            Q64::new(1.0, core::f64::consts::FRAC_PI_8.tan(), 0.0, 0.0)
                .normalize()
                .unwrap();
        assert!((q - expected).norm() <= 2.0 * f64::EPSILON);

        let a = [0.0, 3.0, 0.0];
        let b = [0.0, -5.0, 5.0];
        let q = UQ64::from_two_vectors(&a, &b);
        let expected =
            Q64::new(1.0, (3.0 * core::f64::consts::FRAC_PI_8).tan(), 0.0, 0.0)
                .normalize()
                .unwrap();
        assert!((q - expected).norm() <= 2.0 * f64::EPSILON);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_same_vector() {
        // Test `from_two_vectors` for the case where the vectors are the same
        let a = [1.0, 1.0, 1.0];
        let q = UnitQuaternion::from_two_vectors(&a, &a);
        assert_eq!(q, UnitQuaternion::one());
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_arbitrary_vectors() {
        // Test `from_two_vectors` for arbitrary vectors
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
        let expected = UQ64::from_rotation_vector(&[
            dir[0] * angle,
            dir[1] * angle,
            dir[2] * angle,
        ]);
        assert!((q - expected).norm() <= 2.0 * f64::EPSILON);
    }

    #[cfg(all(feature = "rand", any(feature = "std", feature = "libm")))]
    #[test]
    fn test_from_to_vectors_randomized() {
        // Test `from_two_vectors` in a randomized way.
        use rand::Rng;
        let mut rng = make_seeded_rng();
        let mut gen_coord = move || rng.random::<f32>() * 2.0 - 1.0;
        for _ in 0..100000 {
            let a = [gen_coord(), gen_coord(), gen_coord()];
            let b = [gen_coord(), gen_coord(), gen_coord()];
            let q = UQ32::from_two_vectors(&a, &b);
            let rotated_a = q.rotate_vector(a);
            let dot =
                rotated_a[0] * b[0] + rotated_a[1] * b[1] + rotated_a[2] * b[2];
            let b_norm_sqr = b[0] * b[0] + b[1] * b[1] + b[2] * b[2];
            let a_norm_sqr = a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
            let expected = (a_norm_sqr * b_norm_sqr).sqrt();
            let cos_angle = dot / expected;
            assert!((cos_angle - 1.0).abs() <= 8.0 * f32::EPSILON);
        }
    }

    #[test]
    fn test_default_unit_quaternion() {
        // Test that the default unit quaternion is equal to the identity
        assert_eq!(UQ32::default().into_quaternion(), Q32::ONE);
    }

    #[test]
    fn test_constant_one() {
        // Test that the constant `ONE` is equal to the identity quaternion
        assert_eq!(UQ32::ONE.into_quaternion(), Q32::ONE);
        assert_eq!(
            UnitQuaternion::<i32>::ONE.into_quaternion(),
            Quaternion::<i32>::ONE
        );
    }

    #[test]
    fn test_constant_i() {
        // Test that the constant unit quaternion `I` is equal to the
        // quaternion `I`
        assert_eq!(UQ32::I.into_quaternion(), Q32::I);
    }

    #[test]
    fn test_constant_j() {
        // Test that the constant unit quaternion `J` is equal to the
        // quaternion `J`
        assert_eq!(UQ32::J.into_quaternion(), Q32::J);
    }

    #[test]
    fn test_constant_k() {
        // Test that the constant unit quaternion `K` is equal to the
        // quaternion `K`
        assert_eq!(UQ32::K.into_quaternion(), Q32::K);
    }

    #[test]
    fn test_const_one() {
        // Test that the constant `ONE` is equal to the identity quaternion
        assert_eq!(<UQ32 as ConstOne>::ONE.into_quaternion(), Q32::ONE);
    }

    #[test]
    fn test_one_trait() {
        // Test the functions of the `One` trait for `UnitQuaternion<T>`
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
    fn test_one_func() {
        // Test the `one` method of the `UnitQuaternion` struct
        assert_eq!(UQ32::one().into_quaternion(), Q32::ONE);
    }

    #[test]
    fn test_unit_quaternion_i_func() {
        // Test the `i` method of the `UnitQuaternion` struct
        assert_eq!(UQ32::i().into_quaternion(), Q32::i());
    }

    #[test]
    fn test_unit_quaternion_j_func() {
        // Test the `j` method of the `UnitQuaternion` struct
        assert_eq!(UQ32::j().into_quaternion(), Q32::j());
    }

    #[test]
    fn test_unit_quaternion_k_func() {
        // Test the `k` method of the `UnitQuaternion` struct
        assert_eq!(UQ32::k().into_quaternion(), Q32::k());
    }

    #[test]
    fn test_into_quaternion() {
        // Test that the conversion from a unit quaternion to a quaternion
        // is correct.
        assert_eq!(UQ32::ONE.into_quaternion(), Q32::ONE);
        assert_eq!(UQ32::I.into_quaternion(), Q32::I);
        assert_eq!(UQ32::J.into_quaternion(), Q32::J);
        assert_eq!(UQ32::K.into_quaternion(), Q32::K);
    }

    #[test]
    fn test_into_inner() {
        // Test that the conversion from a unit quaternion to a quaternion
        // is correct.
        assert_eq!(UQ32::ONE.into_inner(), Q32::ONE);
        assert_eq!(UQ32::I.into_inner(), Q32::I);
        assert_eq!(UQ32::J.into_inner(), Q32::J);
        assert_eq!(UQ32::K.into_inner(), Q32::K);
    }

    #[test]
    fn test_as_quaternion() {
        // Test that the conversion from a unit quaternion to a quaternion
        // is correct.
        assert_eq!(UQ32::ONE.as_quaternion(), &Q32::ONE);
        assert_eq!(UQ32::I.as_quaternion(), &Q32::I);
        assert_eq!(UQ32::J.as_quaternion(), &Q32::J);
        assert_eq!(UQ32::K.as_quaternion(), &Q32::K);
    }

    #[test]
    fn test_borrow() {
        // Test that the conversion from a unit quaternion to a quaternion
        // is correct.
        assert_eq!(<UQ32 as Borrow<Q32>>::borrow(&UQ32::ONE), &Q32::ONE);
        assert_eq!(<UQ32 as Borrow<Q32>>::borrow(&UQ32::I), &Q32::I);
        assert_eq!(<UQ32 as Borrow<Q32>>::borrow(&UQ32::J), &Q32::J);
        assert_eq!(<UQ32 as Borrow<Q32>>::borrow(&UQ32::K), &Q32::K);
    }

    #[test]
    fn test_unit_quaternion_conj() {
        // Test the conjugate of unit quaternions
        assert_eq!(UQ32::ONE.conj(), UQ32::ONE);
        assert_eq!(UQ64::I.conj(), -UQ64::I);
        assert_eq!(UQ32::J.conj(), -UQ32::J);
        assert_eq!(UQ64::K.conj(), -UQ64::K);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_unit_quaternion_conj_with_normalize() {
        // Test the conjugate of unit quaternions
        assert_eq!(
            Q32::new(1.0, 2.0, 3.0, 4.0).normalize().unwrap().conj(),
            Q32::new(1.0, -2.0, -3.0, -4.0).normalize().unwrap()
        )
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_unit_quaternion_inv_func() {
        // Test the inverse of unit quaternions
        assert_eq!(
            UQ32::inv(&Q32::new(1.0, 2.0, 3.0, 4.0).normalize().unwrap()),
            Q32::new(1.0, 2.0, 3.0, 4.0).normalize().unwrap().conj()
        )
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_unit_quaternion_inv_trait() {
        // Test the `Inv` trait for unit quaternions
        assert_eq!(
            <UQ32 as Inv>::inv(
                Q32::new(1.0, 2.0, 3.0, 4.0).normalize().unwrap()
            ),
            Q32::new(1.0, 2.0, 3.0, 4.0).normalize().unwrap().conj()
        )
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_unit_quaternion_ref_inv_trait() {
        // Test the `Inv` trait for unit quaternion references
        assert_eq!(
            <&UQ32 as Inv>::inv(
                &Q32::new(1.0, 2.0, 3.0, 4.0).normalize().unwrap()
            ),
            Q32::new(1.0, 2.0, 3.0, 4.0).normalize().unwrap().conj()
        )
    }

    #[test]
    fn test_unit_quaternion_neg() {
        // Test the negation of unit quaternions
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
        // Test the adjustment of the norm of unit quaternions
        let mut q = UQ32::from_euler_angles(1.0, 0.5, 1.5);
        for _ in 0..25 {
            q = q * q;
        }
        assert!((q.into_quaternion().norm() - 1.0).abs() > 0.5);
        assert!(
            (q.adjust_norm().into_quaternion().norm() - 1.0).abs()
                <= 2.0 * f32::EPSILON
        );
    }

    #[test]
    fn test_unit_quaternion_rotate_vector_units() {
        // Test the rotation of unit vectors by unit quaternions
        let v = [1.0, 2.0, 3.0];
        assert_eq!(UQ32::I.rotate_vector(v), [1.0, -2.0, -3.0]);
        assert_eq!(UQ32::J.rotate_vector(v), [-1.0, 2.0, -3.0]);
        assert_eq!(UQ32::K.rotate_vector(v), [-1.0, -2.0, 3.0]);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_unit_quaternion_rotate_vector_normalized() {
        // Test the rotation of normalized vectors by unit quaternions
        let q = Q32::new(1.0, 1.0, 1.0, 1.0).normalize().unwrap();
        let v = [1.0, 2.0, 3.0];
        let result = q.rotate_vector(v);
        assert_eq!(result, [3.0, 1.0, 2.0]);
    }

    // Generates an iterator over unit quaternion test data
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
        // Test the spherical linear interpolation of unit quaternions
        // with `t = 0.0`
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
        // Test the spherical linear interpolation of unit quaternions
        // with `t = 1.0`
        use core::cmp::Ordering;

        for q1 in generate_unit_quaternion_data() {
            for q2 in generate_unit_quaternion_data() {
                let result = q1.slerp(&q2, 1.0);
                match q1.dot(q2).partial_cmp(&0.0) {
                    Some(Ordering::Greater) => {
                        assert!((result - q2).norm() <= f32::EPSILON)
                    }
                    Some(Ordering::Less) => {
                        assert!((result + q2).norm() <= f32::EPSILON)
                    }
                    _ => {}
                }
            }
        }
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_slerp_t_half() {
        // Test the spherical linear interpolation of unit quaternions
        // with `t = 0.5`
        use core::cmp::Ordering;

        for q1 in generate_unit_quaternion_data() {
            for q2 in generate_unit_quaternion_data() {
                let result = q1.slerp(&q2, 0.5);
                let dot_sign = match q1.dot(q2).partial_cmp(&0.0) {
                    Some(Ordering::Greater) => 1.0,
                    Some(Ordering::Less) => -1.0,
                    _ => continue, // uncertain due to rounding, better skip it
                };
                assert!(
                    (result - (q1 + dot_sign * q2).normalize().unwrap()).norm()
                        <= f32::EPSILON
                )
            }
        }
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_slerp_small_angle() {
        // Test the spherical linear interpolation of unit quaternions
        // with a small angles
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
        // Test the square root of the identity unit quaternion
        assert_eq!(UQ32::ONE.sqrt(), UQ32::ONE);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_sqrt_of_negative_identity() {
        // Test the square root of the negative identity unit quaternion
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
        // Test the square root of a general unit quaternion
        let c = Q64::new(1.0, 2.0, -3.0, 4.0).normalize().unwrap();
        let q = c.sqrt();
        assert!((q * q - c).norm() <= f64::EPSILON);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_sqrt_with_negative_real_part() {
        // Test the square root of a unit quaternion with a negative real part
        let c = Q64::new(-4.0, 2.0, -3.0, 1.0).normalize().unwrap();
        let q = c.sqrt();
        assert!((q * q - c).norm() <= f64::EPSILON);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_sqrt_with_subnormal_imaginary_parts() {
        // Test the square root of a unit quaternion with subnormal imaginary parts
        let min_positive = f64::MIN_POSITIVE;
        let q = Quaternion::new(-1.0, min_positive, min_positive, min_positive)
            .normalize()
            .unwrap();
        let result = q.sqrt();
        let expected = Quaternion::new(
            min_positive * 0.75f64.sqrt(),
            (1.0f64 / 3.0).sqrt(),
            (1.0f64 / 3.0).sqrt(),
            (1.0f64 / 3.0).sqrt(),
        )
        .normalize()
        .unwrap();
        assert!(
            (result.0.w - expected.0.w).abs()
                <= 2.0 * expected.0.w * f64::EPSILON
        );
        assert!(
            (result.0.x - expected.0.x).abs()
                <= 2.0 * expected.0.x * f64::EPSILON
        );
        assert!(
            (result.0.y - expected.0.y).abs()
                <= 2.0 * expected.0.y * f64::EPSILON
        );
        assert!(
            (result.0.z - expected.0.z).abs()
                <= 2.0 * expected.0.z * f64::EPSILON
        );
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[cfg(feature = "unstable")]
    #[test]
    fn test_ln_of_identity() {
        // Test the natural logarithm of the identity unit quaternion
        assert_eq!(UQ32::ONE.ln(), PureQuaternion::ZERO);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[cfg(feature = "unstable")]
    #[test]
    fn test_ln_of_normal_case() {
        // Test the natural logarithm of a unit quaternion
        let q = Q64::new(1.0, 2.0, 3.0, 4.0);
        let p = q.normalize().expect("Failed to normalize quaternion").ln();
        assert!((p.z / p.x - q.z / q.x).abs() <= 4.0 * f64::EPSILON);
        assert!((p.y / p.x - q.y / q.x).abs() <= 4.0 * f64::EPSILON);
        assert!((p.norm() - 29.0f64.sqrt().atan()).abs() <= 4.0 * f64::EPSILON);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[cfg(feature = "unstable")]
    #[test]
    fn test_ln_near_positive_real_axis() {
        // Test close to the positive real axis
        let q = Quaternion::new(1.0, 1e-10, 1e-10, 1e-10)
            .normalize()
            .unwrap();
        let ln_q = q.ln();
        let expected = PureQuaternion::new(1e-10, 1e-10, 1e-10); // ln(1) = 0 and imaginary parts small
        assert!((ln_q - expected).norm() <= 1e-11);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[cfg(feature = "unstable")]
    #[test]
    fn test_ln_negative_real_axis() {
        // Test on the negative real axis
        let q = Q32::new(-1.0, 0.0, 0.0, 0.0).normalize().unwrap();
        let ln_q = q.ln();
        let expected = PureQuaternion::new(core::f32::consts::PI, 0.0, 0.0); // ln(-1) = pi*i
        assert!(
            (ln_q - expected).norm() <= core::f32::consts::PI * f32::EPSILON
        );
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[cfg(feature = "unstable")]
    #[test]
    fn test_ln_near_negative_real_axis() {
        // Test a quaternion with a tiny imaginary part
        use core::f32;
        let q = Q32::new(-2.0, 346.0 * f32::EPSILON, 0.0, 0.0);
        let uq = q.normalize().unwrap();
        let ln_uq = uq.ln();
        let expected =
            PureQuaternion::new(f32::consts::PI + q.x / q.w, 0.0, 0.0);
        assert!((ln_uq - expected).norm() <= 8.0 * f32::EPSILON);

        let q = Q32::new(-1.0, f32::MIN_POSITIVE / 192.0, 0.0, 0.0);
        let uq = q.normalize().unwrap();
        let ln_uq = uq.ln();
        let expected = PureQuaternion::new(f32::consts::PI, 0.0, 0.0);
        assert_eq!(ln_uq, expected);
    }

    #[cfg(all(feature = "serde", any(feature = "std", feature = "libm")))]
    #[test]
    fn test_serde_unit_quaternion() {
        // Create a sample quaternion
        let q = Q64::new(1.0, 2.0, 3.0, 4.0).normalize().unwrap();

        // Serialize the quaternion to a JSON string
        let serialized =
            serde_json::to_string(&q).expect("Failed to serialize quaternion");

        // Deserialize the JSON string back into a quaternion
        let deserialized: UQ64 = serde_json::from_str(&serialized)
            .expect("Failed to deserialize quaternion");

        // Assert that the deserialized quaternion is equal to the original
        assert_eq!(q, deserialized);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_serde_unit_quaternion_k() {
        // Create a sample quaternion
        let q = UQ64::K;

        // Serialize the quaternion to a JSON string
        let serialized =
            serde_json::to_string(&q).expect("Failed to serialize quaternion");

        // Deserialize the JSON string back into a quaternion
        let deserialized: UQ64 = serde_json::from_str(&serialized)
            .expect("Failed to deserialize quaternion");

        // Assert that the deserialized quaternion is equal to the original
        assert_eq!(q, deserialized);
    }

    #[cfg(all(feature = "rand", any(feature = "std", feature = "libm")))]
    fn make_seeded_rng() -> impl rand::Rng {
        use rand::SeedableRng;
        rand::rngs::SmallRng::seed_from_u64(0x7F0829AE4D31C6B5)
    }

    #[cfg(all(feature = "rand", any(feature = "std", feature = "libm")))]
    #[test]
    fn test_unit_quaternion_sample_six_sigma() {
        // Test that the sample distribution of unit quaternions is uniform
        use rand::distr::{Distribution, StandardUniform};
        let num_iters = 1_000_000;
        let mut sum: Q64 = num_traits::Zero::zero();
        let rng = make_seeded_rng();
        for q in Distribution::<UQ64>::sample_iter(StandardUniform, rng)
            .take(num_iters)
        {
            sum += q;
        }

        let sum_std_dev = (num_iters as f64).sqrt();
        // The statistical probability of failure is 1.973e-9, unless there is
        // a bug.
        assert!(sum.norm() < 6.0 * sum_std_dev);
    }

    #[cfg(all(feature = "rand", any(feature = "std", feature = "libm")))]
    #[test]
    fn test_unit_quaternion_sample_half_planes() {
        // Test that the sample distribution of unit quaternions is uniform
        use rand::{
            distr::{Distribution, StandardUniform},
            Rng,
        };
        let num_iters = 1_000_000;
        let mut rng = make_seeded_rng();
        const NUM_DIRS: usize = 10;
        let dirs: [UQ64; NUM_DIRS] = [
            UQ64::ONE,
            UQ64::I,
            UQ64::J,
            UQ64::K,
            Q64::new(1.0, 2.0, 3.0, 4.0).normalize().unwrap(),
            Q64::new(4.0, -3.0, 2.0, -1.0).normalize().unwrap(),
            rng.random(),
            rng.random(),
            rng.random(),
            rng.random(),
        ];
        let mut counters = [0; NUM_DIRS];
        for q in Distribution::<UQ64>::sample_iter(StandardUniform, rng)
            .take(num_iters)
        {
            for (dir, counter) in dirs.iter().zip(counters.iter_mut()) {
                if q.dot(*dir) > 0.0 {
                    *counter += 1;
                }
            }
        }

        let six_sigma = 3 * (num_iters as f64).sqrt() as i32;
        let expected_count = num_iters as i32 / 2;
        // The statistical probability of failure of the following loop is
        // 1.973e-8, unless there is a bug.
        for counter in counters {
            assert!((counter - expected_count).abs() < six_sigma);
        }
    }
}
