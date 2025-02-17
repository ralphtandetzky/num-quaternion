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
