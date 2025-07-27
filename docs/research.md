# Norm calculation

## Measurements

### Run-time Measurements

The table below summarizes the run-times of various quaternion norm
calculation algorithms.

Algorithm                                   | run-time [ns]
--------------------------------------------|--------------
`num_quaternion::Q32::norm`                 |    2.7661
`num_quaternion::Q32::fast_norm`            |    1.9039
`num_quaternion::Q64::norm`                 |    3.0271
`num_quaternion::Q64::fast_norm`            |    2.8275
hypot implementation `f32`                  |   15.956
sqrt(a² + b² + c² + d²) `f32`               |    1.8853
hypot implementation `f64`                  |   31.840
sqrt(a² + b² + c² + d²) `f64`               |    2.8441
`quaternion::len<f32>`                      |    1.9660
`quaternion::len<f64>`                      |    2.8473
`quaternion_core::norm<f32>`                |   17.562
`quaternion_core::norm<f64>`                |   41.206
`nalgebra::geometry::Quaternion<f32>::norm` |    1.8997
`nalgebra::geometry::Quaternion<f64>::norm` |    2.8803
`micromath::Quaternion::norm`               |    2.8311
`boost::qvm::mag`, f32 (C++)                |    1.86
`boost::qvm::mag`, f64 (C++)                |    2.84
`Eigen::Quaternionf::norm` (C++)            |    1.51
`Eigen::Quaterniond::norm` (C++)            |    2.86
sqrt(a² + b² + c² + d²) `f32` (C++)         |    1.86
sqrt(a² + b² + c² + d²) `f64` (C++)         |    2.86
hypot(hypot(w,x), hypot(y,z)) `f32` (C++)   |   15.5
hypot(hypot(w,x), hypot(y,z)) `f64` (C++)   |   31.1

These run-times were measured using micro benchmarks, which measure
inverse throughput rather than latency. The measurements have a relative
error of around 2%. All measurements were made for the input quaternion
`1 + 2i + 3j + 4k`.

For Rust, we used the `criterion` crate for benchmarking and rustc 1.81
for compilation. For C++, we used Google Benchmark and Clang 16.0.0. We
manually implemented the norm algorithm in both languages (the rows
with `sqrt(a² + b² + c² + d²)` at the beginning). The manual `f32`
implementations in Rust and C++ have identical run-times. This can be
confirmed by experiments on [godbolt.org](https://godbolt.org/z/Yd8197e4a),
where the code generation for Rustc 1.81 is slightly better than for
Clang 16.0.0, but the difference does not affect run-time.

### Accuracy Measurements

The table below presents the accuracy measurements of various quaternion norm
calculation implementations across different scales.

The columns represent the scale of the input quaternion, and the rows
represent the implementation of the quaternion norm. The values in the
table are the relative RMS error of the quaternion norm.

The column `t [ns]` shows the run-time of the algorithm in nanoseconds. The
column `1.0` is for quaternions with all components uniformly sampled from
the range [-1.0, 1.0]. The column `sqrt(MIN_POS)` is for quaternions with
all components in the range [sqrt(MIN_POS), sqrt(MIN_POS)], where `MIN_POS`
is the smallest positive normal 32-bit IEEE-754 floating point value.
Similarly for `MIN_POS` and `MAX / 2`, where `MAX` is the largest finite `f32` value.

The scales include 1.0, the square root of the minimum positive value
(`sqrt(MIN_POS)`), the minimum positive value (`MIN_POS`), and half of the
maximum value (`MAX / 2`). The table also includes the run-time in nanoseconds
for each implementation.

Implementation \ Scale               | t [ns] |     1.0     | sqrt(MIN_POS) |   MIN_POS    | MAX / 2
-------------------------------------|--------|-------------|---------------|--------------|--------
num_quaternion::Q32::norm            |   2.77 |      0.2723 |        0.2748 |       0.3824 |  0.2723
num_quaternion::Q32::fast_norm       |   1.85 |      0.2723 |        0.7485 | 8388608      |     inf
hypot implementation                 |  15.77 |      0.2762 |        0.2762 |       0.4229 |  0.2763
sqrt(a² + b² + c² + d²)              |   1.85 |      0.2778 |        0.7489 | 8388608      |     inf
quaternion::len                      |   1.95 |      0.2778 |        0.7489 | 8388608      |     inf
quaternion_core::norm                |  17.37 |      0.3026 |        0.3026 |       0.4482 |  0.3027
nalgebra::geometry::Quaternion::norm |   1.89 |      0.2723 |        0.7485 | 8388608      |     inf
micromath::Quaternion::magnitude     |   2.80 | 239721.1370 |  1770435.9112 |    6.0089e25 | 8388608
boost::qvm::mag (C++)                |   1.86 |      0.2779 |        0.7324 | 8388608      |     inf
hypot implementation (C++)           |  15.5  |      0.2762 |        0.2762 |       0.4226 |  0.2761
sqrt(a² + b² + c² + d²) (C++)        |   1.86 |      0.2779 |        0.7324 | 8388608      |     inf
Eigen::Quaternionf::norm (C++)       |   1.51 |      0.2724 |        0.7319 | 8388608      |     inf

The number of samples used for the accuracy measurements is large enough to
guarantee correct numbers except for the last digit displayed here.

## Evaluation

### Run-times

The implementations of

* `num_quaternion::Q32::fast_norm`,
* `quaternion::len`,
* `nalgebra::geometry::Quaternion::norm`,
* `boost::qvm::mag`,
* `Eigen::Quaternionf::norm`

are all very similar to `sqrt(a² + b² + c² + d²)`. This family of functions
has the simplest implementation and the fastest run-times (around `1.86ns`,
except the `Eigen` implementation). Their run-time differences are within the
magnitude of statistical noise. The `Eigen` implementation has a run-time of
`1.51ns`, achieved through special SIMD instructions.

Next is `num_quaternion::Q32::norm` with about `2.77ns` run-time. This can
be justified by its very accurate results for all scaling factors.

Almost as fast is `micromath` (`2.80ns`), but it is very inaccurate (5%
relative error bound) compared to the other algorithms. The implementation
may be faster than their `sqrt()` alternatives if the hardware does not
support built-in floating point square roots.

The `quaternion_core::norm` algorithm computes the norm using
`a.hypot(b).hypot(c).hypot(d)`. This is more than 6 times slower than
all the previous implementations. Even our manual implementation
`a.hypot(b).hypot(c.hypot(d))` is slightly faster and consistently more
accurate. The run-times for the C++ `hypot()` implementations are about
the same.

### Accuracy

All norm implementations except `micromath` have very similar RMS relative accuracies in the unscaled hypercube (column `1.0`).

For a very small scaling factor `MIN_POS`, the norm square `a² + b² + c² + d²`
underflows to zero for all points in the scaled hypercube. This is why the four
fastest implementations all return zero, which is the square root of zero.
Consequently, for quaternions where the squares of the components all underflow
to zero, the relative error is 1, which is 8388608 machine epsilons.

Similarly, if one component's square exceeds the floating point
range, then it overflows and the overall result of the implementations
is infinity. All six of the fast implementations suffer from this
inaccuracy and thus get an infinite relative error for the scale
factor `MAX / 2`. These implementations fail for the case where the
result is larger than `sqrt(MAX) ≈ 1.84 * 10^19`. (`MAX ≈ 1.18 * 10^38`
is the largest finite 32-bit floating point number.) For results
that are less than `sqrt(MIN_POS) ≈ 1.08 * 10^-19`, the results
become increasingly inaccurate. If the true result is less than
`sqrt(MIN_POS * EPSILON / 2) ≈ 2.65 * 10^-23`, the algorithms always
return zero as a result. (`MIN_POS * EPSILON ≈ 1.40 * 10^-45` is the
smallest positive 32-bit floating point number.) In these cases, the
relative error is 1 unless the true result is really zero.

The `hypot()` function is very accurate for all valid inputs and thus
the implementations using it are very accurate for all inputs. These
implementations are

* Our manual implementations in Rust and C++, and
* `quaternion_core::norm`.

The `num_quaternion::Q32::norm` implementation outperforms both of these
with the following trick: It computes the norm square `a² + b² + c² + d²`
and then checks the result to decide if taking the square root is
likely accurate or not. If the square norm is not finite or less than
`2 * MIN_POS`, then the quaternion is brought into a good range by scaling
with an appropriate factor first and the norm of that is computed with the
`num_quaternion::Quaternion::fast_norm` algorithm. Eventually, the result
is scaled back appropriately to give the final result. The scaling is done
by a power of two, so the calculation has no rounding error in that step.
This is more accurate than the `hypot` implementations, because calculating
the square root in the last step only approximately halves the relative error previously introduced. This is because `sqrt(1 + epsilon) ≈ 1 + epsilon / 2`.
The `hypot` function does not provide this nice effect.

One final remark on accuracy: In the column `1.0`, the measurements look very close, putting aside the `micromath` accuracy. However, the differences are
_not_ due to statistical noise here. The number of measurements is large enough
to guarantee that all digits except the last of the RMS error are accurate.
We can see there that in that range,

* `num_quaternion::Q32::norm`,
* `num_quaternion::Q32::fast_norm`,
* `nalgebra::geometry::Quaternion::norm`, and
* `Eigen::Quaternionf::norm`

produce exactly the same result, which is the best result of the investigated
algorithms. This is because in that range, they use exactly the same formula:
`sqrt((a²+c²)+(b²+d²))`. This provides more accurate results on average than
the formula `sqrt(((a²+b²)+c²)+d²)`. Furthermore, the first formula has better
worst-case relative errors. Finally, it is parallelized better in modern
hardware, which leads to less latency of the computation.

### Trade-offs

If you only care about speed, then

* `num_quaternion::Q32::fast_norm`, and
* `nalgebra::geometry::Quaternion::norm`

are your best choices in Rust. These algorithms also provide very good accuracy
if the resulting norm is in the range from `2.65 * 10^-23` up to `1.84 * 10^19`
unless the true result is exactly zero. This is the case for most practical
purposes. If you compute with numbers with exponents whose absolute value is
larger than this and still care about very accurate results, then you should
probably use double precision floating point values. If your hardware does
not support double precision floating point computations (such as a 32-bit
ARM processor), then double precision must be emulated by software, which is
typically very slow.

If you want the most accurate results and you don't care about performance,
then you should consider using 64-bit floating point computations; the
`num_quaternion::Q64::fast_norm` function is your friend. If you just want
a 32-bit floating point quaternion implementation that works correctly for
all inputs and is really fast, `num_quaternion::Q32::norm` is the best pick.

Of course, there are other factors that are important for the decision as
well. For example, if you are already using the `nalgebra` crate, I will
have a hard time convincing you to add another dependency to your project
just to use a really awesome quaternion norm implementation. But I'd
appreciate it if you use `num_quaternion` anyway. 😜
