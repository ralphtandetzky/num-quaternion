#include <benchmark/benchmark.h>

#include <Eigen/Geometry>
#include <boost/qvm/quat.hpp>
#include <boost/qvm/quat_operations.hpp>
#include <boost/qvm/vec.hpp>
#include <cmath>
#include <random>
#include <ranges>
#include <vector>

constexpr size_t batch_size = 2000;

// Generates random unit quaternions
std::vector<std::array<float, 4>> generate_random_quaternions(size_t count)
{
    std::mt19937 rng;
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<std::array<float, 4>> quaternions;
    quaternions.reserve(count);

    for (size_t i = 0; i < count; ++i) {
        while (true) {
            std::array<float, 4> q{
                dist(rng), dist(rng), dist(rng), dist(rng)
            };
            float norm_squared =
                q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
            if (norm_squared != 0.0f) {
                // Normalize the quaternion
                float norm = std::sqrt(norm_squared);
                q[0] /= norm;
                q[1] /= norm;
                q[2] /= norm;
                q[3] /= norm;
                quaternions.push_back(q);
                break;
            }
        }
    }

    return quaternions;
}

// Benchmark for computing a rotation vector (scaled axis) from a quaternion
// using Boost QVM
static void BM_QuaternionToRotationVectorBoostQVM(benchmark::State & state)
{
    const auto inputs = generate_random_quaternions(batch_size);
    for (const auto & _ : state) {
        std::ignore = _;
        for (const auto & q_array : inputs) {
            boost::qvm::quat<float> q{
                q_array[0], q_array[1], q_array[2], q_array[3]
            };
            boost::qvm::vec<float, 3> axis{};
            float angle = boost::qvm::axis_angle(q, axis);
            boost::qvm::vec<float, 3> rotation_vector = { axis.a[0] * angle,
                                                          axis.a[1] * angle,
                                                          axis.a[2] * angle };
            benchmark::DoNotOptimize(rotation_vector);
        }
    }
    state.SetItemsProcessed(
        static_cast<int64_t>(state.iterations() * inputs.size()));
}

BENCHMARK(BM_QuaternionToRotationVectorBoostQVM);

// Benchmark for computing a rotation vector (scaled axis) from a quaternion
// using Eigen
static void BM_QuaternionToRotationVectorEigen(benchmark::State & state)
{
    const auto inputs = generate_random_quaternions(batch_size);
    for (const auto & _ : state) {
        std::ignore = _;
        for (const auto & q_array : inputs) {
            Eigen::Quaternionf q{
                q_array[0], q_array[1], q_array[2], q_array[3]
            };
            Eigen::AngleAxisf angle_axis(q);
            Eigen::Vector3f rotation_vector =
                angle_axis.angle() * angle_axis.axis();
            benchmark::DoNotOptimize(rotation_vector);
        }
    }
    state.SetItemsProcessed(
        static_cast<int64_t>(state.iterations() * inputs.size()));
}

BENCHMARK(BM_QuaternionToRotationVectorEigen);

template <typename T>
class Quaternion
{
public:
    std::array<T, 3> to_rotation_vector() const
    {
        // Compute the angle (2 * acos(w))
        T half_angle = std::acos(w);
        T angle = 2 * half_angle;

        // Compute sin(half_angle) to recover the axis
        T sin_half_angle = std::sin(half_angle);

        if (std::abs(sin_half_angle) < std::numeric_limits<T>::epsilon()) {
            // For very small angles, return zero vector
            return { 0, 0, 0 };
        }

        // Recover the axis and scale by the angle
        T scale = angle / sin_half_angle;
        return { scale * x, scale * y, scale * z };
    }

    T w, x, y, z;
};

// Benchmark for computing a rotation vector from a quaternion using a manual
// implementation
static void BM_QuaternionToRotationVectorManualImpl(benchmark::State & state)
{
    const auto inputs = generate_random_quaternions(batch_size);
    for (const auto & _ : state) {
        std::ignore = _;
        for (const auto & q_array : inputs) {
            Quaternion<float> q{
                q_array[0], q_array[1], q_array[2], q_array[3]
            };
            std::array<float, 3> rotation_vector = q.to_rotation_vector();
            benchmark::DoNotOptimize(rotation_vector);
        }
    }
    state.SetItemsProcessed(
        static_cast<int64_t>(state.iterations() * inputs.size()));
}

BENCHMARK(BM_QuaternionToRotationVectorManualImpl);

BENCHMARK_MAIN();
