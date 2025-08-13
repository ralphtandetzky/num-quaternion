#include <benchmark/benchmark.h>

#include <Eigen/Geometry>
#include <boost/qvm/quat.hpp>
#include <boost/qvm/quat_operations.hpp>
#include <boost/qvm/vec.hpp>
#include <cmath>
#include <random>
#include <ranges>
#include <vector>

// Generates random Rodriguez vectors with norm < PI
std::vector<std::array<float, 3>> generate_random_vectors(size_t count)
{
    std::mt19937 rng(0x7F0829AE4D31C6B5);
    std::uniform_real_distribution<float> dist(-M_PI, M_PI);

    std::vector<std::array<float, 3>> vectors;
    vectors.reserve(count);

    for (size_t i = 0; i < count; ++i) {
        while (true) {
            std::array<float, 3> v{ dist(rng), dist(rng), dist(rng) };
            float norm_squared = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
            if (norm_squared < M_PI * M_PI) {
                vectors.push_back(v);
                break;
            }
        }
    }

    return vectors;
}

// Computes the norms of the given vectors and returns them in an array.
std::vector<float> compute_norms(
    const std::vector<std::array<float, 3>> & vectors)
{
    std::vector<float> norms;
    norms.reserve(vectors.size());
    for (const auto & v : vectors) {
        float norm = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        norms.push_back(norm);
    }
    return norms;
}

// Computes the normalized vectors from the given vectors
std::vector<std::array<float, 3>> compute_normalized_vectors(
    std::vector<std::array<float, 3>> vectors)
{
    for (auto & v : vectors) {
        float norm = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        if (norm != 0) {
            v[0] /= norm;
            v[1] /= norm;
            v[2] /= norm;
        }
    }
    return std::move(vectors);
}

// Benchmark for computing a quaternion from an axis vector and an angle
// using Boost QVM
static void BM_QuaternionFromRotationVectorBoostQVM(benchmark::State & state)
{
    const auto inputs = generate_random_vectors(2000);
    const auto angles = compute_norms(inputs);
    for (auto _ : state) {
        for (size_t i = 0; i < inputs.size(); ++i) {
            const auto & axis = inputs[i];
            float angle = angles[i];
            boost::qvm::vec<float, 3> axis_c = { axis[0], axis[1], axis[2] };
            boost::qvm::quat<float> q;
            boost::qvm::set_rot(q, axis_c, angle);
            benchmark::DoNotOptimize(q);
        }
    }
    state.SetItemsProcessed(state.iterations() * inputs.size());
}

BENCHMARK(BM_QuaternionFromRotationVectorBoostQVM);

// Benchmark for computing a quaternion from an axis vector and an angle
// using Eigen
static void BM_QuaternionFromRotationVectorEigen(benchmark::State & state)
{
    auto inputs = generate_random_vectors(2000);
    const auto angles = compute_norms(inputs);
    inputs = compute_normalized_vectors(std::move(inputs));
    for (auto _ : state) {
        for (size_t i = 0; i < inputs.size(); ++i) {
            const auto & v = inputs[i];
            float angle = angles[i];
            auto q = Eigen::Quaternionf{ Eigen::AngleAxisf{
                angle, Eigen::Vector3f(v[0], v[1], v[2]) } };
            benchmark::DoNotOptimize(q);
        }
    }
    state.SetItemsProcessed(state.iterations() * inputs.size());
}

BENCHMARK(BM_QuaternionFromRotationVectorEigen);

template <typename T>
class Quaternion
{
public:
    Quaternion(T w, T x, T y, T z)
        : w(w)
        , x(x)
        , y(y)
        , z(z)
    {
    }

    static Quaternion from_rotation_vector(const std::array<T, 3> & v)
    {
        T norm = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        T half_angle = norm / 2;
        T sin_half_angle = std::sin(half_angle);
        T imag_factor = sin_half_angle / norm;
        return Quaternion(std::cos(half_angle),
                          imag_factor * v[0],
                          imag_factor * v[1],
                          imag_factor * v[2]);
    }

    T w, x, y, z;
};

// Benchmark for computing a quaternion from a vector (direction = axis,
// norm = angle) using a manual implementation
static void BM_QuaternionFromRotationVectorManualImpl(benchmark::State & state)
{
    auto inputs = generate_random_vectors(2000);
    for (auto _ : state) {
        for (const auto & v : inputs) {
            Quaternion<float> q = Quaternion<float>::from_rotation_vector(v);
            benchmark::DoNotOptimize(q);
        }
    }
    state.SetItemsProcessed(state.iterations() * inputs.size());
}

BENCHMARK(BM_QuaternionFromRotationVectorManualImpl);

BENCHMARK_MAIN();
