#include <benchmark/benchmark.h>

#include <Eigen/Geometry>
#include <boost/qvm/quat.hpp>
#include <boost/qvm/quat_operations.hpp>
#include <cmath>

// NOLINTBEGIN(*-magic-numbers): We allow magic numbers for benchmarking
// purposes. All numbers used here are arbitrary and repetition can be viewed
// as coincidental.

static void BM_QuaternionF32NormBoostQVM(benchmark::State & state)
{
    for (const auto & _ : state) {
        std::ignore = _;
        boost::qvm::quat<float> q{ 1.0f, 2.0f, 3.0f, 4.0f };
        benchmark::DoNotOptimize(q);
        benchmark::ClobberMemory();
        benchmark::DoNotOptimize(boost::qvm::mag(q));
    }
}

BENCHMARK(BM_QuaternionF32NormBoostQVM);

static void BM_QuaternionF64NormBoostQVM(benchmark::State & state)
{
    for (const auto & _ : state) {
        std::ignore = _;
        boost::qvm::quat<double> q{ 1.0f, 2.0f, 3.0f, 4.0f };
        benchmark::DoNotOptimize(q);
        benchmark::ClobberMemory();
        benchmark::DoNotOptimize(boost::qvm::mag(q));
    }
}

BENCHMARK(BM_QuaternionF64NormBoostQVM);

static void BM_QuaternionF32NormEigen(benchmark::State & state)
{
    for (const auto & _ : state) {
        std::ignore = _;
        Eigen::Quaternionf q{ 1.0f, 2.0f, 3.0f, 4.0f };
        benchmark::DoNotOptimize(q);
        benchmark::ClobberMemory();
        benchmark::DoNotOptimize(q.norm());
    }
}

BENCHMARK(BM_QuaternionF32NormEigen);

static void BM_QuaternionF64NormEigen(benchmark::State & state)
{
    for (const auto & _ : state) {
        std::ignore = _;
        Eigen::Quaterniond q{ 1.0, 2.0, 3.0, 4.0 };
        benchmark::DoNotOptimize(q);
        benchmark::ClobberMemory();
        benchmark::DoNotOptimize(q.norm());
    }
}

BENCHMARK(BM_QuaternionF64NormEigen);

template <typename T>
class Quaternion
{
public:
    T norm() const { return std::sqrt(w * w + x * x + y * y + z * z); }

    T w, x, y, z;
};

static void BM_QuaternionF32NormManualImpl(benchmark::State & state)
{
    for (const auto & _ : state) {
        std::ignore = _;
        Quaternion<float> q{ 1.0f, 2.0f, 3.0f, 4.0f };
        benchmark::DoNotOptimize(q);
        benchmark::ClobberMemory();
        benchmark::DoNotOptimize(q.norm());
    }
}

BENCHMARK(BM_QuaternionF32NormManualImpl);

static void BM_QuaternionF64NormManualImpl(benchmark::State & state)
{
    for (const auto & _ : state) {
        std::ignore = _;
        Quaternion<double> q{ 1.0f, 2.0f, 3.0f, 4.0f };
        benchmark::DoNotOptimize(q);
        benchmark::ClobberMemory();
        benchmark::DoNotOptimize(q.norm());
    }
}

BENCHMARK(BM_QuaternionF64NormManualImpl);

static void BM_QuaternionF32NormManualHypotImpl(benchmark::State & state)
{
    for (const auto & _ : state) {
        std::ignore = _;
        Quaternion<float> q{ 1.0f, 2.0f, 3.0f, 4.0f };
        benchmark::DoNotOptimize(q);
        benchmark::ClobberMemory();
        benchmark::DoNotOptimize(
            std::hypot(std::hypot(q.w, q.x), std::hypot(q.y, q.z)));
    }
}

BENCHMARK(BM_QuaternionF32NormManualHypotImpl);

static void BM_QuaternionF64NormManualHypotImpl(benchmark::State & state)
{
    for (const auto & _ : state) {
        std::ignore = _;
        Quaternion<double> q{ 1.0f, 2.0f, 3.0f, 4.0f };
        benchmark::DoNotOptimize(q);
        benchmark::ClobberMemory();
        benchmark::DoNotOptimize(
            std::hypot(std::hypot(q.w, q.x), std::hypot(q.y, q.z)));
    }
}

// NOLINTEND(*-magic-numbers)

BENCHMARK(BM_QuaternionF64NormManualHypotImpl);

BENCHMARK_MAIN();
