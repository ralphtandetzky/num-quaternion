#include <benchmark/benchmark.h>

#include <boost/qvm/quat.hpp>
#include <boost/qvm/quat_operations.hpp>

#include <Eigen/Dense>

#include <cmath>

static void BM_QuaternionF32NormBoostQVM(benchmark::State &state) {
  for (auto _ : state) {
    boost::qvm::quat<float> q{1.0f, 2.0f, 3.0f, 4.0f};
    benchmark::DoNotOptimize(q);
    benchmark::ClobberMemory();
    benchmark::DoNotOptimize(boost::qvm::mag(q));
  }
}

BENCHMARK(BM_QuaternionF32NormBoostQVM);

static void BM_QuaternionF64NormBoostQVM(benchmark::State &state) {
  for (auto _ : state) {
    boost::qvm::quat<double> q{1.0f, 2.0f, 3.0f, 4.0f};
    benchmark::DoNotOptimize(q);
    benchmark::ClobberMemory();
    benchmark::DoNotOptimize(boost::qvm::mag(q));
  }
}

BENCHMARK(BM_QuaternionF64NormBoostQVM);

static void BM_QuaternionF32NormEigen(benchmark::State &state) {
  for (auto _ : state) {
    Eigen::Quaternionf q{1.0f, 2.0f, 3.0f, 4.0f};
    benchmark::DoNotOptimize(q);
    benchmark::ClobberMemory();
    benchmark::DoNotOptimize(q.norm());
  }
}

BENCHMARK(BM_QuaternionF32NormEigen);

static void BM_QuaternionF64NormEigen(benchmark::State &state) {
  for (auto _ : state) {
    Eigen::Quaterniond q{1.0, 2.0, 3.0, 4.0};
    benchmark::DoNotOptimize(q);
    benchmark::ClobberMemory();
    benchmark::DoNotOptimize(q.norm());
  }
}

BENCHMARK(BM_QuaternionF64NormEigen);

template <typename T> class Quaternion {
public:
  Quaternion(T w, T x, T y, T z) : w(w), x(x), y(y), z(z) {}

  T norm() const { return std::sqrt(w * w + x * x + y * y + z * z); }

private:
  T w, x, y, z;
};

static void BM_QuaternionF32NormManualImpl(benchmark::State &state) {
  for (auto _ : state) {
    Quaternion<float> q{1.0f, 2.0f, 3.0f, 4.0f};
    benchmark::DoNotOptimize(q);
    benchmark::ClobberMemory();
    benchmark::DoNotOptimize(q.norm());
  }
}

BENCHMARK(BM_QuaternionF32NormManualImpl);

static void BM_QuaternionF64NormManualImpl(benchmark::State &state) {
  for (auto _ : state) {
    Quaternion<double> q{1.0f, 2.0f, 3.0f, 4.0f};
    benchmark::DoNotOptimize(q);
    benchmark::ClobberMemory();
    benchmark::DoNotOptimize(q.norm());
  }
}

BENCHMARK(BM_QuaternionF64NormManualImpl);

BENCHMARK_MAIN();
