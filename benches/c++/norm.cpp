#include <benchmark/benchmark.h>

#include <boost/qvm/quat.hpp>
#include <boost/qvm/quat_operations.hpp>

#include <cmath>

static void BM_QuaternionNormBoostQVM(benchmark::State &state) {
  for (auto _ : state) {
    boost::qvm::quat<float> q{1.0f, 2.0f, 3.0f, 4.0f};
    benchmark::DoNotOptimize(q);
    benchmark::ClobberMemory();
    benchmark::DoNotOptimize(boost::qvm::mag(q));
  }
}

BENCHMARK(BM_QuaternionNormBoostQVM);

template <typename T> class Quaternion {
public:
  Quaternion(T w, T x, T y, T z) : w(w), x(x), y(y), z(z) {}

  T norm() const { return std::sqrt(w * w + x * x + y * y + z * z); }

private:
  T w, x, y, z;
};

static void BM_QuaternionNormManualImpl(benchmark::State &state) {
  for (auto _ : state) {
    Quaternion<float> q{1.0f, 2.0f, 3.0f, 4.0f};
    benchmark::DoNotOptimize(q);
    benchmark::ClobberMemory();
    benchmark::DoNotOptimize(q.norm());
  }
}

BENCHMARK(BM_QuaternionNormManualImpl);

BENCHMARK_MAIN();
