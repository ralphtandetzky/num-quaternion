#include <benchmark/benchmark.h>

#include <boost/qvm/quat.hpp>
#include <boost/qvm/quat_operations.hpp>

#include <Eigen/Geometry>

#include <cmath>

static void BM_QuaternionF32FromEulerAnglesBoostQVM(benchmark::State &state) {
  for (auto _ : state) {
    float angles[3] = {1.0f, 2.0f, 3.0f};
    benchmark::DoNotOptimize(angles);
    benchmark::ClobberMemory();
    benchmark::DoNotOptimize(boost::qvm::quat<float>(
        boost::qvm::rotx_quat(angles[0]) *
        boost::qvm::quat<float>(boost::qvm::roty_quat(angles[1])) *
        boost::qvm::rotz_quat(angles[2])));
  }
}

BENCHMARK(BM_QuaternionF32FromEulerAnglesBoostQVM);

static void BM_QuaternionF64FromEulerAnglesBoostQVM(benchmark::State &state) {
  for (auto _ : state) {
    double angles[3] = {1.0, 2.0, 3.0};
    benchmark::DoNotOptimize(angles);
    benchmark::ClobberMemory();
    benchmark::DoNotOptimize(boost::qvm::quat<double>(
        boost::qvm::rotx_quat(angles[0]) *
        boost::qvm::quat<double>(boost::qvm::roty_quat(angles[1])) *
        boost::qvm::rotz_quat(angles[2])));
  }
}

BENCHMARK(BM_QuaternionF64FromEulerAnglesBoostQVM);

static void BM_QuaternionF32FromEulerAnglesEigen(benchmark::State &state) {
  for (auto _ : state) {
    float angles[3] = {1.0f, 2.0f, 3.0f};
    benchmark::DoNotOptimize(angles);
    benchmark::ClobberMemory();
    benchmark::DoNotOptimize(Eigen::Quaternionf(
        Eigen::AngleAxisf(angles[0], Eigen::Vector3f::UnitX()) *
        Eigen::AngleAxisf(angles[1], Eigen::Vector3f::UnitY()) *
        Eigen::AngleAxisf(angles[2], Eigen::Vector3f::UnitZ())));
  }
}

BENCHMARK(BM_QuaternionF32FromEulerAnglesEigen);

static void BM_QuaternionF64FromEulerAnglesEigen(benchmark::State &state) {
  for (auto _ : state) {
    double angles[3] = {1.0, 2.0, 3.0};
    benchmark::DoNotOptimize(angles);
    benchmark::ClobberMemory();
    benchmark::DoNotOptimize(Eigen::Quaterniond(
        Eigen::AngleAxisd(angles[0], Eigen::Vector3d::UnitX()) *
        Eigen::AngleAxisd(angles[1], Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(angles[2], Eigen::Vector3d::UnitZ())));
  }
}

BENCHMARK(BM_QuaternionF64FromEulerAnglesEigen);

BENCHMARK_MAIN();
