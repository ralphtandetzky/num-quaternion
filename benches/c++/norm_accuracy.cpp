#include <Eigen/Geometry>
#include <boost/qvm/quat.hpp>
#include <boost/qvm/quat_operations.hpp>
#include <cfloat>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <ranges>
#include <string>

const size_t NUM_SAMPLES = 10000000;

float norm_boost_qvm(float w, float x, float y, float z)
{
    boost::qvm::quat<float> q = { w, x, y, z };
    return boost::qvm::mag(q);
}

float norm_manual(float w, float x, float y, float z)
{
    return std::hypot(std::hypot(w, x), std::hypot(y, z));
}

float norm_manual_fast(float w, float x, float y, float z)
{
    return std::sqrt(w * w + x * x + y * y + z * z);
}

float norm_eigen(float w, float x, float y, float z)
{
    Eigen::Quaternionf q(w, x, y, z);
    return q.norm();
}

using NormFunc = float (*)(float, float, float, float);

size_t utf8_length(const std::string_view str)
{
    return std::ranges::count_if(
        str, [](const char c) { return (c & 0xC0) != 0x80; });
}

int main()
{
    constexpr std::array<std::pair<NormFunc, std::string_view>, 4>
        norm_funcs = { { { norm_boost_qvm, "boost::qvm::mag" },
                         { norm_manual, "hypot implementation" },
                         { norm_manual_fast, "sqrt(a² + b² + c² + d²)" },
                         { norm_eigen, "Eigen::Quaternionf::norm" } } };

    const size_t func_space = std::ranges::max(
        std::ranges::views::transform(norm_funcs, [](const auto & pair) {
            return utf8_length(pair.second.data());
        }));

    size_t col_width = 13;

    std::cout << "Benchmarking the relative accuracy of quaternion norm "
                 "implementations for different scales of the\n";
    std::cout << "input quaternion.\n\n";
    std::cout << std::setw(func_space) << "Implementation \\ Scale"
              << " | " << std::setw(col_width) << "1.0"
              << " | " << std::setw(col_width) << "sqrt(MIN_POS)"
              << " | " << std::setw(col_width) << "MIN_POS"
              << " | " << std::setw(col_width) << "MAX / 2"
              << "\n";
    std::cout << std::string(func_space, '=')
              << "=|=" << std::string(col_width, '=')
              << "=|=" << std::string(col_width, '=')
              << "=|=" << std::string(col_width, '=')
              << "=|=" << std::string(col_width, '=') << "\n";

    const float scales[] = {
        1.0f, std::sqrt(FLT_MIN), FLT_MIN, FLT_MAX / 2.0f
    };

    for (const auto & [norm_func, func_name] : norm_funcs) {
        std::cout << std::setw(func_space + func_name.length() -
                               utf8_length(func_name))
                  << func_name;
        for (const float scale : scales) {
            std::mt19937 rng(0x7F0829AE4D31C6B5);
            std::uniform_real_distribution<float> dist(-scale, scale);
            double sum_sqr_error = 0.0;
            for (size_t j = 0; j < NUM_SAMPLES; ++j) {
                const float w = dist(rng);
                const float x = dist(rng);
                const float y = dist(rng);
                const float z = dist(rng);
                const float norm_f32 = norm_func(w, x, y, z);
                const double norm_f64 = std::sqrt(
                    static_cast<double>(w) * w + static_cast<double>(x) * x +
                    static_cast<double>(y) * y + static_cast<double>(z) * z);
                sum_sqr_error += std::pow(norm_f32 / norm_f64 - 1.0, 2);
            }
            const double mean_sqr_error = sum_sqr_error / NUM_SAMPLES;
            const double rms_error = std::sqrt(mean_sqr_error);
            const double rms_error_in_eps =
                rms_error / std::numeric_limits<float>::epsilon();
            using namespace std::literals::string_view_literals;
            const auto color_code = (rms_error_in_eps < 0.3)   ? "92" sv
                                    : (rms_error_in_eps < 1.0) ? "93" sv
                                                               : "91" sv;
            std::cout << " | \x1b[" << color_code << "m" << std::setw(col_width)
                      << rms_error_in_eps << "\x1b[0m" << std::flush;
        }
        std::cout << "\n";
    }

    std::cout << "\n\nThe columns of the table determine the scale of the "
                 "input quaternion.\n";
    std::cout << "The rows of the table determine the implementation of the "
                 "quaternion norm.\n";
    std::cout << "The values in the table are the relative RMS error of the "
                 "quaternion norm.\n";
    std::cout << "\nThe column `1.0` is for quaternions with all components "
                 "uniformly sampled from the range [-1.0, 1.0].\n";
    std::cout << "The column `sqrt(MIN_POS)` is for quaternions with all "
                 "components in the range [sqrt(MIN_POS), sqrt(MIN_POS)],\n";
    std::cout << "where `MIN_POS` is the minimal positive normal 32-bit "
                 "IEEE-754 floating point value. Similarly for `MIN_POS`\n";
    std::cout << "and `MAX / 2`, where `MAX` is the maximal finite `f32` "
                 "value.\n";

    return 0;
}
