#include "litepgo/fem/local_kernel.hpp"

#include <array>
#include <gtest/gtest.h>

namespace {
struct Cache {
    struct Sample {
        double point;
        double volume_weight;
    };
    static constexpr int num_quadrature_points = 2;
    std::array<Sample, 2> samples{{{0.2, 0.4}, {0.8, 0.6}}};
    const Sample& sample(int q) const {
        return samples.at(static_cast<std::size_t>(q));
    }
};
// Pure local energy: the integration layer never sees fields or constitutive inputs.
struct ValueOnly {
    Eigen::Vector3d coefficients{0.2, -0.4, 0.6};
    double value(double point) const {
        return 0.5 * (1 + point) * coefficients.squaredNorm();
    }
};
struct FirstOnly : ValueOnly {
    static constexpr int num_dofs = 3;
    Eigen::Vector3d gradient(double point) const {
        return (1 + point) * coefficients;
    }
};
struct SecondOnly : ValueOnly {
    static constexpr int num_dofs = 3;
    Eigen::Matrix3d hessian(double point) const {
        return (1 + point) * Eigen::Matrix3d::Identity();
    }
};
struct FullEnergy : FirstOnly {
    Eigen::Matrix3d hessian(double point) const {
        return (1 + point) * Eigen::Matrix3d::Identity();
    }
};
struct WrongGradient : FullEnergy {
    Eigen::Vector2d gradient(double) const;
};
struct WrongHessian : FullEnergy {
    Eigen::Matrix2d hessian(double) const;
};
struct WrongPoint {
    double value(const Eigen::Vector3d&) const;
};
struct MissingValue {
    [[maybe_unused]] static constexpr int num_dofs = 3;
    Eigen::Matrix3d hessian(double) const;
};
struct WrongValue {
    float value(double) const;
};
struct ZeroDofs : FullEnergy {
    static constexpr int num_dofs = 0;
};

template <class Energy>
concept HasValue = requires(const litepgo::LocalKernel<Cache>& k, const Energy& e) { k.value(e); };
template <class Energy>
concept HasGradient = requires(const litepgo::LocalKernel<Cache>& k, const Energy& e) { k.gradient(e); };
template <class Energy>
concept HasHessian = requires(const litepgo::LocalKernel<Cache>& k, const Energy& e) { k.hessian(e); };
static_assert(HasValue<ValueOnly> && !HasGradient<ValueOnly> && !HasHessian<ValueOnly>);
static_assert(HasGradient<FirstOnly> && !HasHessian<FirstOnly>);
static_assert(!HasGradient<SecondOnly> && HasHessian<SecondOnly>);
static_assert(HasGradient<FullEnergy> && HasHessian<FullEnergy>);
static_assert(!HasGradient<WrongGradient> && HasHessian<WrongGradient>);
static_assert(HasGradient<WrongHessian> && !HasHessian<WrongHessian>);
static_assert(!HasValue<WrongPoint> && !HasValue<WrongValue> && !HasHessian<MissingValue>);
static_assert(!HasGradient<ZeroDofs> && !HasHessian<ZeroDofs>);

TEST(LocalKernel, IntegratesIndependentCapabilitiesAndOwnsCache) {
    Cache cache;
    const litepgo::LocalKernel kernel(cache);
    FullEnergy energy;
    const double factor = 0.4 * 1.2 + 0.6 * 1.8;
    cache.samples[0].volume_weight = 100;
    EXPECT_NEAR(kernel.value(energy), factor * 0.5 * energy.coefficients.squaredNorm(), 1e-13);
    EXPECT_TRUE(kernel.gradient(energy).isApprox(factor * energy.coefficients, 1e-13));
    EXPECT_TRUE(kernel.hessian(energy).isApprox(factor * Eigen::Matrix3d::Identity(), 1e-13));
    EXPECT_DOUBLE_EQ(kernel.value(ValueOnly{}), kernel.value(energy));
    EXPECT_EQ(kernel.gradient(FirstOnly{}), kernel.gradient(energy));
    EXPECT_EQ(kernel.hessian(SecondOnly{}), kernel.hessian(energy));
    constexpr double h = 1e-4;
    for (int j = 0; j < 3; ++j) {
        auto plus = energy, minus = energy;
        plus.coefficients[j] += h;
        minus.coefficients[j] -= h;
        EXPECT_NEAR(kernel.gradient(energy)[j], (kernel.value(plus) - kernel.value(minus)) / (2 * h), 1e-10);
        EXPECT_TRUE(
            kernel.hessian(energy).col(j).isApprox((kernel.gradient(plus) - kernel.gradient(minus)) / (2 * h), 1e-10));
    }
    energy.coefficients *= 2;
    EXPECT_TRUE(kernel.gradient(energy).isApprox(factor * energy.coefficients, 1e-13));
}
} // namespace
