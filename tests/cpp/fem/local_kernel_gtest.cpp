#include "litepgo/fem/hex_q1_basis.hpp"
#include "litepgo/fem/local_kernel.hpp"
#include <gtest/gtest.h>
#include <type_traits>

namespace {
using namespace litepgo;
struct Rule {
    static constexpr int num_points = 2;
    static const Eigen::Matrix<double, 2, 3>& points() {
        static const Eigen::Matrix<double, 2, 3> p =
            (Eigen::Matrix<double, 2, 3>() << 0.2, 0.3, 0.4, 0.8, 0.3, 0.4).finished();
        return p;
    }
    static const Eigen::Vector2d& weights() {
        static const Eigen::Vector2d w(0.4, 0.6);
        return w;
    }
};
VolumeQuadratureCache cache(double scale = 1) {
    Eigen::Matrix<double, 8, 3> X;
    for (int i = 0; i < 8; ++i)
        X.row(i) = scale * HexQ1Basis::reference_node(i);
    return VolumeQuadratureCache(VolumeReferenceGeometry<HexQ1Basis>(X), Rule{});
}
struct ValueOnly : LocalValue {
    Eigen::Vector3d coefficients{0.2, -0.4, 0.6};
    double value(const VolumePointData& p) const override {
        return 0.5 * (1 + p.position.x()) * coefficients.squaredNorm();
    }
};
struct FirstOnly : LocalGradient {
    Eigen::Vector3d coefficients{0.2, -0.4, 0.6};
    int num_dofs() const override {
        return 3;
    }
    Eigen::VectorXd gradient(const VolumePointData& p) const override {
        return (1 + p.position.x()) * coefficients;
    }
};
struct SecondOnly : LocalHessian {
    int num_dofs() const override {
        return 3;
    }
    Eigen::MatrixXd hessian(const VolumePointData& p) const override {
        return (1 + p.position.x()) * Eigen::Matrix3d::Identity();
    }
};
struct FullEnergy : ValueOnly, LocalGradient, LocalHessian {
    int num_dofs() const override {
        return 3;
    }
    Eigen::VectorXd gradient(const VolumePointData& p) const override {
        return (1 + p.position.x()) * coefficients;
    }
    Eigen::MatrixXd hessian(const VolumePointData& p) const override {
        return (1 + p.position.x()) * Eigen::Matrix3d::Identity();
    }
};
static_assert(!std::is_base_of_v<LocalHessian, FirstOnly>);
static_assert(!std::is_base_of_v<LocalGradient, SecondOnly>);
static_assert(!std::is_base_of_v<LocalValue, SecondOnly>);
TEST(LocalKernel, IntegratesIndependentCapabilitiesAndOwnsCache) {
    auto original = cache();
    const LocalKernel kernel(original);
    original = cache(2);
    FullEnergy energy;
    const double factor = 0.4 * 1.2 + 0.6 * 1.8;
    EXPECT_NEAR(kernel.value(energy), factor * 0.5 * energy.coefficients.squaredNorm(), 1e-13);
    EXPECT_TRUE(kernel.gradient(energy).isApprox(factor * energy.coefficients, 1e-13));
    EXPECT_TRUE(kernel.hessian(energy).isApprox(factor * Eigen::Matrix3d::Identity(), 1e-13));
    EXPECT_DOUBLE_EQ(kernel.value(ValueOnly{}), kernel.value(energy));
    EXPECT_EQ(kernel.gradient(FirstOnly{}), kernel.gradient(energy));
    EXPECT_EQ(kernel.hessian(SecondOnly{}), kernel.hessian(energy));
    for (int j = 0; j < 3; ++j) {
        auto plus = energy, minus = energy;
        plus.coefficients[j] += 1e-4;
        minus.coefficients[j] -= 1e-4;
        EXPECT_NEAR(kernel.gradient(energy)[j], (kernel.value(plus) - kernel.value(minus)) / 2e-4, 1e-10);
        EXPECT_TRUE(
            kernel.hessian(energy).col(j).isApprox((kernel.gradient(plus) - kernel.gradient(minus)) / 2e-4, 1e-10));
    }
    const LocalKernel temporary(cache());
    EXPECT_EQ(temporary.hessian(energy), kernel.hessian(energy));
}
struct BadGradient : LocalGradient {
    int n = 3;
    int num_dofs() const override {
        return n;
    }
    Eigen::VectorXd gradient(const VolumePointData&) const override {
        return Eigen::Vector2d::Zero();
    }
};
struct BadHessian : LocalHessian {
    int num_dofs() const override {
        return 3;
    }
    Eigen::MatrixXd hessian(const VolumePointData&) const override {
        return Eigen::MatrixXd::Zero(3, 2);
    }
};
TEST(LocalKernel, RejectsInvalidRuntimeDerivativeSizes) {
    const LocalKernel kernel(cache());
    BadGradient bad;
    EXPECT_THROW(kernel.gradient(bad), std::invalid_argument);
    bad.n = 0;
    EXPECT_THROW(kernel.gradient(bad), std::invalid_argument);
    EXPECT_THROW(kernel.hessian(BadHessian{}), std::invalid_argument);
}
} // namespace
