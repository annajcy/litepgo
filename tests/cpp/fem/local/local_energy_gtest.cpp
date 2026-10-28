#include "litepgo/energy/energy.hpp"
#include "litepgo/fem/local/local_energy.hpp"

#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>

namespace {
using namespace litepgo;

struct ValueOnly : LocalEnergy3D {
    Eigen::Vector3d coefficients{0.2, -0.4, 0.6};
    int num_dofs() const override {
        return 3;
    }
    double eval_value(const Eigen::Vector3d& xi) const override {
        return 0.5 * (1 + xi.x()) * coefficients.squaredNorm();
    }
};
struct FirstOnly : ValueOnly {
    bool has_gradient() const noexcept override {
        return true;
    }
    Eigen::VectorXd eval_gradient(const Eigen::Vector3d& xi) const override {
        return (1 + xi.x()) * coefficients;
    }
};
struct SecondOnly : ValueOnly {
    bool has_hessian() const noexcept override {
        return true;
    }
    Eigen::MatrixXd eval_hessian(const Eigen::Vector3d& xi) const override {
        return (1 + xi.x()) * Eigen::Matrix3d::Identity();
    }
};
struct FullEnergy : FirstOnly {
    bool has_hessian() const noexcept override {
        return true;
    }
    Eigen::MatrixXd eval_hessian(const Eigen::Vector3d& xi) const override {
        return (1 + xi.x()) * Eigen::Matrix3d::Identity();
    }
};

TEST(LocalEnergy3D, OverflowingWeightedDensityAndDerivativeAccumulationsThrowDomainError) {
    FullEnergy energy;
    const Eigen::Vector3d xi = Eigen::Vector3d::Zero();
    const double max = std::numeric_limits<double>::max();
    energy.coefficients.setConstant(2);
    EXPECT_THROW(energy.accumulate(xi, max, nullptr, nullptr), std::domain_error);

    energy.coefficients << 1, 0, 0;
    Eigen::VectorXd g = Eigen::Vector3d::Constant(max);
    Eigen::MatrixXd H = Eigen::Matrix3d::Constant(max);
    EXPECT_THROW(energy.accumulate(xi, max, &g, nullptr), std::domain_error);
    EXPECT_THROW(energy.accumulate(xi, max, nullptr, &H), std::domain_error);
}

TEST(LocalEnergy3D, DerivativeCapabilitiesAreIndependent) {
    ValueOnly value;
    FirstOnly first;
    SecondOnly second;
    FullEnergy full;
    EXPECT_FALSE(value.has_gradient());
    EXPECT_FALSE(value.has_hessian());
    EXPECT_TRUE(first.has_gradient());
    EXPECT_FALSE(first.has_hessian());
    EXPECT_FALSE(second.has_gradient());
    EXPECT_TRUE(second.has_hessian());
    EXPECT_TRUE(full.has_gradient());
    EXPECT_TRUE(full.has_hessian());
    const Eigen::Vector3d point = Eigen::Vector3d::Zero();
    EXPECT_THROW(value.eval_gradient(point), std::logic_error);
    EXPECT_THROW(value.eval_hessian(point), std::logic_error);
}

TEST(LocalEnergy3D, DefaultAccumulationAddsWeightedValuesAndChecksRequests) {
    const FullEnergy full;
    const Eigen::Vector3d point(0.2, 0.3, 0.4);
    Eigen::VectorXd g = Eigen::Vector3d::Constant(3);
    Eigen::MatrixXd h = Eigen::Matrix3d::Constant(5);
    const double weight = -0.7;
    EXPECT_DOUBLE_EQ(full.accumulate(point, weight, &g, &h), weight * full.eval_value(point));
    EXPECT_TRUE(g.isApprox(Eigen::Vector3d::Constant(3) + weight * full.eval_gradient(point), 1e-14));
    EXPECT_TRUE(h.isApprox(Eigen::Matrix3d::Constant(5) + weight * full.eval_hessian(point), 1e-14));
    EXPECT_DOUBLE_EQ(full.accumulate(point, weight, nullptr, nullptr), weight * full.eval_value(point));
    const auto before_g = g;
    const auto before_h = h;
    EXPECT_THROW(full.accumulate(point, std::numeric_limits<double>::infinity(), &g, &h), std::invalid_argument);
    EXPECT_THROW(full.accumulate(Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN()), 1, &g, &h),
                 std::invalid_argument);
    EXPECT_THROW(ValueOnly{}.accumulate(point, 1, &g, nullptr), std::logic_error);
    EXPECT_THROW(FirstOnly{}.accumulate(point, 1, nullptr, &h), std::logic_error);
    EXPECT_EQ(g, before_g);
    EXPECT_EQ(h, before_h);
    Eigen::VectorXd wrong_g = Eigen::Vector2d::Zero();
    Eigen::MatrixXd wrong_h = Eigen::MatrixXd::Zero(3, 2);
    EXPECT_THROW(full.accumulate(point, 1, &wrong_g, nullptr), std::invalid_argument);
    EXPECT_THROW(full.accumulate(point, 1, nullptr, &wrong_h), std::invalid_argument);
}
} // namespace
