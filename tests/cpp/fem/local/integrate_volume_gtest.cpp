#include "litepgo/fem/basis/hex_q1_basis.hpp"
#include "litepgo/fem/local/integrate_volume.hpp"
#include "litepgo/fem/quadrature/gauss_legendre_3d.hpp"
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>

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
Interpolation<HexQ1Basis> geometry(double scale = 1) {
    Eigen::Matrix<double, 8, 3> X;
    for (int i = 0; i < 8; ++i)
        X.row(i) = scale * HexQ1Basis::reference_node(i);
    return Interpolation<HexQ1Basis>(X);
}
struct ValueOnly : LocalEnergy {
    int num_dofs() const override {
        return 3;
    }
    Eigen::Vector3d coefficients{0.2, -0.4, 0.6};
    double value(const RestPointData& p) const override {
        return 0.5 * (1 + p.position.x()) * coefficients.squaredNorm();
    }
};
struct FirstOnly : ValueOnly {
    bool has_gradient() const noexcept override {
        return true;
    }
    Eigen::VectorXd gradient(const RestPointData& p) const override {
        return (1 + p.position.x()) * coefficients;
    }
};
struct SecondOnly : ValueOnly {
    bool has_hessian() const noexcept override {
        return true;
    }
    Eigen::MatrixXd hessian(const RestPointData& p) const override {
        return (1 + p.position.x()) * Eigen::Matrix3d::Identity();
    }
};
struct FullEnergy : FirstOnly {
    mutable int value_calls = 0, gradient_calls = 0, hessian_calls = 0, dof_calls = 0;
    bool has_hessian() const noexcept override {
        return true;
    }
    int num_dofs() const override {
        ++dof_calls;
        return 3;
    }
    double value(const RestPointData& p) const override {
        ++value_calls;
        return ValueOnly::value(p);
    }
    Eigen::VectorXd gradient(const RestPointData& p) const override {
        ++gradient_calls;
        return FirstOnly::gradient(p);
    }
    Eigen::MatrixXd hessian(const RestPointData& p) const override {
        ++hessian_calls;
        return (1 + p.position.x()) * Eigen::Matrix3d::Identity();
    }
};
TEST(LocalEnergy, DerivativeCapabilitiesAreIndependent) {
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
    const RestPointData point{};
    EXPECT_THROW(value.gradient(point), std::logic_error);
    EXPECT_THROW(value.hessian(point), std::logic_error);
}

TEST(LocalEnergy, DefaultAccumulationAddsWeightedValuesAndChecksRequests) {
    FullEnergy full;
    const auto point = rest_point(geometry(), {0.2, 0.3, 0.4});
    Eigen::VectorXd g = Eigen::Vector3d::Constant(3);
    Eigen::MatrixXd h = Eigen::Matrix3d::Constant(5);
    const double weight = -0.7;
    EXPECT_DOUBLE_EQ(full.accumulate(point, weight, &g, &h), weight * full.value(point));
    EXPECT_TRUE(g.isApprox(Eigen::Vector3d::Constant(3) + weight * full.gradient(point), 1e-14));
    EXPECT_TRUE(h.isApprox(Eigen::Matrix3d::Constant(5) + weight * full.hessian(point), 1e-14));
    const auto before_g = g;
    const auto before_h = h;
    EXPECT_THROW(full.accumulate(point, std::numeric_limits<double>::infinity(), &g, &h), std::invalid_argument);
    EXPECT_THROW(ValueOnly{}.accumulate(point, 1, &g, nullptr), std::logic_error);
    EXPECT_THROW(FirstOnly{}.accumulate(point, 1, nullptr, &h), std::logic_error);
    EXPECT_EQ(g, before_g);
    EXPECT_EQ(h, before_h);
}

// Even if derivative methods exist, queries govern whether integration is allowed.
struct DisabledDerivatives : FullEnergy {
    bool has_gradient() const noexcept override {
        return false;
    }
    bool has_hessian() const noexcept override {
        return false;
    }
};
TEST(IntegrateVolume, ChecksCapabilitiesBeforeChangingOutputsOrEvaluatingPoints) {
    DisabledDerivatives energy;
    const auto reference = geometry();
    Eigen::VectorXd g = Eigen::VectorXd::Constant(1, 7);
    Eigen::MatrixXd h = Eigen::MatrixXd::Constant(1, 1, 9);
    EXPECT_THROW(integrate_volume<Rule>(reference, energy, &g), std::logic_error);
    EXPECT_THROW(integrate_volume<Rule>(geometry(0), energy, &g), std::logic_error);
    EXPECT_THROW(integrate_volume<Rule>(reference, energy, nullptr, &h), std::logic_error);
    FirstOnly first;
    SecondOnly second;
    EXPECT_THROW(integrate_volume<Rule>(reference, first, &g, &h), std::logic_error);
    EXPECT_THROW(integrate_volume<Rule>(reference, second, &g, &h), std::logic_error);
    EXPECT_EQ(g, Eigen::VectorXd::Constant(1, 7));
    EXPECT_EQ(h, Eigen::MatrixXd::Constant(1, 1, 9));
    EXPECT_EQ(energy.dof_calls, 0);
    EXPECT_EQ(energy.value_calls, 0);
    EXPECT_EQ(energy.gradient_calls, 0);
    EXPECT_EQ(energy.hessian_calls, 0);
    EXPECT_DOUBLE_EQ(integrate_volume<Rule>(reference, energy), integrate_volume<Rule>(reference, ValueOnly{}));
    EXPECT_EQ(energy.dof_calls, 0);
    EXPECT_EQ(energy.gradient_calls, 0);
    EXPECT_EQ(energy.hessian_calls, 0);
}

TEST(IntegrateVolume, IntegratesIndependentOutputsInOnePassAndOverwritesBuffers) {
    const auto reference = geometry();
    FullEnergy energy;
    Eigen::VectorXd g = Eigen::VectorXd::Ones(1);
    Eigen::MatrixXd h = Eigen::MatrixXd::Ones(1, 2);
    const double value = integrate_volume<Rule>(reference, energy, &g, &h);
    const double factor = 0.4 * 1.2 + 0.6 * 1.8;
    EXPECT_NEAR(value, factor * 0.5 * energy.coefficients.squaredNorm(), 1e-13);
    EXPECT_TRUE(g.isApprox(factor * energy.coefficients, 1e-13));
    EXPECT_TRUE(h.isApprox(factor * Eigen::Matrix3d::Identity(), 1e-13));
    EXPECT_EQ(energy.dof_calls, 1);
    EXPECT_EQ(energy.value_calls, Rule::num_points);
    EXPECT_EQ(energy.gradient_calls, Rule::num_points);
    EXPECT_EQ(energy.hessian_calls, Rule::num_points);
    EXPECT_DOUBLE_EQ(integrate_volume<Rule>(reference, energy, &g, &h), value);
    EXPECT_TRUE(g.isApprox(factor * energy.coefficients, 1e-13));
    EXPECT_TRUE(h.isApprox(factor * Eigen::Matrix3d::Identity(), 1e-13));
    Eigen::VectorXd first_g;
    Eigen::MatrixXd second_h;
    EXPECT_DOUBLE_EQ(integrate_volume<Rule>(reference, FirstOnly{}, &first_g), value);
    EXPECT_DOUBLE_EQ(integrate_volume<Rule>(reference, SecondOnly{}, nullptr, &second_h), value);
    EXPECT_EQ(first_g, g);
    EXPECT_EQ(second_h, h);
    EXPECT_DOUBLE_EQ(integrate_volume<Rule>(geometry(), ValueOnly{}), value);
    EXPECT_NEAR(integrate_volume<Rule>(geometry(2), ValueOnly{}),
                8 * (0.4 * 1.4 + 0.6 * 2.6) * 0.5 * energy.coefficients.squaredNorm(), 1e-13);
    for (int j = 0; j < 3; ++j) {
        auto plus = energy, minus = energy;
        plus.coefficients[j] += 1e-4;
        minus.coefficients[j] -= 1e-4;
        Eigen::VectorXd plus_g, minus_g;
        const double plus_value = integrate_volume<Rule>(reference, plus, &plus_g);
        const double minus_value = integrate_volume<Rule>(reference, minus, &minus_g);
        EXPECT_NEAR(g[j], (plus_value - minus_value) / 2e-4, 1e-10);
        EXPECT_TRUE(h.col(j).isApprox((plus_g - minus_g) / 2e-4, 1e-10));
    }
}
struct BadGradient : FirstOnly {
    int n = 3;
    int num_dofs() const override {
        return n;
    }
    Eigen::VectorXd gradient(const RestPointData&) const override {
        return Eigen::Vector2d::Zero();
    }
};
struct BadHessian : SecondOnly {
    Eigen::MatrixXd hessian(const RestPointData&) const override {
        return Eigen::MatrixXd::Zero(3, 2);
    }
};
TEST(IntegrateVolume, RejectsInvalidRuntimeDerivativeSizes) {
    const auto reference = geometry();
    Eigen::VectorXd g;
    Eigen::MatrixXd h;
    BadGradient bad;
    EXPECT_THROW(integrate_volume<Rule>(reference, bad, &g), std::invalid_argument);
    bad.n = 0;
    EXPECT_THROW(integrate_volume<Rule>(reference, bad, &g), std::invalid_argument);
    EXPECT_NO_THROW(integrate_volume<Rule>(reference, bad));
    EXPECT_THROW(integrate_volume<Rule>(reference, BadHessian{}, nullptr, &h), std::invalid_argument);
}

struct ResizingAccumulator : FullEnergy {
    mutable int calls = 0;
    double accumulate(const RestPointData&, double, Eigen::VectorXd* gradient,
                      Eigen::MatrixXd* hessian) const override {
        ++calls;
        if (gradient)
            gradient->resize(4);
        if (hessian)
            hessian->resize(4, 4);
        return 0;
    }
};

TEST(IntegrateVolume, RejectsResizedAccumulatorOutputsAtTheFirstPoint) {
    const ResizingAccumulator energy;
    Eigen::VectorXd g;
    Eigen::MatrixXd h;
    EXPECT_THROW(integrate_volume<Rule>(geometry(), energy, &g), std::invalid_argument);
    EXPECT_EQ(energy.calls, 1);
    EXPECT_THROW(integrate_volume<Rule>(geometry(), energy, nullptr, &h), std::invalid_argument);
    EXPECT_EQ(energy.calls, 2);
}
} // namespace

namespace {
struct UnitDensity : LocalEnergy {
    mutable int calls = 0;
    int num_dofs() const override {
        return 0;
    }
    double value(const RestPointData&) const override {
        ++calls;
        return 1;
    }
};
struct NonFiniteWeights : Rule {
    static const Eigen::Vector2d& weights() {
        static const Eigen::Vector2d w(0.4, std::numeric_limits<double>::quiet_NaN());
        return w;
    }
};
struct OverflowingWeights : Rule {
    static const Eigen::Vector2d& weights() {
        static const Eigen::Vector2d w = Eigen::Vector2d::Constant(std::numeric_limits<double>::max());
        return w;
    }
};
struct NonFinitePoints : Rule {
    static const Eigen::Matrix<double, 2, 3>& points() {
        static const Eigen::Matrix<double, 2, 3> p =
            Eigen::Matrix<double, 2, 3>::Constant(std::numeric_limits<double>::infinity());
        return p;
    }
};
} // namespace

TEST(IntegrateVolume, IntegratesUnitTranslatedAndWarpedVolumes) {
    EXPECT_NEAR(integrate_volume<Rule>(geometry(), UnitDensity{}), 1, 1e-14);
    EXPECT_NEAR(integrate_volume<GaussLegendre3D<2>>(geometry(), UnitDensity{}), 1, 1e-14);
    Eigen::Matrix<double, 8, 3> X;
    for (int i = 0; i < 8; ++i) {
        const auto xi = HexQ1Basis::reference_node(i);
        X.row(i) = (Eigen::Vector3d(1, -2, 3) + Eigen::Vector3d(2, 3, 4).asDiagonal() * xi).transpose();
    }
    EXPECT_NEAR(integrate_volume<GaussLegendre3D<2>>(Interpolation<HexQ1Basis>(X), UnitDensity{}), 24, 1e-13);
    for (int i = 0; i < 8; ++i) {
        const auto xi = HexQ1Basis::reference_node(i);
        X.row(i) << xi.x() + 0.4 * xi.y() * xi.z(), xi.y() + 0.2 * xi.x() * xi.z(), xi.z();
    }
    EXPECT_NEAR(integrate_volume<GaussLegendre3D<2>>(Interpolation<HexQ1Basis>(X), UnitDensity{}), 1 - 0.08 / 3, 1e-14);
}

TEST(IntegrateVolume, RejectsInvalidGeometryAtTheEvaluatedPoint) {
    UnitDensity energy;
    EXPECT_THROW(integrate_volume<Rule>(geometry(0), energy), std::invalid_argument);
    EXPECT_THROW(integrate_volume<Rule>(geometry(-1), energy), std::invalid_argument);
    EXPECT_EQ(energy.calls, 0);
    Eigen::Matrix<double, 8, 3> X;
    for (int i = 0; i < 8; ++i) {
        const auto xi = HexQ1Basis::reference_node(i);
        X.row(i) << xi.x(), xi.y(), xi.z() * (1 - 1.5 * xi.x());
    }
    EXPECT_THROW(integrate_volume<Rule>(Interpolation<HexQ1Basis>(X), energy), std::invalid_argument);
    EXPECT_EQ(energy.calls, 1); // Only the first point has positive det(J).
}

TEST(IntegrateVolume, RejectsNonFinitePointsWeightsAndOverflowingVolumeWeights) {
    UnitDensity energy;
    EXPECT_THROW(integrate_volume<NonFinitePoints>(geometry(), energy), std::invalid_argument);
    EXPECT_THROW(integrate_volume<OverflowingWeights>(geometry(2), energy), std::invalid_argument);
    EXPECT_EQ(energy.calls, 0);
    EXPECT_THROW(integrate_volume<NonFiniteWeights>(geometry(), energy), std::invalid_argument);
    EXPECT_EQ(energy.calls, 1);
}
