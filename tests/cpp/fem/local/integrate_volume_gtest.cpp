#include "litepgo/fem/basis/3d/hex_q1_basis.hpp"
#include "litepgo/fem/field/interpolated_field.hpp"
#include "litepgo/fem/local/integrate_volume.hpp"
#include "litepgo/fem/quadrature/3d/gauss_legendre_3d.hpp"
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

#include <Eigen/LU>

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
InterpolatedField<HexQ1Basis> make_X_field(double scale = 1) {
    Eigen::Matrix<double, 8, 3> X;
    for (int i = 0; i < 8; ++i)
        X.row(i) = scale * HexQ1Basis::reference_node(i);
    return InterpolatedField<HexQ1Basis>(X);
}
struct ValueOnly : LocalEnergy {
    explicit ValueOnly(InterpolatedField<HexQ1Basis> coordinates = make_X_field()) : X_field(std::move(coordinates)) {}
    const InterpolatedField<HexQ1Basis> X_field;
    int num_dofs() const override {
        return 3;
    }
    Eigen::Vector3d coefficients{0.2, -0.4, 0.6};
    double value(const Eigen::Vector3d& p) const override {
        return 0.5 * (1 + X_field.value(p).x()) * coefficients.squaredNorm();
    }
};
struct FirstOnly : ValueOnly {
    using ValueOnly::ValueOnly;
    bool has_gradient() const noexcept override {
        return true;
    }
    Eigen::VectorXd gradient(const Eigen::Vector3d& p) const override {
        return (1 + X_field.value(p).x()) * coefficients;
    }
};
struct SecondOnly : ValueOnly {
    bool has_hessian() const noexcept override {
        return true;
    }
    Eigen::MatrixXd hessian(const Eigen::Vector3d& p) const override {
        return (1 + X_field.value(p).x()) * Eigen::Matrix3d::Identity();
    }
};
struct FullEnergy : FirstOnly {
    using FirstOnly::FirstOnly;
    mutable int value_calls = 0, gradient_calls = 0, hessian_calls = 0, dof_calls = 0;
    bool has_hessian() const noexcept override {
        return true;
    }
    int num_dofs() const override {
        ++dof_calls;
        return 3;
    }
    double value(const Eigen::Vector3d& p) const override {
        ++value_calls;
        return ValueOnly::value(p);
    }
    Eigen::VectorXd gradient(const Eigen::Vector3d& p) const override {
        ++gradient_calls;
        return FirstOnly::gradient(p);
    }
    Eigen::MatrixXd hessian(const Eigen::Vector3d& p) const override {
        ++hessian_calls;
        return (1 + X_field.value(p).x()) * Eigen::Matrix3d::Identity();
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
    const Eigen::Vector3d point = Eigen::Vector3d::Zero();
    EXPECT_THROW(value.gradient(point), std::logic_error);
    EXPECT_THROW(value.hessian(point), std::logic_error);
}

TEST(LocalEnergy, DefaultAccumulationAddsWeightedValuesAndChecksRequests) {
    FullEnergy full;
    const Eigen::Vector3d point(0.2, 0.3, 0.4);
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
    using FullEnergy::FullEnergy;
    bool has_gradient() const noexcept override {
        return false;
    }
    bool has_hessian() const noexcept override {
        return false;
    }
};
TEST(IntegrateVolume, ChecksCapabilitiesBeforeChangingOutputsOrEvaluatingPoints) {
    DisabledDerivatives energy;
    const DisabledDerivatives invalid_geometry(make_X_field(0));
    Eigen::VectorXd g = Eigen::VectorXd::Constant(1, 7);
    Eigen::MatrixXd h = Eigen::MatrixXd::Constant(1, 1, 9);
    EXPECT_THROW(integrate_volume<Rule>(energy.X_field, energy, &g), std::logic_error);
    EXPECT_THROW(integrate_volume<Rule>(invalid_geometry.X_field, invalid_geometry, &g), std::logic_error);
    EXPECT_THROW(integrate_volume<Rule>(energy.X_field, energy, nullptr, &h), std::logic_error);
    FirstOnly first;
    SecondOnly second;
    EXPECT_THROW(integrate_volume<Rule>(first.X_field, first, &g, &h), std::logic_error);
    EXPECT_THROW(integrate_volume<Rule>(second.X_field, second, &g, &h), std::logic_error);
    EXPECT_EQ(g, Eigen::VectorXd::Constant(1, 7));
    EXPECT_EQ(h, Eigen::MatrixXd::Constant(1, 1, 9));
    EXPECT_EQ(energy.dof_calls, 0);
    EXPECT_EQ(energy.value_calls, 0);
    EXPECT_EQ(invalid_geometry.dof_calls, 0);
    EXPECT_EQ(invalid_geometry.value_calls, 0);
    EXPECT_EQ(energy.gradient_calls, 0);
    EXPECT_EQ(energy.hessian_calls, 0);
    EXPECT_DOUBLE_EQ(integrate_volume<Rule>(energy.X_field, energy),
                     integrate_volume<Rule>(make_X_field(), ValueOnly{}));
    EXPECT_EQ(energy.dof_calls, 0);
    EXPECT_EQ(energy.gradient_calls, 0);
    EXPECT_EQ(energy.hessian_calls, 0);
}

TEST(IntegrateVolume, IntegratesIndependentOutputsInOnePassAndOverwritesBuffers) {
    FullEnergy energy;
    Eigen::VectorXd g = Eigen::VectorXd::Ones(1);
    Eigen::MatrixXd h = Eigen::MatrixXd::Ones(1, 2);
    const double value = integrate_volume<Rule>(energy.X_field, energy, &g, &h);
    const double factor = 0.4 * 1.2 + 0.6 * 1.8;
    EXPECT_NEAR(value, factor * 0.5 * energy.coefficients.squaredNorm(), 1e-13);
    EXPECT_TRUE(g.isApprox(factor * energy.coefficients, 1e-13));
    EXPECT_TRUE(h.isApprox(factor * Eigen::Matrix3d::Identity(), 1e-13));
    EXPECT_EQ(energy.dof_calls, 1);
    EXPECT_EQ(energy.value_calls, Rule::num_points);
    EXPECT_EQ(energy.gradient_calls, Rule::num_points);
    EXPECT_EQ(energy.hessian_calls, Rule::num_points);
    EXPECT_DOUBLE_EQ(integrate_volume<Rule>(energy.X_field, energy, &g, &h), value);
    EXPECT_TRUE(g.isApprox(factor * energy.coefficients, 1e-13));
    EXPECT_TRUE(h.isApprox(factor * Eigen::Matrix3d::Identity(), 1e-13));
    Eigen::VectorXd first_g;
    Eigen::MatrixXd second_h;
    EXPECT_DOUBLE_EQ(integrate_volume<Rule>(make_X_field(), FirstOnly{}, &first_g), value);
    EXPECT_DOUBLE_EQ(integrate_volume<Rule>(make_X_field(), SecondOnly{}, nullptr, &second_h), value);
    EXPECT_EQ(first_g, g);
    EXPECT_EQ(second_h, h);
    EXPECT_DOUBLE_EQ(integrate_volume<Rule>(make_X_field(), ValueOnly{make_X_field()}), value);
    EXPECT_NEAR(integrate_volume<Rule>(make_X_field(2), ValueOnly{make_X_field(2)}),
                8 * (0.4 * 1.4 + 0.6 * 2.6) * 0.5 * energy.coefficients.squaredNorm(), 1e-13);
    for (int j = 0; j < 3; ++j) {
        auto plus = energy, minus = energy;
        plus.coefficients[j] += 1e-4;
        minus.coefficients[j] -= 1e-4;
        Eigen::VectorXd plus_g, minus_g;
        const double plus_value = integrate_volume<Rule>(plus.X_field, plus, &plus_g);
        const double minus_value = integrate_volume<Rule>(minus.X_field, minus, &minus_g);
        EXPECT_NEAR(g[j], (plus_value - minus_value) / 2e-4, 1e-10);
        EXPECT_TRUE(h.col(j).isApprox((plus_g - minus_g) / 2e-4, 1e-10));
    }
}
struct BadGradient : FirstOnly {
    int n = 3;
    int num_dofs() const override {
        return n;
    }
    Eigen::VectorXd gradient(const Eigen::Vector3d&) const override {
        return Eigen::Vector2d::Zero();
    }
};
struct BadHessian : SecondOnly {
    Eigen::MatrixXd hessian(const Eigen::Vector3d&) const override {
        return Eigen::MatrixXd::Zero(3, 2);
    }
};
TEST(IntegrateVolume, RejectsInvalidRuntimeDerivativeSizes) {
    Eigen::VectorXd g;
    Eigen::MatrixXd h;
    BadGradient bad;
    EXPECT_THROW(integrate_volume<Rule>(bad.X_field, bad, &g), std::invalid_argument);
    bad.n = 0;
    EXPECT_THROW(integrate_volume<Rule>(bad.X_field, bad, &g), std::invalid_argument);
    EXPECT_NO_THROW(integrate_volume<Rule>(bad.X_field, bad));
    EXPECT_THROW(integrate_volume<Rule>(make_X_field(), BadHessian{}, nullptr, &h), std::invalid_argument);
}

struct ResizingAccumulator : FullEnergy {
    mutable int calls = 0;
    double accumulate(const Eigen::Vector3d&, double, Eigen::VectorXd* gradient,
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
    EXPECT_THROW(integrate_volume<Rule>(energy.X_field, energy, &g), std::invalid_argument);
    EXPECT_EQ(energy.calls, 1);
    EXPECT_THROW(integrate_volume<Rule>(energy.X_field, energy, nullptr, &h), std::invalid_argument);
    EXPECT_EQ(energy.calls, 2);
}
} // namespace

namespace {
struct UnitDensity : LocalEnergy {
    mutable int calls = 0;
    int num_dofs() const override {
        return 0;
    }
    double value(const Eigen::Vector3d&) const override {
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
    const UnitDensity energy;
    EXPECT_NEAR(integrate_volume<Rule>(make_X_field(), energy), 1, 1e-14);
    EXPECT_NEAR(integrate_volume<GaussLegendre3D<2>>(make_X_field(), energy), 1, 1e-14);
    Eigen::Matrix<double, 8, 3> X;
    for (int i = 0; i < 8; ++i) {
        const auto xi = HexQ1Basis::reference_node(i);
        X.row(i) = (Eigen::Vector3d(1, -2, 3) + Eigen::Vector3d(2, 3, 4).asDiagonal() * xi).transpose();
    }
    const InterpolatedField<HexQ1Basis> box(X);
    EXPECT_NEAR(integrate_volume<GaussLegendre3D<2>>(box, energy), 24, 1e-13);
    for (int i = 0; i < 8; ++i) {
        const auto xi = HexQ1Basis::reference_node(i);
        X.row(i) << xi.x() + 0.4 * xi.y() * xi.z(), xi.y() + 0.2 * xi.x() * xi.z(), xi.z();
    }
    const InterpolatedField<HexQ1Basis> warped(X);
    EXPECT_NEAR(integrate_volume<GaussLegendre3D<2>>(warped, energy), 1 - 0.08 / 3, 1e-14);
}

TEST(IntegrateVolume, RejectsInvalidGeometryAtTheEvaluatedPoint) {
    const UnitDensity energy;
    EXPECT_THROW(integrate_volume<Rule>(make_X_field(0), energy), std::invalid_argument);
    EXPECT_THROW(integrate_volume<Rule>(make_X_field(-1), energy), std::invalid_argument);
    EXPECT_EQ(energy.calls, 0);
    Eigen::Matrix<double, 8, 3> X_local;
    for (int i = 0; i < 8; ++i) {
        const auto xi = HexQ1Basis::reference_node(i);
        X_local.row(i) << xi.x(), xi.y(), xi.z() * (1 - 1.5 * xi.x());
    }
    const InterpolatedField<HexQ1Basis> warped(X_local);
    EXPECT_THROW(integrate_volume<Rule>(warped, energy), std::invalid_argument);
    EXPECT_EQ(energy.calls, 1);
}

TEST(IntegrateVolume, RejectsNonFinitePointsWeightsAndOverflowingVolumeWeights) {
    const UnitDensity energy;
    EXPECT_THROW(integrate_volume<NonFinitePoints>(make_X_field(), energy), std::invalid_argument);
    EXPECT_THROW(integrate_volume<OverflowingWeights>(make_X_field(2), energy), std::invalid_argument);
    EXPECT_EQ(energy.calls, 0);
    EXPECT_THROW(integrate_volume<NonFiniteWeights>(make_X_field(), energy), std::invalid_argument);
    EXPECT_EQ(energy.calls, 1);
}

namespace {
struct CountingCoordinates final : Field {
    int components = 3;
    Eigen::Matrix<double, Eigen::Dynamic, 3> derivative = 2 * Eigen::Matrix3d::Identity();
    std::shared_ptr<int> calls = std::make_shared<int>(0);
    std::shared_ptr<int> value_calls = std::make_shared<int>(0);
    int num_components() const override {
        return components;
    }
    int num_dofs() const override {
        return 0;
    }
    Eigen::MatrixXd dvalue_dcoeffs(const Eigen::Vector3d&) const override {
        throw std::logic_error("Coordinate coefficients are fixed");
    }
    Eigen::VectorXd value(const Eigen::Vector3d&) const override {
        ++*value_calls;
        throw std::logic_error("Coordinate position is not needed");
    }
    Eigen::Matrix<double, Eigen::Dynamic, 3> dvalue_dxi(const Eigen::Vector3d&) const override {
        ++*calls;
        return derivative;
    }
};
} // namespace

TEST(IntegrateVolume, ValidatesRawCoordinateDerivativesWithoutRequestingPositionValues) {
    CountingCoordinates X_field;
    const UnitDensity energy;
    EXPECT_DOUBLE_EQ(integrate_volume<Rule>(X_field, energy), 8);
    EXPECT_EQ(*X_field.calls, Rule::num_points);
    const auto rejects_geometry = [&] {
        const int before = energy.calls;
        EXPECT_THROW(integrate_volume<Rule>(X_field, energy), std::invalid_argument);
        EXPECT_EQ(energy.calls, before);
    };
    for (int rows : {2, 4}) {
        X_field.derivative = Eigen::Matrix<double, Eigen::Dynamic, 3>::Zero(rows, 3);
        rejects_geometry();
    }
    X_field.derivative = Eigen::Matrix3d::Identity();
    for (int components : {1, 2, 4}) {
        X_field.components = components;
        rejects_geometry();
    }
    X_field.components = 3;
    for (double bad : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(),
                       -std::numeric_limits<double>::infinity()}) {
        X_field.derivative(0, 0) = bad;
        rejects_geometry();
        X_field.derivative = Eigen::Matrix3d::Identity();
    }
    EXPECT_THROW(integrate_volume<NonFinitePoints>(X_field, energy), std::invalid_argument);
    for (double scale : {0., -1., 1e200}) {
        X_field.derivative = scale * Eigen::Matrix3d::Identity();
        rejects_geometry();
    }
    // Computing a volume weight does not require a representable inverse.
    X_field.derivative = Eigen::Vector3d(1e-310, 1e100, 1e100).asDiagonal();
    EXPECT_GT(integrate_volume<Rule>(X_field, energy), 0);
    EXPECT_EQ(*X_field.value_calls, 0);
}
