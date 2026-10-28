#include "litepgo/fem/basis/3d/hex_hermite_basis.hpp"
#include "litepgo/fem/basis/3d/hex_q1_basis.hpp"
#include "litepgo/fem/basis/3d/tet_p1_basis.hpp"
#include "litepgo/fem/field/deformation_gradient_field.hpp"
#include "litepgo/fem/field/identity_field.hpp"
#include "litepgo/fem/field/interpolated_field.hpp"

#include <Eigen/LU>

#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <type_traits>
#include <utility>

namespace {
using namespace litepgo;

Eigen::Matrix<double, 8, 3> nodes() {
    Eigen::Matrix<double, 8, 3> X;
    for (int a = 0; a < 8; ++a)
        X.row(a) = HexQ1Basis::reference_node(a).transpose();
    return X;
}

template <class B> void check_derivative() {
    auto X = nodes();
    X(6, 0) += 0.3;
    const InterpolatedField3D<HexQ1Basis> X_field(X);
    Eigen::MatrixXd coefficients(B::num_basis_functions, 3);
    for (int j = 0; j < coefficients.size(); ++j)
        coefficients(j / 3, j % 3) = 0.02 * std::sin(j + 0.3);
    const InterpolatedField3D<B> u_field(coefficients);
    const DeformationGradientField3D deformation(u_field, X_field);
    const Field3D& field = deformation;
    EXPECT_EQ(field.num_components(), 9);
    EXPECT_EQ(field.num_dofs(), coefficients.size());
    for (const Eigen::Vector3d xi : {Eigen::Vector3d(0.2, 0.3, 0.4), Eigen::Vector3d(-0.2, 1.1, 0.1)}) {
        const auto derivative = field.eval_dvalue_dcoeffs(xi);
        const Eigen::MatrixXd expected =
            Eigen::kroneckerProduct(X_field.eval_dvalue_dxi(xi).inverse().transpose(), Eigen::Matrix3d::Identity()) *
            u_field.eval_dgrad_xi_dcoeffs(xi);
        EXPECT_TRUE(derivative.isApprox(expected, 1e-14));
        for (int j = 0; j < coefficients.size(); ++j) {
            auto plus = coefficients, minus = coefficients;
            plus(j / 3, j % 3) += 1e-5;
            minus(j / 3, j % 3) -= 1e-5;
            const InterpolatedField3D<B> a(plus), b(minus);
            const Eigen::VectorXd fd = (DeformationGradientField3D(a, X_field).eval_value(xi) -
                                        DeformationGradientField3D(b, X_field).eval_value(xi)) /
                                       2e-5;
            EXPECT_LT((fd - derivative.col(j)).norm(), 1e-9 * (1 + derivative.col(j).norm()));
        }
    }
}

TEST(DeformationGradientField3D, PolymorphicCoefficientDerivativesMatchFiniteDifferencesForEveryBasis) {
    check_derivative<HexQ1Basis>();
    check_derivative<TetP1Basis>();
    check_derivative<HexHermiteBasis>();
}

TEST(DeformationGradientField3D, BorrowsNamedFieldsAndRejectsTemporaryInputs) {
    using UField = InterpolatedField3D<HexQ1Basis>;
    static_assert(!std::is_constructible_v<DeformationGradientField3D, UField&&, const UField&>);
    static_assert(!std::is_constructible_v<DeformationGradientField3D, const UField&, UField&&>);
    static_assert(!std::is_constructible_v<DeformationGradientField3D, UField&&, UField&&>);
    static_assert(!std::is_constructible_v<DeformationGradientField3D, const UField&&, const UField&>);
    static_assert(!std::is_constructible_v<DeformationGradientField3D, const UField&, const UField&&>);
    const Eigen::Matrix3d A = (Eigen::Matrix3d() << 0.1, 0.2, 0, -0.3, 0.1, 0.2, 0, 0, 0.3).finished();
    const Eigen::MatrixXd coefficients = nodes() * A.transpose();
    UField u_field(coefficients), X_field(nodes());
    const DeformationGradientField3D deformation(u_field, X_field);
    const auto copy = deformation;
    EXPECT_EQ(&copy.coordinate_field(), &X_field);
    const Eigen::Vector3d xi(0.2, 0.3, 0.4);
    const auto owned_value = deformation.eval_value(xi);
    EXPECT_TRUE(owned_value.reshaped(3, 3).isApprox(Eigen::Matrix3d::Identity() + A, 1e-14));
    u_field = UField(2 * coefficients);
    EXPECT_TRUE(copy.eval_value(xi).reshaped(3, 3).isApprox(Eigen::Matrix3d::Identity() + 2 * A, 1e-14));
    X_field = UField(2 * nodes());
    EXPECT_TRUE(copy.eval_value(xi).reshaped(3, 3).isApprox(Eigen::Matrix3d::Identity() + A, 1e-14));
    EXPECT_TRUE(owned_value.reshaped(3, 3).isApprox(Eigen::Matrix3d::Identity() + A, 1e-14));
}

TEST(DeformationGradientField3D, ValidatesComponentsParametersGeometryAndOverflow) {
    const InterpolatedField3D<HexQ1Basis> X_field(nodes()), invalid_components(Eigen::MatrixXd::Zero(8, 2));
    const InterpolatedField3D<HexQ1Basis> u_field(Eigen::MatrixXd::Zero(8, 3));
    EXPECT_THROW((DeformationGradientField3D(invalid_components, X_field)), std::invalid_argument);
    EXPECT_THROW((DeformationGradientField3D(u_field, invalid_components)), std::invalid_argument);
    const DeformationGradientField3D deformation(u_field, X_field);
    for (double bad : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
        const Eigen::Vector3d xi = Eigen::Vector3d::Constant(bad);
        EXPECT_THROW(deformation.eval_value(xi), std::invalid_argument);
        EXPECT_THROW(deformation.eval_dvalue_dcoeffs(xi), std::invalid_argument);
    }
    const InterpolatedField3D<HexQ1Basis> collapsed(Eigen::MatrixXd::Zero(8, 3));
    const DeformationGradientField3D invalid_geometry(u_field, collapsed);
    EXPECT_THROW(invalid_geometry.eval_value({0.2, 0.3, 0.4}), std::invalid_argument);
    const InterpolatedField3D<HexQ1Basis> tiny(nodes() * Eigen::Vector3d(1e-200, 1e100, 1e100).asDiagonal());
    const InterpolatedField3D<HexQ1Basis> enormous(1e200 * nodes());
    const DeformationGradientField3D overflow(enormous, tiny);
    EXPECT_THROW(overflow.eval_value({0.2, 0.3, 0.4}), std::invalid_argument);
    const InterpolatedField3D<HexQ1Basis> large_inverse(nodes() * Eigen::Vector3d(1e-308, 1e154, 1e154).asDiagonal());
    const DeformationGradientField3D overflow_derivative(u_field, large_inverse);
    EXPECT_THROW(overflow_derivative.eval_dvalue_dcoeffs({2, 2, 2}), std::invalid_argument);
}

TEST(DeformationGradientField3D, ConstantDisplacementHasIdentityValueAndZeroCoefficientDerivative) {
    const IdentityField3D u_field(Eigen::Vector3d(1, 2, 3));
    const InterpolatedField3D<HexQ1Basis> X_field(nodes());
    const DeformationGradientField3D deformation(u_field, X_field);
    const Field3D& field = deformation;
    EXPECT_EQ(field.eval_value({0.2, 0.3, 0.4}), Eigen::Matrix3d::Identity().reshaped());
    EXPECT_TRUE(field.eval_dvalue_dcoeffs({0.2, 0.3, 0.4}).isZero());
}

TEST(DeformationGradientField3D, DifferentDisplacementBasisOnWarpedCoordinatesUsesPhysicalChainRule) {
    using TensorBasis = TensorProductBasis3D<LinearLagrangeBasis>;
    const auto warped = [](const Eigen::Vector3d& xi) -> Eigen::Vector3d {
        return {xi.x() + 0.4 * xi.y() * xi.z(), xi.y() + 0.2 * xi.x() * xi.z(), xi.z()};
    };
    const auto inverse = [](const Eigen::Vector3d& X) -> Eigen::Vector3d {
        const double det = 1 - 0.08 * X.z() * X.z();
        return {(X.x() - 0.4 * X.z() * X.y()) / det, (X.y() - 0.2 * X.z() * X.x()) / det, X.z()};
    };
    auto X = nodes();
    Eigen::Matrix<double, 8, 3> coefficients;
    for (int a = 0; a < 8; ++a) {
        X.row(a) = warped(HexQ1Basis::reference_node(a));
        const Eigen::Vector3d xi(a / 4, (a / 2) % 2, a % 2);
        coefficients.row(a) << xi.x() * xi.y(), xi.y() * xi.z(), xi.z() * xi.x();
    }
    const InterpolatedField3D<TensorBasis> u_field(coefficients);
    const InterpolatedField3D<HexQ1Basis> X_field(X);
    const DeformationGradientField3D deformation(u_field, X_field);
    const Eigen::Vector3d xi(0.2, 0.3, 0.4);
    const Eigen::Matrix3d F = deformation.eval_value(xi).reshaped(3, 3);
    const Eigen::Vector3d position = warped(xi);
    for (int d = 0; d < 3; ++d) {
        Eigen::Vector3d plus = position, minus = position;
        plus[d] += 1e-5;
        minus[d] -= 1e-5;
        const Eigen::Vector3d fd = (u_field.eval_value(inverse(plus)) - u_field.eval_value(inverse(minus))) / 2e-5;
        EXPECT_TRUE(fd.isApprox((F - Eigen::Matrix3d::Identity()).col(d), 3e-9));
    }
}

class InvalidDisplacement final : public Field3D {
  public:
    Eigen::MatrixXd mixed = Eigen::MatrixXd::Zero(9, 3);
    int num_components() const override {
        return 3;
    }
    int num_dofs() const override {
        return 3;
    }
    Eigen::VectorXd eval_value(const Eigen::Vector3d&) const override {
        return Eigen::Vector3d::Zero();
    }
    Eigen::MatrixXd eval_dvalue_dcoeffs(const Eigen::Vector3d&) const override {
        return Eigen::Matrix3d::Identity();
    }
    Eigen::Matrix<double, Eigen::Dynamic, 3> eval_dvalue_dxi(const Eigen::Vector3d&) const override {
        return Eigen::Matrix3d::Zero();
    }
    Eigen::MatrixXd eval_dgrad_xi_dcoeffs(const Eigen::Vector3d&) const override {
        return mixed;
    }
};

TEST(DeformationGradientField3D, RejectsInvalidMixedDerivativeShapesAndValues) {
    InvalidDisplacement u_field;
    const InterpolatedField3D<HexQ1Basis> X_field(nodes());
    const DeformationGradientField3D deformation(u_field, X_field);
    for (const auto& shape : {std::pair{8, 3}, std::pair{9, 2}}) {
        u_field.mixed = Eigen::MatrixXd::Zero(shape.first, shape.second);
        EXPECT_THROW(deformation.eval_dvalue_dcoeffs({0.2, 0.3, 0.4}), std::invalid_argument);
    }
    u_field.mixed = Eigen::MatrixXd::Constant(9, 3, std::numeric_limits<double>::quiet_NaN());
    EXPECT_THROW(deformation.eval_dvalue_dcoeffs({0.2, 0.3, 0.4}), std::invalid_argument);
}

TEST(DeformationGradientField3D, UnsupportedSpatialDerivativesThrowThroughBaseReference) {
    const IdentityField3D u_field(Eigen::Vector3d::Zero());
    const InterpolatedField3D<HexQ1Basis> X_field(nodes());
    const DeformationGradientField3D deformation(u_field, X_field);
    const Field3D& field = deformation;
    EXPECT_THROW(field.eval_dvalue_dxi({0.2, 0.3, 0.4}), std::logic_error);
    EXPECT_THROW(field.eval_dgrad_xi_dcoeffs({0.2, 0.3, 0.4}), std::logic_error);
}
} // namespace

namespace {
// For u(xi) = xi, F - I is the inverse coordinate derivative.
Eigen::Matrix3d inverse_from_deformation(const litepgo::Field3D& X_field, const Eigen::Vector3d& xi) {
    const litepgo::InterpolatedField3D<litepgo::HexQ1Basis> u_field(nodes());
    const litepgo::DeformationGradientField3D deformation(u_field, X_field);
    return deformation.eval_value(xi).reshaped(3, 3) - Eigen::Matrix3d::Identity();
}
} // namespace

namespace {

// Interior points, corners, a face, and an exterior point, independent of quadrature.
const std::array<Eigen::Vector3d, 7> parameters{{Eigen::Vector3d(0.2, 0.3, 0.4), Eigen::Vector3d(0.83, 0.17, 0.61),
                                                 Eigen::Vector3d(0.5, 0.5, 0.5), Eigen::Vector3d(0, 0, 0),
                                                 Eigen::Vector3d(1, 1, 1), Eigen::Vector3d(0, 0.3, 0.7),
                                                 Eigen::Vector3d(-0.2, 1.2, 0.4)}};

Eigen::Matrix<double, 8, 3> affine_geometry(const Eigen::Matrix3d& dX_dxi, const Eigen::Vector3d& origin) {
    return (nodes() * dX_dxi.transpose()).rowwise() + origin.transpose();
}

// Non-affine but globally invertible on the cube: det(J0) = 1 - alpha*beta*zeta^2.
constexpr double alpha = 0.4;
constexpr double beta = 0.2;

Eigen::Vector3d warped_position(const Eigen::Vector3d& xi) {
    return {xi.x() + alpha * xi.y() * xi.z(), xi.y() + beta * xi.x() * xi.z(), xi.z()};
}

Eigen::Vector3d warped_parameter(const Eigen::Vector3d& X) {
    const double determinant = 1.0 - alpha * beta * X.z() * X.z();
    return {(X.x() - alpha * X.z() * X.y()) / determinant, (X.y() - beta * X.z() * X.x()) / determinant, X.z()};
}

Eigen::Matrix<double, 8, 3> warped_geometry() {
    const auto reference_nodes = nodes();
    Eigen::Matrix<double, 8, 3> X_nodes;
    for (int i = 0; i < 8; ++i) {
        X_nodes.row(i) = warped_position(reference_nodes.row(i).transpose()).transpose();
    }
    return X_nodes;
}

} // namespace

TEST(DeformationGradientField3D, CoordinateDerivativesOnUnitCubeAtArbitraryParameters) {
    const litepgo::InterpolatedField3D<litepgo::HexQ1Basis> reference(nodes());
    for (const auto& xi : parameters) {
        EXPECT_TRUE(reference.eval_value(xi).isApprox(xi, 1e-14));
        EXPECT_TRUE(reference.eval_dvalue_dxi(xi).isApprox(Eigen::Matrix3d::Identity(), 1e-14));
        EXPECT_TRUE(inverse_from_deformation(reference, xi).isIdentity(1e-14));
        EXPECT_NEAR(reference.eval_dvalue_dxi(xi).determinant(), 1.0, 1e-14);
    }
}

TEST(DeformationGradientField3D, CoordinateDerivativesOnTranslatedBox) {
    const Eigen::Vector3d lengths(2.0, 3.0, 4.0);
    const Eigen::Matrix3d expected_dX_dxi = lengths.asDiagonal();
    const Eigen::Vector3d origin(1.0, -2.0, 3.0);
    const litepgo::InterpolatedField3D<litepgo::HexQ1Basis> reference(affine_geometry(expected_dX_dxi, origin));
    for (const auto& xi : parameters) {
        EXPECT_TRUE(reference.eval_value(xi).isApprox(origin + lengths.asDiagonal() * xi, 1e-14));
        EXPECT_TRUE(reference.eval_dvalue_dxi(xi).isApprox(expected_dX_dxi, 1e-14));
        EXPECT_TRUE(inverse_from_deformation(reference, xi).isApprox(expected_dX_dxi.inverse(), 1e-14));
        EXPECT_NEAR(reference.eval_dvalue_dxi(xi).determinant(), 24.0, 1e-13);
    }
}

TEST(DeformationGradientField3D, CoordinateDerivativesOnShearedHexAtArbitraryParameters) {
    Eigen::Matrix3d dX_dxi;
    dX_dxi << 2, 0.3, 0.1, 0.2, 1.5, 0.1, 0, 0.2, 1.1;
    const Eigen::Vector3d origin(1.2, -0.7, 2.1);
    const litepgo::InterpolatedField3D<litepgo::HexQ1Basis> reference(affine_geometry(dX_dxi, origin));

    for (const auto& xi : parameters) {
        EXPECT_TRUE(reference.eval_value(xi).isApprox(origin + dX_dxi * xi, 1e-13));
        EXPECT_TRUE(reference.eval_dvalue_dxi(xi).isApprox(dX_dxi, 1e-13));
        EXPECT_TRUE(inverse_from_deformation(reference, xi).isApprox(dX_dxi.inverse(), 1e-13));
        EXPECT_NEAR(reference.eval_dvalue_dxi(xi).determinant(), dX_dxi.determinant(), 1e-13);
    }
}

TEST(DeformationGradientField3D, InverseCoordinateDerivativeMatchesPhysicalFiniteDifferencesOnWarpedHex) {
    const litepgo::InterpolatedField3D<litepgo::HexQ1Basis> reference(warped_geometry());
    constexpr double h = 1e-6;
    for (const auto& xi : parameters) {
        const Eigen::Vector3d position = warped_position(xi);
        EXPECT_TRUE(reference.eval_value(xi).isApprox(position, 1e-14));
        EXPECT_NEAR(reference.eval_dvalue_dxi(xi).determinant(), 1.0 - alpha * beta * xi.z() * xi.z(), 1e-14);
        for (int d = 0; d < 3; ++d) {
            Eigen::Vector3d plus = position;
            Eigen::Vector3d minus = position;
            plus[d] += h;
            minus[d] -= h;
            const Eigen::Vector3d derivative = (warped_parameter(plus) - warped_parameter(minus)) / (2.0 * h);
            EXPECT_TRUE((inverse_from_deformation(reference, xi).col(d) - derivative).isZero(2e-9));
        }
    }
}

TEST(DeformationGradientField3D, CoordinateDerivativesUseFieldCoefficientSnapshots) {
    Eigen::Matrix<double, 8, 3> X_nodes = nodes();
    const litepgo::InterpolatedField3D<litepgo::HexQ1Basis> reference(X_nodes);
    X_nodes.setConstant(std::numeric_limits<double>::quiet_NaN());
    const Eigen::Vector3d point_xi = Eigen::Vector3d(0.2, 0.3, 0.4);
    EXPECT_NEAR(reference.eval_dvalue_dxi(point_xi).determinant(), 1.0, 1e-14);
    EXPECT_TRUE(inverse_from_deformation(reference, point_xi).allFinite());
}

TEST(DeformationGradientField3D, CoordinateDerivativesRejectNonFiniteGeometry) {
    for (double value : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(),
                         -std::numeric_limits<double>::infinity()}) {
        Eigen::Matrix<double, 8, 3> X_nodes = nodes();
        X_nodes(4, 2) = value;
        EXPECT_THROW((litepgo::InterpolatedField3D<litepgo::HexQ1Basis>(X_nodes)), std::invalid_argument);
    }
}

TEST(DeformationGradientField3D, CoordinateDerivativesRejectNonFiniteParameters) {
    const litepgo::InterpolatedField3D<litepgo::HexQ1Basis> reference(nodes());
    for (double value : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(),
                         -std::numeric_limits<double>::infinity()}) {
        for (int d = 0; d < 3; ++d) {
            Eigen::Vector3d xi(0.2, 0.3, 0.4);
            xi[d] = value;
            EXPECT_THROW(inverse_from_deformation(reference, xi), std::invalid_argument);
        }
    }
}

TEST(DeformationGradientField3D, CoordinateDerivativesRejectCollapsedAndInvertedGeometry) {
    const Eigen::Vector3d xi(0.2, 0.3, 0.4);
    const litepgo::InterpolatedField3D<litepgo::HexQ1Basis> collapsed(Eigen::Matrix<double, 8, 3>::Zero());
    EXPECT_THROW(inverse_from_deformation(collapsed, xi), std::invalid_argument);
    Eigen::Matrix<double, 8, 3> X_nodes = nodes();
    X_nodes.col(2).setZero();
    const litepgo::InterpolatedField3D<litepgo::HexQ1Basis> flat(X_nodes);
    EXPECT_THROW(inverse_from_deformation(flat, xi), std::invalid_argument);
    X_nodes = nodes();
    X_nodes.col(0) *= -1.0;
    const litepgo::InterpolatedField3D<litepgo::HexQ1Basis> inverted(X_nodes);
    EXPECT_THROW(inverse_from_deformation(inverted, xi), std::invalid_argument);
}

TEST(DeformationGradientField3D, CoordinateDerivativesValidateTheRequestedPoint) {
    Eigen::Matrix<double, 8, 3> X_nodes = nodes();
    // det(J0) = 1 - 1.5*xi: validity depends on the requested parameter point.
    for (int i = 0; i < 8; ++i) {
        X_nodes(i, 2) *= 1.0 - 1.5 * X_nodes(i, 0);
    }
    const litepgo::InterpolatedField3D<litepgo::HexQ1Basis> reference(X_nodes);
    EXPECT_NO_THROW(inverse_from_deformation(reference, Eigen::Vector3d(0.2, 0.3, 0.4)));
    EXPECT_THROW(inverse_from_deformation(reference, Eigen::Vector3d(0.8, 0.3, 0.4)), std::invalid_argument);
}

TEST(DeformationGradientField3D, CoordinateDerivativesAcceptSmallPositiveDeterminantsAndRejectOverflow) {
    const Eigen::Vector3d xi(0.2, 0.3, 0.4);
    const litepgo::InterpolatedField3D<litepgo::HexQ1Basis> small(1e-6 * nodes());
    EXPECT_GT(small.eval_dvalue_dxi(xi).determinant(), 0.0);
    EXPECT_TRUE(inverse_from_deformation(small, xi).allFinite());
    const litepgo::InterpolatedField3D<litepgo::HexQ1Basis> enormous(1e200 * nodes());
    EXPECT_THROW(inverse_from_deformation(enormous, xi), std::invalid_argument);
}

TEST(DeformationGradientField3D, CoordinateDerivativesRejectNonVolumeComponentsAndNonFiniteInverse) {
    for (int components : {1, 2, 4}) {
        const litepgo::InterpolatedField3D<litepgo::HexQ1Basis> X_field(Eigen::MatrixXd::Zero(8, components));
        EXPECT_THROW(inverse_from_deformation(X_field, Eigen::Vector3d::Zero()), std::invalid_argument);
    }
    const litepgo::InterpolatedField3D<litepgo::HexQ1Basis> X_field(nodes() *
                                                                    Eigen::Vector3d(1e-310, 1e100, 1e100).asDiagonal());
    EXPECT_NO_THROW(X_field.eval_dvalue_dxi(Eigen::Vector3d::Constant(0.5)));
    EXPECT_THROW(inverse_from_deformation(X_field, Eigen::Vector3d::Constant(0.5)), std::invalid_argument);
}

namespace {
class SpatialDerivativeOnlyField final : public litepgo::Field3D {
  public:
    int components = 3;
    Eigen::Matrix<double, Eigen::Dynamic, 3> derivative = Eigen::Matrix3d::Identity();
    int num_components() const override {
        return components;
    }
    int num_dofs() const override {
        return 0;
    }
    Eigen::VectorXd eval_value(const Eigen::Vector3d&) const override {
        throw std::logic_error("Field3D value is not needed");
    }
    Eigen::MatrixXd eval_dvalue_dcoeffs(const Eigen::Vector3d&) const override {
        throw std::logic_error("Field3D coefficients are fixed");
    }
    Eigen::Matrix<double, Eigen::Dynamic, 3> eval_dvalue_dxi(const Eigen::Vector3d&) const override {
        return derivative;
    }
};
} // namespace

TEST(DeformationGradientField3D, CoordinateDerivativesValidateSpatialDerivativesWithoutRequestingValues) {
    SpatialDerivativeOnlyField raw;
    const litepgo::Field3D& field = raw;
    const litepgo::InterpolatedField3D<litepgo::HexQ1Basis> u_field(nodes());
    const litepgo::DeformationGradientField3D deformation(u_field, field);
    const Eigen::Vector3d xi(0.2, 0.3, 0.4);
    EXPECT_TRUE(field.eval_dvalue_dxi(xi).isIdentity());
    EXPECT_TRUE(inverse_from_deformation(field, xi).isIdentity());
    for (int rows : {2, 4}) {
        raw.derivative = Eigen::Matrix<double, Eigen::Dynamic, 3>::Zero(rows, 3);
        EXPECT_THROW(inverse_from_deformation(field, xi), std::invalid_argument);
        EXPECT_THROW(deformation.eval_dvalue_dcoeffs(xi), std::invalid_argument);
    }
    raw.derivative = Eigen::Matrix3d::Identity();
    raw.components = 2;
    EXPECT_THROW(inverse_from_deformation(field, xi), std::invalid_argument);
    EXPECT_THROW(deformation.eval_dvalue_dcoeffs(xi), std::invalid_argument);
    raw.components = 3;
    for (double bad : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(),
                       -std::numeric_limits<double>::infinity()}) {
        raw.derivative(0, 0) = bad;
        EXPECT_THROW(inverse_from_deformation(field, xi), std::invalid_argument);
        EXPECT_THROW(deformation.eval_dvalue_dcoeffs(xi), std::invalid_argument);
        raw.derivative = Eigen::Matrix3d::Identity();
        EXPECT_THROW(inverse_from_deformation(field, Eigen::Vector3d::Constant(bad)), std::invalid_argument);
        EXPECT_THROW(deformation.eval_dvalue_dcoeffs(Eigen::Vector3d::Constant(bad)), std::invalid_argument);
    }
    raw.derivative = -Eigen::Matrix3d::Identity();
    EXPECT_EQ(field.eval_dvalue_dxi(xi), raw.derivative);
    EXPECT_THROW(inverse_from_deformation(field, xi), std::invalid_argument);
    EXPECT_THROW(deformation.eval_dvalue_dcoeffs(xi), std::invalid_argument);
}

TEST(DeformationGradientField3D, ValueUsesOnlyDisplacementDerivativesAndRetainsValidation) {
    SpatialDerivativeOnlyField displacement;
    displacement.derivative << 0.1, 0.2, -0.03, 0.04, -0.1, 0.02, 0.01, -0.02, 0.3;
    const litepgo::InterpolatedField3D<litepgo::HexQ1Basis> coordinates(2 * nodes());
    const litepgo::DeformationGradientField3D deformation(displacement, coordinates);
    const Eigen::Vector3d xi(0.2, 0.3, 0.4);
    EXPECT_TRUE(deformation.eval_value(xi).reshaped(3, 3).isApprox(
        Eigen::Matrix3d::Identity() + 0.5 * displacement.derivative, 1e-14));
    displacement.components = 2;
    EXPECT_THROW(deformation.eval_value(xi), std::invalid_argument);
    displacement.components = 3;
    for (int rows : {2, 4}) {
        displacement.derivative = Eigen::Matrix<double, Eigen::Dynamic, 3>::Zero(rows, 3);
        EXPECT_THROW(deformation.eval_value(xi), std::invalid_argument);
    }
    displacement.derivative = Eigen::Matrix3d::Zero();
    for (double bad : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(),
                       -std::numeric_limits<double>::infinity()}) {
        displacement.derivative(0, 0) = bad;
        EXPECT_THROW(deformation.eval_value(xi), std::invalid_argument);
    }
}
