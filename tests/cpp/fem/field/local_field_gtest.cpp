#include "litepgo/dof/constant_dof_map.hpp"
#include "litepgo/dof/element_dof_map.hpp"
#include "litepgo/fem/basis/hex_hermite_basis.hpp"
#include "litepgo/fem/basis/hex_q1_basis.hpp"
#include "litepgo/fem/basis/tet_p1_basis.hpp"
#include "litepgo/fem/field/deformation_gradient_field.hpp"
#include "litepgo/fem/field/identity_field.hpp"
#include "litepgo/fem/field/interpolated_value_field.hpp"
#include "litepgo/fem/geometry/rest.hpp"
#include "litepgo/fem/quadrature/gauss_legendre_3d.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <limits>

namespace {
using namespace litepgo;
Eigen::Matrix<double, 8, 3> nodes() {
    Eigen::Matrix<double, 8, 3> X;
    for (int i = 0; i < 8; ++i)
        X.row(i) = HexQ1Basis::reference_node(i).transpose();
    return X;
}

template <class B> void check_deformation_jacobian() {
    auto X = nodes();
    X(6, 0) += 0.3;
    const Interpolation<HexQ1Basis> geometry(X);
    Eigen::MatrixXd u(B::num_basis_functions, 3);
    for (int j = 0; j < u.size(); ++j)
        u(j / 3, j % 3) = 0.02 * std::sin(j + 0.3);
    const DeformationGradientField<B> field(u);
    constexpr double step = 1e-5;
    for (const Eigen::Vector3d xi : {Eigen::Vector3d(0.2, 0.3, 0.4), Eigen::Vector3d(-0.2, 1.1, 0.1)}) {
        const auto point = rest_point(geometry, xi);
        const auto J = field.jacobian(point);
        const Eigen::MatrixXd previous =
            Eigen::kroneckerProduct(point.inverse_jacobian.transpose(), Eigen::Matrix3d::Identity()) *
            Interpolation<B>(u).gradient_matrix(xi);
        EXPECT_TRUE(J.isApprox(previous, 1e-14));
        for (int j = 0; j < u.size(); ++j) {
            auto plus = u, minus = u;
            plus(j / 3, j % 3) += step;
            minus(j / 3, j % 3) -= step;
            const Eigen::Matrix3d fd =
                (DeformationGradientField<B>(plus).value(point) - DeformationGradientField<B>(minus).value(point)) /
                (2 * step);
            EXPECT_LT((fd.reshaped() - J.col(j)).norm(), 1e-9 * (1 + J.col(j).norm())) << "DOF " << j;
        }
    }
}

TEST(LocalField, DeformationJacobiansMatchFiniteDifferencesForAllBases) {
    check_deformation_jacobian<HexQ1Basis>();
    check_deformation_jacobian<TetP1Basis>();
    check_deformation_jacobian<HexHermiteBasis>();
}

template <class B> void check_deformation_pullback() {
    auto X = nodes();
    X(6, 0) += 0.3;
    const Interpolation<HexQ1Basis> geometry(X);
    const DeformationGradientField<B> field(Eigen::MatrixXd::Zero(B::num_basis_functions, 3));
    Eigen::Matrix3d P;
    P << 1, -2, 0.3, 0.4, 2, -0.7, 0.8, 0.1, -3;
    Eigen::Matrix<double, 9, 9> H;
    for (int j = 0; j < H.size(); ++j)
        H.data()[j] = std::sin(0.7 * j + 0.2); // Deliberately nonsymmetric.
    for (const Eigen::Vector3d xi : {Eigen::Vector3d(0.2, 0.3, 0.4), Eigen::Vector3d(-0.2, 1.1, 0.1)}) {
        const auto point = rest_point(geometry, xi);
        const auto J = field.jacobian(point);
        for (double weight : {-0.4, 0., 0.7}) {
            for (int request = 0; request < 4; ++request) {
                Eigen::VectorXd g = Eigen::VectorXd::Constant(field.num_dofs(), 2);
                Eigen::MatrixXd h = Eigen::MatrixXd::Constant(field.num_dofs(), field.num_dofs(), 3);
                const Eigen::VectorXd expected_g = request & 1 ? (g + weight * J.transpose() * P.reshaped()).eval() : g;
                const Eigen::MatrixXd expected_h = request & 2 ? (h + weight * J.transpose() * H * J).eval() : h;
                field.accumulate_derivatives(point, P, H, weight, request & 1 ? &g : nullptr,
                                             request & 2 ? &h : nullptr);
                EXPECT_TRUE(g.isApprox(expected_g, 1e-13));
                EXPECT_TRUE(h.isApprox(expected_h, 1e-13));
            }
        }
    }
}

TEST(LocalField, DeformationPullbacksMatchDenseContractionsForAllBases) {
    check_deformation_pullback<HexQ1Basis>();
    check_deformation_pullback<TetP1Basis>();
    check_deformation_pullback<HexHermiteBasis>();
}

TEST(LocalField, DeformationPullbackRejectsInvalidDimensionsWeightsAndGeometry) {
    const DeformationGradientField<HexQ1Basis> field(Eigen::MatrixXd::Zero(8, 3));
    auto point = rest_point(Interpolation<HexQ1Basis>(nodes()), {0.2, 0.3, 0.4});
    Eigen::VectorXd g = Eigen::VectorXd::Constant(24, 2);
    Eigen::MatrixXd h = Eigen::MatrixXd::Constant(24, 24, 3);
    const Eigen::MatrixXd P = Eigen::Matrix3d::Identity(), H = Eigen::Matrix<double, 9, 9>::Identity();
    EXPECT_THROW(field.accumulate_derivatives(point, Eigen::VectorXd::Ones(8), H, 1, &g, nullptr),
                 std::invalid_argument);
    EXPECT_THROW(field.accumulate_derivatives(point, P, Eigen::Matrix3d::Identity(), 1, &g, &h), std::invalid_argument);
    EXPECT_THROW(field.accumulate_derivatives(point, P, H, std::numeric_limits<double>::infinity(), &g, &h),
                 std::invalid_argument);
    EXPECT_EQ(g, Eigen::VectorXd::Constant(24, 2));
    EXPECT_EQ(h, Eigen::MatrixXd::Constant(24, 24, 3));
    g.resize(23);
    EXPECT_THROW(field.accumulate_derivatives(point, P, H, 1, &g, nullptr), std::invalid_argument);
    g.resize(24);
    h.resize(23, 24);
    EXPECT_THROW(field.accumulate_derivatives(point, P, H, 1, nullptr, &h), std::invalid_argument);
    point.inverse_jacobian(0, 0) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(field.accumulate_derivatives(point, P, H, 1, &g, nullptr), std::invalid_argument);
}

template <bool InvalidGradient> struct InvalidFieldBasis {
    static constexpr int num_basis_functions = HexQ1Basis::num_basis_functions;
    using Sample = HexQ1Basis::Sample;
    static Sample evaluate(const Eigen::Vector3d& xi) {
        auto sample = HexQ1Basis::evaluate(xi);
        if constexpr (InvalidGradient)
            sample.gradients(0, 0) = std::numeric_limits<double>::quiet_NaN();
        else
            sample.values[0] = std::numeric_limits<double>::quiet_NaN();
        return sample;
    }
};

TEST(LocalField, DeformationJacobianValidatesGradientsAndOverflowIndependentlyOfBasisValues) {
    auto point = rest_point(Interpolation<HexQ1Basis>(nodes()), {0.2, 0.3, 0.4});
    const DeformationGradientField<InvalidFieldBasis<true>> bad(Eigen::MatrixXd::Zero(8, 3));
    const DeformationGradientField<InvalidFieldBasis<false>> valid(Eigen::MatrixXd::Zero(8, 3));
    EXPECT_THROW(bad.jacobian(point), std::invalid_argument);
    EXPECT_NO_THROW(valid.jacobian(point));
    point.inverse_jacobian.setConstant(std::numeric_limits<double>::max());
    EXPECT_THROW(valid.jacobian(point), std::invalid_argument);
}
TEST(LocalField, AffineValuesAndCoefficientJacobians) {
    const auto X = nodes();
    const Interpolation<HexQ1Basis> geometry(X);
    const auto point = litepgo::rest_point(geometry, {0.2, 0.3, 0.4});
    Eigen::Matrix3d A;
    A << 0.1, 0.2, 0, -0.3, 0.1, 0.2, 0.1, 0, 0.3;
    Eigen::Matrix<double, 8, 3> u = X * A.transpose();
    Eigen::Matrix<double, 8, 2> c;
    c.col(0) = 2 + X.col(0).array();
    c.col(1) = 3 - X.col(1).array();
    const DeformationGradientField<litepgo::HexQ1Basis> displacement(u);
    const InterpolatedValueField<litepgo::HexQ1Basis> parameters(c);
    u.setZero();
    EXPECT_TRUE(displacement.value(point).isApprox(Eigen::Matrix3d::Identity() + A, 1e-13));
    EXPECT_TRUE(parameters.value(point).isApprox(Eigen::Vector2d(2.2, 2.7), 1e-13));
    constexpr double h = 1e-4;
    u = X * A.transpose();
    for (int j = 0; j < 24; ++j) {
        auto plus = u, minus = u;
        plus(j / 3, j % 3) += h;
        minus(j / 3, j % 3) -= h;
        const Eigen::Matrix3d fd = (DeformationGradientField<litepgo::HexQ1Basis>(plus).value(point) -
                                    DeformationGradientField<litepgo::HexQ1Basis>(minus).value(point)) /
                                   (2 * h);
        EXPECT_TRUE(displacement.jacobian(point).col(j).isApprox(
            Eigen::Map<const Eigen::Matrix<double, 9, 1>>(fd.data()), 1e-9));
    }
    for (int j = 0; j < 16; ++j) {
        auto plus = c, minus = c;
        plus(j / 2, j % 2) += h;
        minus(j / 2, j % 2) -= h;
        EXPECT_TRUE((parameters.jacobian(point).col(j).isApprox(
            (InterpolatedValueField<litepgo::HexQ1Basis>(plus).value(point) -
             InterpolatedValueField<litepgo::HexQ1Basis>(minus).value(point)) /
                (2 * h),
            1e-9)));
    }
    u(0, 0) = std::numeric_limits<double>::quiet_NaN();
    c(0, 0) = std::numeric_limits<double>::infinity();
    EXPECT_THROW((DeformationGradientField<litepgo::HexQ1Basis>(u)), std::invalid_argument);
    EXPECT_THROW((InterpolatedValueField<litepgo::HexQ1Basis>(c)), std::invalid_argument);
}
TEST(LocalField, DirectValuesIgnoreLocationAndOwnData) {
    Eigen::Matrix3d input;
    input << 1, 2, 3, 4, 5, 6, 7, 8, 9;
    const IdentityField<Eigen::Matrix3d> field(input);
    input.setZero();
    litepgo::RestPointData first_point{}, second_point{};
    first_point.position.setConstant(0.2);
    second_point.position.setConstant(100.0);
    EXPECT_EQ(field.value(first_point), field.value(second_point));
    EXPECT_EQ(field.value(litepgo::RestPointData{})(2, 2), 9);
    EXPECT_TRUE(field.jacobian(litepgo::RestPointData{}).isIdentity());
    const auto original = field.value(litepgo::RestPointData{});
    for (int j = 0; j < 9; ++j) {
        auto perturbed = original;
        perturbed.data()[j] += 1e-4;
        const Eigen::Matrix3d fd =
            (IdentityField<Eigen::Matrix3d>(perturbed).value(litepgo::RestPointData{}) - original) / 1e-4;
        EXPECT_LT(
            (Eigen::Map<const Eigen::Matrix<double, 9, 1>>(fd.data()) - field.jacobian(litepgo::RestPointData{}).col(j))
                .norm(),
            1e-10);
    }
    const Eigen::Matrix<double, 3, 3, Eigen::RowMajor> row_major = original;
    EXPECT_EQ((IdentityField<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(row_major).value(litepgo::RestPointData{})),
              original);
    input(0, 0) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW((IdentityField<Eigen::Matrix3d>(input)), std::invalid_argument);
}
} // namespace

TEST(LocalField, ExplicitlyExtractedValuesOwnOnlySelectedSnapshots) {
    litepgo::Vec data = Vec::Zero(18);
    const ElementDofMap map(TetTopology(4, {{0, 1, 2, 3}, {0, 1, 2, 3}}), 9);
    data.head<9>() = Eigen::Matrix3d::Identity().eval().reshaped();
    data.tail<9>() = (2 * Eigen::Matrix3d::Identity()).eval().reshaped();
    const litepgo::IdentityField<Eigen::Matrix3d> first(map.gather(0, data).reshaped(3, 3));
    const litepgo::IdentityField<Eigen::Matrix3d> second(map.gather(1, data).reshaped(3, 3));
    EXPECT_EQ(first.num_dofs(), 9);
    EXPECT_EQ(data.size(), 18);
    data.head<9>() = (3 * Eigen::Matrix3d::Identity()).eval().reshaped();
    EXPECT_EQ(first.value(litepgo::RestPointData{}), Eigen::Matrix3d::Identity());
    EXPECT_EQ(second.value(litepgo::RestPointData{}), 2 * Eigen::Matrix3d::Identity());
    EXPECT_THROW((litepgo::IdentityField<Eigen::Matrix3d>(map.gather(-1, data).reshaped(3, 3))), std::out_of_range);
    EXPECT_THROW((litepgo::IdentityField<Eigen::Matrix3d>(map.gather(2, data).reshaped(3, 3))), std::out_of_range);
    data[0] = std::numeric_limits<double>::infinity();
    EXPECT_THROW((litepgo::IdentityField<Eigen::Matrix3d>(map.gather(0, data).reshaped(3, 3))), std::invalid_argument);
    const auto owned = [] {
        litepgo::Vec temporary = Vec::Zero(2);
        temporary << 4, 5;
        return litepgo::IdentityField<Eigen::Vector2d>(
            ElementDofMap(TetTopology(4, {{0, 1, 2, 3}}), 2).gather(0, temporary));
    }();
    EXPECT_EQ(owned.value(litepgo::RestPointData{}), Eigen::Vector2d(4, 5));

    litepgo::Vec shared(Eigen::Vector2d(2, 3));
    const litepgo::IdentityField<Eigen::Vector2d> snapshot(shared);
    shared << 4, 5;
    EXPECT_EQ(snapshot.value(litepgo::RestPointData{}), Eigen::Vector2d(2, 3));
    EXPECT_EQ(snapshot.value(litepgo::RestPointData{}), snapshot.value(litepgo::RestPointData{}));
    EXPECT_TRUE(snapshot.jacobian(litepgo::RestPointData{}).isIdentity());
    const auto shared_owned = [] {
        const litepgo::Vec temporary(Eigen::Vector2d(6, 7));
        return litepgo::IdentityField<Eigen::Vector2d>(temporary);
    }();
    EXPECT_EQ(shared_owned.value(litepgo::RestPointData{}), Eigen::Vector2d(6, 7));
    shared[0] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW((litepgo::IdentityField<Eigen::Vector2d>(shared)), std::invalid_argument);
}

TEST(LocalField, ConstantMatrixJacobianUsesColumnMajorDofs) {
    Eigen::Matrix<double, 2, 2, Eigen::RowMajor> input;
    input << 1, 2, 3, 4;
    const litepgo::Vec data(input.reshaped().eval());
    const litepgo::IdentityField<Eigen::Matrix2d> field(data.reshaped(2, 2));
    EXPECT_EQ(field.value(litepgo::RestPointData{}), input);
    for (int j = 0; j < 4; ++j) {
        auto plus = data, minus = data;
        plus[j] += 1e-4;
        minus[j] -= 1e-4;
        const Eigen::Matrix2d fd =
            (litepgo::IdentityField<Eigen::Matrix2d>(plus.reshaped(2, 2)).value(litepgo::RestPointData{}) -
             litepgo::IdentityField<Eigen::Matrix2d>(minus.reshaped(2, 2)).value(litepgo::RestPointData{})) /
            2e-4;
        EXPECT_LT(
            (Eigen::Map<const Eigen::Vector4d>(fd.data()) - field.jacobian(litepgo::RestPointData{}).col(j)).norm(),
            1e-10);
    }
}

TEST(LocalField, FieldsRejectIncompatibleShapesAndNonFiniteParameters) {
    EXPECT_THROW(litepgo::DeformationGradientField<litepgo::HexQ1Basis>(Eigen::MatrixXd::Zero(8, 2)),
                 std::invalid_argument);
    EXPECT_THROW(litepgo::DeformationGradientField<litepgo::HexQ1Basis>(Eigen::MatrixXd::Zero(7, 3)),
                 std::invalid_argument);
    EXPECT_THROW(litepgo::InterpolatedValueField<litepgo::HexQ1Basis>(Eigen::MatrixXd::Zero(7, 2)),
                 std::invalid_argument);
    const litepgo::DeformationGradientField<litepgo::HexQ1Basis> deformation(Eigen::MatrixXd::Zero(8, 3));
    const litepgo::InterpolatedValueField<litepgo::HexQ1Basis> parameters(Eigen::MatrixXd::Zero(8, 4));
    const litepgo::AffineField<Eigen::Matrix3d>& deformation_interface = deformation;
    const litepgo::AffineField<Eigen::VectorXd>& parameter_interface = parameters;
    EXPECT_THROW(deformation_interface.value(litepgo::RestPointData{}), std::invalid_argument);
    EXPECT_THROW(deformation_interface.jacobian(litepgo::RestPointData{}), std::invalid_argument);
    auto point = litepgo::rest_point(litepgo::Interpolation<litepgo::HexQ1Basis>(nodes()), {0.2, 0.3, 0.4});
    EXPECT_EQ(deformation_interface.num_dofs(), 24);
    EXPECT_EQ(parameter_interface.num_dofs(), 32);
    EXPECT_EQ(parameter_interface.value(point).size(), 4);
    EXPECT_EQ(parameter_interface.jacobian(point).cols(), 32);
    const auto valid = point;
    point.inverse_jacobian.setConstant(std::numeric_limits<double>::infinity());
    EXPECT_THROW(deformation_interface.value(point), std::invalid_argument);
    EXPECT_THROW(deformation_interface.jacobian(point), std::invalid_argument);
    EXPECT_NO_THROW(parameter_interface.value(point));
    point = valid;
    point.xi.setConstant(std::numeric_limits<double>::quiet_NaN());
    EXPECT_THROW(deformation_interface.value(point), std::invalid_argument);
    EXPECT_THROW(deformation_interface.jacobian(point), std::invalid_argument);
    EXPECT_THROW(parameter_interface.value(point), std::invalid_argument);
    EXPECT_THROW(parameter_interface.jacobian(point), std::invalid_argument);
}

TEST(LocalField, IdentityFieldPreservesExplicitValueTypeAndValidatesDynamicValues) {
    const litepgo::RestPointData point{};
    Eigen::VectorXd input = Eigen::VectorXd::LinSpaced(4, 1, 4);
    const litepgo::IdentityField<Eigen::VectorXd> field(input);
    static_assert(std::is_same_v<decltype(field.value(point)), Eigen::VectorXd>);
    const litepgo::IdentityField<Eigen::Vector2d> fixed(Eigen::Vector2d(2, 3));
    static_assert(std::is_same_v<decltype(fixed.value(point)), Eigen::Vector2d>);
    EXPECT_EQ(field.num_dofs(), 4);
    EXPECT_TRUE(field.jacobian(point).isIdentity());
    input.setZero();
    EXPECT_EQ(field.value(point), Eigen::VectorXd::LinSpaced(4, 1, 4));
    for (int j = 0; j < 4; ++j) {
        auto plus = field.value(point), minus = plus;
        plus[j] += 1e-4;
        minus[j] -= 1e-4;
        const Eigen::VectorXd fd = (litepgo::IdentityField<Eigen::VectorXd>(plus).value(point) -
                                    litepgo::IdentityField<Eigen::VectorXd>(minus).value(point)) /
                                   2e-4;
        EXPECT_TRUE(fd.isApprox(field.jacobian(point).col(j), 1e-10));
    }
    EXPECT_THROW((litepgo::IdentityField<Eigen::VectorXd>{Eigen::VectorXd{}}), std::invalid_argument);
    input[0] = std::numeric_limits<double>::infinity();
    EXPECT_THROW((litepgo::IdentityField<Eigen::VectorXd>{input}), std::invalid_argument);
}

TEST(LocalField, FieldsChooseTheirOwnBasisOnWarpedGeometry) {
    using TensorBasis = litepgo::TensorProductBasis3D<litepgo::LagrangeBasis>;
    const auto warped = [](const Eigen::Vector3d& xi) -> Eigen::Vector3d {
        return {xi.x() + 0.4 * xi.y() * xi.z(), xi.y() + 0.2 * xi.x() * xi.z(), xi.z()};
    };
    const auto inverse = [](const Eigen::Vector3d& X) -> Eigen::Vector3d {
        const double det = 1 - 0.08 * X.z() * X.z();
        return {(X.x() - 0.4 * X.z() * X.y()) / det, (X.y() - 0.2 * X.z() * X.x()) / det, X.z()};
    };
    Eigen::Matrix<double, 8, 3> X, u;
    for (int i = 0; i < 8; ++i) {
        X.row(i) = warped(litepgo::HexQ1Basis::reference_node(i)).transpose();
        const Eigen::Vector3d xi(i / 4, (i / 2) % 2, i % 2);
        u.row(i) << xi.x() * xi.y(), xi.y() * xi.z(), xi.z() * xi.x();
    }
    const litepgo::Interpolation<litepgo::HexQ1Basis> geometry(X);
    const litepgo::DeformationGradientField<TensorBasis> deformation(u);
    const litepgo::InterpolatedValueField<TensorBasis> values(u);
    const litepgo::InterpolatedValueField<litepgo::HexQ1Basis> positions(X);
    constexpr double h = 1e-5;
    for (int q = 0; q < litepgo::GaussLegendre3D<2>::num_points; ++q) {
        const auto point = litepgo::rest_point(geometry, litepgo::GaussLegendre3D<2>::points().row(q).transpose());
        EXPECT_TRUE(positions.value(point).isApprox(point.position, 1e-14));
        const auto& xi = point.xi;
        EXPECT_TRUE(
            values.value(point).isApprox(Eigen::Vector3d(xi.x() * xi.y(), xi.y() * xi.z(), xi.z() * xi.x()), 1e-14));
        for (int d = 0; d < 3; ++d) {
            Eigen::Vector3d plus = point.position, minus = point.position;
            plus[d] += h;
            minus[d] -= h;
            const Eigen::Vector3d fd = (values.value(litepgo::rest_point(geometry, inverse(plus))) -
                                        values.value(litepgo::rest_point(geometry, inverse(minus)))) /
                                       (2 * h);
            EXPECT_TRUE(fd.isApprox((deformation.value(point) - Eigen::Matrix3d::Identity()).col(d), 3e-9));
        }
        for (int j = 0; j < 24; ++j) {
            auto plus = u, minus = u;
            plus(j / 3, j % 3) += h;
            minus(j / 3, j % 3) -= h;
            const Eigen::Matrix3d fd = (litepgo::DeformationGradientField<TensorBasis>(plus).value(point) -
                                        litepgo::DeformationGradientField<TensorBasis>(minus).value(point)) /
                                       (2 * h);
            EXPECT_TRUE(fd.reshaped().isApprox(deformation.jacobian(point).col(j), 3e-9));
            const Eigen::VectorXd value_fd = (litepgo::InterpolatedValueField<TensorBasis>(plus).value(point) -
                                              litepgo::InterpolatedValueField<TensorBasis>(minus).value(point)) /
                                             (2 * h);
            EXPECT_TRUE(value_fd.isApprox(values.jacobian(point).col(j), 3e-9));
        }
    }
}
