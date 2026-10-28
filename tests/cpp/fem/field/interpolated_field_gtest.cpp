#include "litepgo/fem/basis/3d/hex_q1_basis.hpp"
#include "litepgo/fem/basis/3d/tet_p1_basis.hpp"
#include "litepgo/fem/field/interpolated_field.hpp"

#include <Eigen/LU>
#include <stdexcept>

#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <type_traits>

namespace {
using namespace litepgo;
static_assert(std::is_same_v<InterpolatedFieldT<HexQ1Basis, 3>, InterpolatedField3D<HexQ1Basis>>);

template <class B>
concept CanInterpolate = requires { typename InterpolatedField3D<B>; };
static_assert(CanInterpolate<HexQ1Basis>);
static_assert(!CanInterpolate<LinearLagrangeBasis>);
static_assert(InterpolatedField3D<HexQ1Basis>::num_basis_functions == 8);

Eigen::Matrix<double, 8, 3> corners() {
    Eigen::Matrix<double, 8, 3> X_nodes;
    for (int i = 0; i < 8; ++i)
        X_nodes.row(i) = HexQ1Basis::reference_node(i).transpose();
    return X_nodes;
}

// Static evaluation must not require constructing a basis object.
template <int InvalidPart> struct InvalidBasis {
    InvalidBasis() = delete;
    static constexpr int num_basis_functions = 8;
    static auto eval_N(const Eigen::Vector3d& xi) {
        auto result = HexQ1Basis::eval_N(xi);
        if constexpr (InvalidPart == 0)
            result.setConstant(std::numeric_limits<double>::quiet_NaN());
        return result;
    }
    static auto eval_dN_dxi(const Eigen::Vector3d& xi) {
        auto result = HexQ1Basis::eval_dN_dxi(xi);
        if constexpr (InvalidPart == 1)
            result.setConstant(std::numeric_limits<double>::infinity());
        return result;
    }
    static auto eval_d2N_dxi2(const Eigen::Vector3d& xi) {
        auto result = HexQ1Basis::eval_d2N_dxi2(xi);
        if constexpr (InvalidPart == 2)
            result.setConstant(std::numeric_limits<double>::infinity());
        return result;
    }
};

template <int Components, int Storage> void check_coefficient_derivatives() {
    SCOPED_TRACE(Components);
    SCOPED_TRACE(Storage);
    Eigen::Matrix<double, 8, Components, Storage> coefficients;
    for (int i = 0; i < 8; ++i)
        for (int c = 0; c < Components; ++c)
            coefficients(i, c) = std::sin(0.37 * (i * Components + c + 1));
    const InterpolatedField3D<HexQ1Basis> concrete(coefficients);
    const Field3D& field = concrete;
    const Eigen::Vector3d xi(0.2, 0.3, 0.4);
    const Eigen::VectorXd flattened = coefficients.template reshaped<Eigen::RowMajor>();
    EXPECT_EQ(field.num_components(), Components);
    EXPECT_EQ(field.num_dofs(), 8 * Components);
    const auto gradient = field.eval_dvalue_dxi(xi);
    const auto hessian = field.eval_d2value_dxi2(xi);
    EXPECT_EQ(hessian.rows(), Components);
    EXPECT_EQ(hessian.cols(), 6);
    EXPECT_EQ(field.eval_dhess_xi_dcoeffs(xi).rows(), 6 * Components);
    EXPECT_EQ(field.eval_dhess_xi_dcoeffs(xi).cols(), field.num_dofs());
    EXPECT_TRUE((field.eval_dvalue_dcoeffs(xi) * flattened).isApprox(field.eval_value(xi), 1e-14));
    EXPECT_TRUE((field.eval_dgrad_xi_dcoeffs(xi) * flattened).isApprox(gradient.reshaped(), 1e-14));
    EXPECT_TRUE((field.eval_dhess_xi_dcoeffs(xi) * flattened).isApprox(hessian.reshaped(), 1e-14));
    constexpr double h = 1e-4;
    for (int j = 0; j < field.num_dofs(); ++j) {
        auto plus = coefficients, minus = coefficients;
        plus(j / Components, j % Components) += h;
        minus(j / Components, j % Components) -= h;
        const InterpolatedField3D<HexQ1Basis> a(plus), b(minus);
        const Eigen::VectorXd value_fd = (a.eval_value(xi) - b.eval_value(xi)) / (2 * h);
        const Eigen::MatrixXd gradient_fd = (a.eval_dvalue_dxi(xi) - b.eval_dvalue_dxi(xi)) / (2 * h);
        const Eigen::MatrixXd hessian_fd = (a.eval_d2value_dxi2(xi) - b.eval_d2value_dxi2(xi)) / (2 * h);
        EXPECT_TRUE(value_fd.isApprox(field.eval_dvalue_dcoeffs(xi).col(j), 1e-9));
        EXPECT_TRUE(gradient_fd.reshaped().isApprox(field.eval_dgrad_xi_dcoeffs(xi).col(j), 1e-9));
        EXPECT_TRUE(hessian_fd.reshaped().isApprox(field.eval_dhess_xi_dcoeffs(xi).col(j), 1e-9));
    }
}
} // namespace

TEST(InterpolatedField3D, ValueAndGradientMatricesDifferentiateRowOrderedDofs) {
    check_coefficient_derivatives<1, Eigen::ColMajor>();
    check_coefficient_derivatives<2, Eigen::ColMajor>();
    check_coefficient_derivatives<2, Eigen::RowMajor>();
    check_coefficient_derivatives<3, Eigen::ColMajor>();
    check_coefficient_derivatives<3, Eigen::RowMajor>();
}

TEST(InterpolatedField3D, GeometryAndFieldsUseTheSameParameterInterpolation) {
    Eigen::Matrix3d dX_dxi, A;
    dX_dxi << 2, 0.3, 0.1, 0.2, 1.5, 0.1, 0, 0.2, 1.1;
    A << 0.2, -0.1, 0.3, 0.4, 0.2, -0.3, -0.2, 0.5, 0.1;
    const Eigen::Vector3d origin(1.2, -0.7, 2.1), offset(-0.3, 0.4, 0.7);
    const Eigen::Matrix<double, 8, 3> X_nodes = (corners() * dX_dxi.transpose()).rowwise() + origin.transpose();
    const InterpolatedField3D<HexQ1Basis> X_field(X_nodes);
    const InterpolatedField3D<HexQ1Basis> displacement((X_nodes * A.transpose()).rowwise() + offset.transpose());
    Eigen::Matrix<double, 2, 3> M;
    M << 0.4, -0.1, 0.2, -0.3, 0.5, 0.1;
    const InterpolatedField3D<HexQ1Basis> parameters((X_nodes * M.transpose()).rowwise() + Eigen::RowVector2d(2, 3));
    for (const Eigen::Vector3d xi : {Eigen::Vector3d(0.2, 0.3, 0.4), Eigen::Vector3d(-0.2, 1.2, 0.4)}) {
        EXPECT_TRUE(X_field.eval_value(xi).isApprox(origin + dX_dxi * xi, 1e-13));
        EXPECT_TRUE(X_field.eval_dvalue_dxi(xi).isApprox(dX_dxi, 1e-13));
        EXPECT_TRUE(displacement.eval_value(xi).isApprox(A * X_field.eval_value(xi) + offset, 1e-13));
        EXPECT_TRUE(displacement.eval_dvalue_dxi(xi).isApprox(A * dX_dxi, 1e-13));
        EXPECT_TRUE((displacement.eval_dvalue_dxi(xi) * X_field.eval_dvalue_dxi(xi).inverse()).isApprox(A, 1e-13));
        EXPECT_TRUE(parameters.eval_value(xi).isApprox(M * X_field.eval_value(xi) + Eigen::Vector2d(2, 3), 1e-13));
        EXPECT_TRUE((parameters.eval_dvalue_dxi(xi) * X_field.eval_dvalue_dxi(xi).inverse()).isApprox(M, 1e-13));
    }
}

TEST(InterpolatedField3D, ParameterGradientMatchesFiniteDifferences) {
    Eigen::Matrix<double, 8, 3> coefficients;
    for (int i = 0; i < 8; ++i)
        for (int c = 0; c < 3; ++c)
            coefficients(i, c) = std::sin(0.37 * (3 * i + c + 1));
    const InterpolatedField3D<HexQ1Basis> field(coefficients);
    const Eigen::Vector3d xi(0.2, 0.3, 0.4);
    constexpr double h = 1e-6;
    for (int d = 0; d < 3; ++d) {
        Eigen::Vector3d plus = xi, minus = xi;
        plus[d] += h;
        minus[d] -= h;
        EXPECT_TRUE(((field.eval_value(plus) - field.eval_value(minus)) / (2 * h))
                        .isApprox(field.eval_dvalue_dxi(xi).col(d), 3e-9));
        const Eigen::Matrix3d second_fd = (field.eval_dvalue_dxi(plus) - field.eval_dvalue_dxi(minus)) / (2 * h);
        constexpr int columns[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}};
        for (int e = 0; e < 3; ++e)
            EXPECT_LT((second_fd.col(e) - field.eval_d2value_dxi2(xi).col(columns[d][e])).norm(), 2e-9);
    }
}

TEST(InterpolatedField3D, CoefficientsAreCopiedAndTranslationHasZeroGradient) {
    const Eigen::Vector3d xi(0.2, 0.3, 0.4), translation(0.7, -0.2, 1.3);
    Eigen::Matrix<double, 8, 3, Eigen::RowMajor> coefficients = translation.transpose().replicate<8, 1>();
    const InterpolatedField3D<HexQ1Basis> field(coefficients);
    coefficients.setConstant(std::numeric_limits<double>::quiet_NaN());
    EXPECT_TRUE(field.eval_value(xi).isApprox(translation, 1e-14));
    EXPECT_TRUE(field.eval_dvalue_dxi(xi).isZero(1e-14));
    EXPECT_EQ(field.num_dofs(), 24);
    EXPECT_EQ(field.num_components(), 3);
}

TEST(InterpolatedField3D, ValueAndGradientValidateIndependently) {
    const Eigen::Vector3d xi(0.2, 0.3, 0.4);
    const InterpolatedField3D<InvalidBasis<0>> bad_values(corners());
    EXPECT_THROW(bad_values.eval_value(xi), std::invalid_argument);
    EXPECT_THROW(bad_values.eval_dvalue_dcoeffs(xi), std::invalid_argument);
    EXPECT_TRUE(bad_values.eval_dvalue_dxi(xi).isIdentity(1e-14));
    EXPECT_TRUE(bad_values.eval_dgrad_xi_dcoeffs(xi).allFinite());
    EXPECT_TRUE(bad_values.eval_d2value_dxi2(xi).allFinite());
    EXPECT_TRUE(bad_values.eval_dhess_xi_dcoeffs(xi).allFinite());
    const InterpolatedField3D<InvalidBasis<1>> bad_gradients(corners());
    EXPECT_THROW(bad_gradients.eval_dvalue_dxi(xi), std::invalid_argument);
    EXPECT_THROW(bad_gradients.eval_dgrad_xi_dcoeffs(xi), std::invalid_argument);
    EXPECT_TRUE(bad_gradients.eval_value(xi).isApprox(xi, 1e-14));
    EXPECT_TRUE(bad_gradients.eval_dvalue_dcoeffs(xi).allFinite());
    EXPECT_TRUE(bad_gradients.eval_d2value_dxi2(xi).allFinite());
    EXPECT_TRUE(bad_gradients.eval_dhess_xi_dcoeffs(xi).allFinite());
    const InterpolatedField3D<InvalidBasis<2>> bad_hessians(corners());
    EXPECT_THROW(bad_hessians.eval_d2value_dxi2(xi), std::invalid_argument);
    EXPECT_THROW(bad_hessians.eval_dhess_xi_dcoeffs(xi), std::invalid_argument);
    EXPECT_TRUE(bad_hessians.eval_value(xi).allFinite());
    EXPECT_TRUE(bad_hessians.eval_dvalue_dxi(xi).allFinite());
}

TEST(InterpolatedField3D, RejectsIncompatibleCoefficientsAndNonFiniteInputsOrResults) {
    for (int rows : {0, 7, 9})
        EXPECT_THROW(InterpolatedField3D<HexQ1Basis>(Eigen::MatrixXd::Zero(rows, 3)), std::invalid_argument);
    EXPECT_THROW(InterpolatedField3D<HexQ1Basis>(Eigen::MatrixXd(8, 0)), std::invalid_argument);
    const InterpolatedField3D<HexQ1Basis> field(corners());
    for (double bad : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(),
                       -std::numeric_limits<double>::infinity()}) {
        auto coefficients = corners();
        coefficients(1, 1) = bad;
        EXPECT_THROW((InterpolatedField3D<HexQ1Basis>(coefficients)), std::invalid_argument);
        const Eigen::Vector3d xi(bad, 0.3, 0.4);
        EXPECT_THROW(field.eval_value(xi), std::invalid_argument);
        EXPECT_THROW(field.eval_dvalue_dxi(xi), std::invalid_argument);
        EXPECT_THROW(field.eval_dvalue_dcoeffs(xi), std::invalid_argument);
        EXPECT_THROW(field.eval_dgrad_xi_dcoeffs(xi), std::invalid_argument);
        EXPECT_THROW(field.eval_d2value_dxi2(xi), std::invalid_argument);
        EXPECT_THROW(field.eval_dhess_xi_dcoeffs(xi), std::invalid_argument);
    }
    const InterpolatedField3D<HexQ1Basis> enormous(Eigen::MatrixXd::Constant(8, 3, std::numeric_limits<double>::max()));
    EXPECT_THROW(enormous.eval_value(Eigen::Vector3d::Constant(2)), std::domain_error);
    EXPECT_THROW(enormous.eval_dvalue_dxi(Eigen::Vector3d::Constant(2)), std::domain_error);
    EXPECT_THROW(enormous.eval_d2value_dxi2(Eigen::Vector3d::Constant(2)), std::domain_error);
    EXPECT_TRUE(enormous.eval_dvalue_dcoeffs(Eigen::Vector3d::Constant(2)).allFinite());
    EXPECT_TRUE(enormous.eval_dgrad_xi_dcoeffs(Eigen::Vector3d::Constant(2)).allFinite());
    EXPECT_TRUE(enormous.eval_dhess_xi_dcoeffs(Eigen::Vector3d::Constant(2)).allFinite());
}

TEST(InterpolatedField3D, LinearTetFieldsReturnZeroSecondSpatialAndCoefficientDerivatives) {
    const InterpolatedField3D<TetP1Basis> concrete(Eigen::Matrix<double, 4, 3>::Random());
    const Field3D& field = concrete;
    const Eigen::Vector3d xi(0.2, 0.3, 0.1);
    EXPECT_EQ(field.eval_d2value_dxi2(xi).rows(), 3);
    EXPECT_EQ(field.eval_d2value_dxi2(xi).cols(), 6);
    EXPECT_TRUE(field.eval_d2value_dxi2(xi).isZero());
    EXPECT_EQ(field.eval_dhess_xi_dcoeffs(xi).rows(), 18);
    EXPECT_EQ(field.eval_dhess_xi_dcoeffs(xi).cols(), 12);
    EXPECT_TRUE(field.eval_dhess_xi_dcoeffs(xi).isZero());
}
