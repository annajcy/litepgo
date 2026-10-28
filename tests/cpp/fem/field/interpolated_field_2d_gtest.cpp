#include "litepgo/dof/vertex_dof_map.hpp"
#include "litepgo/fem/basis/2d/quad_hermite_basis.hpp"
#include "litepgo/fem/field/interpolated_field.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <type_traits>

namespace {
using namespace litepgo;
using F = InterpolatedField2D<QuadHermiteBasis>;
static_assert(std::is_same_v<F, InterpolatedFieldT<QuadHermiteBasis, 2>>);
static_assert(std::is_base_of_v<Field2D, F>);
static_assert(std::is_same_v<Field3D, FieldT<3>>);
static_assert(std::is_same_v<Field2D::Coordinate, Eigen::Vector2d>);
static_assert(std::is_same_v<decltype(std::declval<const F&>().eval_dvalue_dxi(Eigen::Vector2d::Zero())),
                             Eigen::Matrix<double, Eigen::Dynamic, 2>>);

template <int Components, int Storage> void check_derivatives() {
    SCOPED_TRACE(Components);
    SCOPED_TRACE(Storage);
    Eigen::Matrix<double, 16, Components, Storage> coefficients;
    for (int a = 0; a < 16; ++a)
        for (int c = 0; c < Components; ++c)
            coefficients(a, c) = std::sin(0.37 * (a * Components + c + 1));
    const F concrete(coefficients);
    const Field2D& field = concrete;
    const Eigen::Vector2d xi(0.23, 0.61);
    const Eigen::VectorXd flattened = coefficients.template reshaped<Eigen::RowMajor>();
    const auto J = field.eval_dvalue_dcoeffs(xi);
    const auto G = field.eval_dgrad_xi_dcoeffs(xi);
    const auto H = field.eval_dhess_xi_dcoeffs(xi);
    EXPECT_EQ(field.num_components(), Components);
    EXPECT_EQ(field.num_dofs(), 16 * Components);
    EXPECT_EQ(G.rows(), 2 * Components);
    EXPECT_EQ(H.rows(), 3 * Components);
    EXPECT_EQ(H.cols(), field.num_dofs());
    EXPECT_TRUE((J * flattened).isApprox(field.eval_value(xi), 1e-13));
    EXPECT_TRUE((G * flattened).isApprox(field.eval_dvalue_dxi(xi).reshaped(), 1e-13));
    EXPECT_TRUE((H * flattened).isApprox(field.eval_d2value_dxi2(xi).reshaped(), 1e-13));

    constexpr double coefficient_step = 1e-4;
    for (int j = 0; j < field.num_dofs(); ++j) {
        auto plus = coefficients, minus = coefficients;
        plus(j / Components, j % Components) += coefficient_step;
        minus(j / Components, j % Components) -= coefficient_step;
        const F a(plus), b(minus);
        const Eigen::VectorXd value_fd = (a.eval_value(xi) - b.eval_value(xi)) / (2 * coefficient_step);
        const Eigen::MatrixXd grad_fd = (a.eval_dvalue_dxi(xi) - b.eval_dvalue_dxi(xi)) / (2 * coefficient_step);
        const Eigen::MatrixXd hess_fd = (a.eval_d2value_dxi2(xi) - b.eval_d2value_dxi2(xi)) / (2 * coefficient_step);
        EXPECT_LT((value_fd - J.col(j)).norm(), 1e-10);
        EXPECT_LT((grad_fd.reshaped() - G.col(j)).norm(), 1e-10);
        EXPECT_LT((hess_fd.reshaped() - H.col(j)).norm(), 1e-10);
    }
    constexpr double spatial_step = 1e-6;
    for (int d = 0; d < 2; ++d) {
        Eigen::Vector2d plus = xi, minus = xi;
        plus[d] += spatial_step;
        minus[d] -= spatial_step;
        EXPECT_LT(
            ((field.eval_value(plus) - field.eval_value(minus)) / (2 * spatial_step) - field.eval_dvalue_dxi(xi).col(d))
                .norm(),
            2e-9);
        for (int e = 0; e < 2; ++e)
            EXPECT_LT(((field.eval_dvalue_dxi(plus).col(e) - field.eval_dvalue_dxi(minus).col(e)) / (2 * spatial_step) -
                       field.eval_d2value_dxi2(xi).col(d + e))
                          .norm(),
                      3e-9);
    }
}

TEST(InterpolatedField2D, ValuesAndAllDerivativesUseRowOrderedCoefficientsAndColumnOrderedOutputs) {
    check_derivatives<1, Eigen::ColMajor>();
    check_derivatives<2, Eigen::ColMajor>();
    check_derivatives<2, Eigen::RowMajor>();
    check_derivatives<3, Eigen::ColMajor>();
    check_derivatives<3, Eigen::RowMajor>();
}

template <int InvalidPart> struct InvalidBasis {
    InvalidBasis() = delete;
    static constexpr int num_basis_functions = 16;
    static auto eval_N(const Eigen::Vector2d& xi) {
        auto result = QuadHermiteBasis::eval_N(xi);
        if constexpr (InvalidPart == 0)
            result.setConstant(std::numeric_limits<double>::quiet_NaN());
        return result;
    }
    static auto eval_dN_dxi(const Eigen::Vector2d& xi) {
        auto result = QuadHermiteBasis::eval_dN_dxi(xi);
        if constexpr (InvalidPart == 1)
            result.setConstant(std::numeric_limits<double>::infinity());
        return result;
    }
    static auto eval_d2N_dxi2(const Eigen::Vector2d& xi) {
        auto result = QuadHermiteBasis::eval_d2N_dxi2(xi);
        if constexpr (InvalidPart == 2)
            result.setConstant(std::numeric_limits<double>::infinity());
        return result;
    }
};

TEST(InterpolatedField2D, SecondDerivativesValidateOnlyTheirOwnBasisData) {
    const Eigen::Vector2d xi(0.2, 0.3);
    const Eigen::MatrixXd coefficients = Eigen::MatrixXd::Ones(16, 3);
    const InterpolatedField2D<InvalidBasis<0>> bad_values(coefficients);
    EXPECT_THROW(bad_values.eval_value(xi), std::invalid_argument);
    EXPECT_THROW(bad_values.eval_dvalue_dcoeffs(xi), std::invalid_argument);
    EXPECT_TRUE(bad_values.eval_d2value_dxi2(xi).allFinite());
    EXPECT_TRUE(bad_values.eval_dhess_xi_dcoeffs(xi).allFinite());
    const InterpolatedField2D<InvalidBasis<1>> bad_gradients(coefficients);
    EXPECT_THROW(bad_gradients.eval_dvalue_dxi(xi), std::invalid_argument);
    EXPECT_THROW(bad_gradients.eval_dgrad_xi_dcoeffs(xi), std::invalid_argument);
    EXPECT_TRUE(bad_gradients.eval_d2value_dxi2(xi).allFinite());
    EXPECT_TRUE(bad_gradients.eval_dhess_xi_dcoeffs(xi).allFinite());
    const InterpolatedField2D<InvalidBasis<2>> bad_hessians(coefficients);
    EXPECT_THROW(bad_hessians.eval_d2value_dxi2(xi), std::invalid_argument);
    EXPECT_THROW(bad_hessians.eval_dhess_xi_dcoeffs(xi), std::invalid_argument);
    EXPECT_TRUE(bad_hessians.eval_value(xi).allFinite());
    EXPECT_TRUE(bad_hessians.eval_dvalue_dxi(xi).allFinite());
}

TEST(InterpolatedField2D, RejectsInvalidCoefficientsCoordinatesAndOverflow) {
    EXPECT_THROW(F(Eigen::MatrixXd::Zero(15, 3)), std::invalid_argument);
    EXPECT_THROW(F(Eigen::MatrixXd(16, 0)), std::invalid_argument);
    Eigen::MatrixXd coefficients = Eigen::MatrixXd::Ones(16, 3);
    const F field(coefficients);
    coefficients(0, 0) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(F{coefficients}, std::invalid_argument);
    for (double bad : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
        const Eigen::Vector2d xi(0.2, bad);
        EXPECT_THROW(field.eval_value(xi), std::invalid_argument);
        EXPECT_THROW(field.eval_dvalue_dcoeffs(xi), std::invalid_argument);
        EXPECT_THROW(field.eval_dvalue_dxi(xi), std::invalid_argument);
        EXPECT_THROW(field.eval_dgrad_xi_dcoeffs(xi), std::invalid_argument);
        EXPECT_THROW(field.eval_d2value_dxi2(xi), std::invalid_argument);
        EXPECT_THROW(field.eval_dhess_xi_dcoeffs(xi), std::invalid_argument);
    }
    const F enormous(Eigen::MatrixXd::Constant(16, 3, std::numeric_limits<double>::max()));
    EXPECT_THROW(enormous.eval_d2value_dxi2(Eigen::Vector2d(2, 2)), std::invalid_argument);
    EXPECT_TRUE(enormous.eval_dhess_xi_dcoeffs(Eigen::Vector2d(2, 2)).allFinite());
}

TEST(InterpolatedField2D, SharedHermiteDofsPreserveC1AcrossUnequalParameterSpans) {
    const QuadTopology topology(6, {{0, 1, 2, 3}, {1, 4, 5, 2}});
    const VertexDofMap map(topology, 12);
    Vec dofs(map.num_dofs());
    for (int j = 0; j < dofs.size(); ++j)
        dofs[j] = std::sin(0.13 * (j + 1));
    const auto coefficients = [&](int element, double hx) {
        const Eigen::MatrixXd gathered = map.gather(element, dofs);
        Eigen::MatrixXd result = gathered.reshaped<Eigen::RowMajor>(16, 3);
        const std::array<double, 4> scales{1, hx, 2, 2 * hx};
        for (int a = 0; a < 16; ++a)
            result.row(a) *= scales[static_cast<std::size_t>(a % 4)];
        return result;
    };
    const F left(coefficients(0, 1)), right(coefficients(1, 3));
    for (double t : {0., 0.17, 0.5, 0.83, 1.}) {
        const Eigen::Vector2d a(1, t), b(0, t);
        EXPECT_TRUE(left.eval_value(a).isApprox(right.eval_value(b), 1e-13));
        EXPECT_TRUE(left.eval_dvalue_dxi(a).col(0).isApprox(right.eval_dvalue_dxi(b).col(0) / 3, 1e-13));
        EXPECT_TRUE(left.eval_dvalue_dxi(a).col(1).isApprox(right.eval_dvalue_dxi(b).col(1), 1e-13));
    }
}
} // namespace
