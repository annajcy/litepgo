#include "litepgo/fem/basis/1d/cubic_hermite_basis.hpp"
#include "litepgo/fem/basis/1d/linear_lagrange_basis.hpp"
#include "litepgo/fem/basis/2d/tensor_product_basis_2d.hpp"
#include "litepgo/fem/field/interpolated_field.hpp"

#include <gtest/gtest.h>

namespace {
using namespace litepgo;
using B = TensorProductBasis2D<LinearLagrangeBasis>;
static_assert(Basis2D<B>);
static_assert(!Basis3D<B>);
static_assert(Basis2D<TensorProductBasis2D<CubicHermiteBasis>>);

TEST(TensorProductBasis2D, InterpolatesInLastAxisFastestOrderAndReproducesBilinears) {
    Eigen::Matrix<double, 4, 2> nodes;
    nodes << 0, 0, 0, 1, 1, 0, 1, 1;
    Eigen::Vector4d coefficients;
    for (int a = 0; a < 4; ++a) {
        EXPECT_EQ(B::eval_N(nodes.row(a).transpose()), Eigen::Vector4d::Unit(a));
        coefficients[a] = 1 + 2 * nodes(a, 0) - 3 * nodes(a, 1) + 4 * nodes(a, 0) * nodes(a, 1);
    }
    for (const Eigen::Vector2d xi : {Eigen::Vector2d(0.2, 0.7), Eigen::Vector2d(-0.2, 1.3)}) {
        const auto N = B::eval_N(xi);
        const auto dN_dxi = B::eval_dN_dxi(xi);
        const auto d2N_dxi2 = B::eval_d2N_dxi2(xi);
        EXPECT_NEAR(N.sum(), 1, 1e-14);
        EXPECT_TRUE(dN_dxi.colwise().sum().isZero(1e-14));
        EXPECT_NEAR(coefficients.dot(N), 1 + 2 * xi.x() - 3 * xi.y() + 4 * xi.x() * xi.y(), 1e-14);
        EXPECT_NEAR(coefficients.dot(dN_dxi.col(0)), 2 + 4 * xi.y(), 1e-14);
        EXPECT_NEAR(coefficients.dot(dN_dxi.col(1)), -3 + 4 * xi.x(), 1e-14);
        EXPECT_DOUBLE_EQ(coefficients.dot(d2N_dxi2.col(0)), 0);
        EXPECT_DOUBLE_EQ(coefficients.dot(d2N_dxi2.col(1)), 4);
        EXPECT_DOUBLE_EQ(coefficients.dot(d2N_dxi2.col(2)), 0);
        constexpr double h = 1e-6;
        for (int d = 0; d < 2; ++d) {
            Eigen::Vector2d plus = xi, minus = xi;
            plus[d] += h;
            minus[d] -= h;
            const Eigen::Matrix<double, 4, 2> fd = (B::eval_dN_dxi(plus) - B::eval_dN_dxi(minus)) / (2 * h);
            for (int e = 0; e < 2; ++e)
                EXPECT_TRUE((fd.col(e) - d2N_dxi2.col(d + e)).isZero(3e-10));
        }
    }
}

TEST(TensorProductBasis2D, BilinearFieldsProvideZeroPureAndNonzeroMixedSecondDerivatives) {
    const InterpolatedField2D<B> concrete(Eigen::Vector4d(1, 2, 3, 7));
    const Field2D& field = concrete;
    const Eigen::Vector2d xi(0.2, 0.3);
    EXPECT_EQ(field.num_dofs(), 4);
    EXPECT_EQ(field.eval_dvalue_dxi(xi).cols(), 2);
    EXPECT_TRUE(field.eval_value(xi).allFinite());
    EXPECT_TRUE(field.eval_dgrad_xi_dcoeffs(xi).allFinite());
    EXPECT_EQ(field.eval_d2value_dxi2(xi), Eigen::RowVector3d(0, 3, 0));
    Eigen::Matrix<double, 3, 4> expected = Eigen::Matrix<double, 3, 4>::Zero();
    expected.row(1) << 1, -1, -1, 1;
    EXPECT_EQ(field.eval_dhess_xi_dcoeffs(xi), expected);
}
} // namespace
