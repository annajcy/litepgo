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
static_assert(!SecondOrderBasis<B, 2>);
static_assert(SecondOrderBasis<TensorProductBasis2D<CubicHermiteBasis>, 2>);

TEST(TensorProductBasis2D, InterpolatesInLastAxisFastestOrderAndReproducesBilinears) {
    Eigen::Matrix<double, 4, 2> nodes;
    nodes << 0, 0, 0, 1, 1, 0, 1, 1;
    Eigen::Vector4d coefficients;
    for (int a = 0; a < 4; ++a) {
        EXPECT_EQ(B::evaluate(nodes.row(a).transpose()).N, Eigen::Vector4d::Unit(a));
        coefficients[a] = 1 + 2 * nodes(a, 0) - 3 * nodes(a, 1) + 4 * nodes(a, 0) * nodes(a, 1);
    }
    for (const Eigen::Vector2d xi : {Eigen::Vector2d(0.2, 0.7), Eigen::Vector2d(-0.2, 1.3)}) {
        const auto sample = B::evaluate(xi);
        EXPECT_NEAR(sample.N.sum(), 1, 1e-14);
        EXPECT_TRUE(sample.dN_dxi.colwise().sum().isZero(1e-14));
        EXPECT_NEAR(coefficients.dot(sample.N), 1 + 2 * xi.x() - 3 * xi.y() + 4 * xi.x() * xi.y(), 1e-14);
        EXPECT_NEAR(coefficients.dot(sample.dN_dxi.col(0)), 2 + 4 * xi.y(), 1e-14);
        EXPECT_NEAR(coefficients.dot(sample.dN_dxi.col(1)), -3 + 4 * xi.x(), 1e-14);
    }
}

TEST(TensorProductBasis2D, FirstOrderOnlyBasesRemainUsableAndRejectUnsupportedHessians) {
    const InterpolatedField2D<B> concrete(Eigen::Vector4d(1, 2, 3, 4));
    const Field2D& field = concrete;
    const Eigen::Vector2d xi(0.2, 0.3);
    EXPECT_EQ(field.num_dofs(), 4);
    EXPECT_EQ(field.dvalue_dxi(xi).cols(), 2);
    EXPECT_TRUE(field.value(xi).allFinite());
    EXPECT_TRUE(field.dgrad_xi_dcoeffs(xi).allFinite());
    EXPECT_THROW(field.d2value_dxi2(xi), std::logic_error);
    EXPECT_THROW(field.dhess_xi_dcoeffs(xi), std::logic_error);
}
} // namespace
