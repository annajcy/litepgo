#include "litepgo/fem/quadrature/quadrature.hpp"
#include "litepgo/fem/quadrature/3d/tet_centroid_quadrature.hpp"

#include <gtest/gtest.h>

static_assert(litepgo::Quadrature3D<litepgo::TetCentroidQuadrature>);

TEST(TetCentroidQuadrature, IntegratesConstantAndLinearFunctions) {
    const auto& points = litepgo::TetCentroidQuadrature::points();
    const auto& weights = litepgo::TetCentroidQuadrature::weights();
    EXPECT_EQ(points.rows(), 1);
    EXPECT_GT(points.minCoeff(), 0);
    EXPECT_LT(points.row(0).sum(), 1);
    EXPECT_DOUBLE_EQ(weights.sum(), 1.0 / 6);
    for (int d = 0; d < 3; ++d)
        EXPECT_DOUBLE_EQ(weights.dot(points.col(d)), 1.0 / 24);
}
