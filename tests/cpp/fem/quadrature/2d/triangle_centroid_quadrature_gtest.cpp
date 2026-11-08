#include "litepgo/fem/basis/2d/triangle_p1_basis.hpp"
#include "litepgo/fem/quadrature/2d/triangle_centroid_quadrature.hpp"
#include "litepgo/fem/quadrature/quadrature.hpp"
#include "litepgo/fem/response/interpolated_response.hpp"
#include <memory>

#include <Eigen/Geometry>
#include <gtest/gtest.h>

namespace litepgo::fem::test {

namespace {
static_assert(Quadrature2D<TriangleCentroidQuadrature>);
static_assert(!Quadrature3D<TriangleCentroidQuadrature>);

TEST(TriangleCentroidQuadrature, IntegratesConstantAndLinearFunctions) {
    const auto& points = TriangleCentroidQuadrature::points();
    const auto& weights = TriangleCentroidQuadrature::weights();
    EXPECT_EQ(points.rows(), 1);
    EXPECT_GT(points.minCoeff(), 0);
    EXPECT_LT(points.row(0).sum(), 1);
    EXPECT_DOUBLE_EQ(weights.sum(), 0.5);
    EXPECT_GT(weights.minCoeff(), 0);
    for (int d = 0; d < 2; ++d)
        EXPECT_DOUBLE_EQ(weights.dot(points.col(d)), 1.0 / 6);
    const Eigen::Vector3d barycentric = TriangleP1Basis::eval_N(points.row(0).transpose());
    EXPECT_TRUE(barycentric.isApprox(Eigen::Vector3d::Constant(1.0 / 3), 1e-15));
}

TEST(TriangleCentroidQuadrature, IntegratesAffineFieldsOnEmbeddedTrianglesUsingReferenceArea) {
    Eigen::Matrix<double, 3, 3> coordinates;
    coordinates << 1, -2, 0.5, 3, -2, 0.5, 1, 1, 4.5;
    const Eigen::Vector3d slope(0.3, -0.7, 0.2);
    for (bool reverse_winding : {false, true}) {
        SCOPED_TRACE(reverse_winding);
        if (reverse_winding)
            coordinates.row(1).swap(coordinates.row(2));
        auto X_field = std::make_shared<InterpolatedResponse<TriangleP1Basis>>(coordinates);
        const Eigen::Vector3d nodal_values = (coordinates * slope).array() + 0.4;
        auto scalar_field = std::make_shared<InterpolatedResponse<TriangleP1Basis>>(nodal_values);
        double area = 0, integral = 0;
        for (int q = 0; q < TriangleCentroidQuadrature::num_points; ++q) {
            const Eigen::Vector2d xi = TriangleCentroidQuadrature::points().row(q).transpose();
            const Eigen::Matrix<double, 3, 2> J = X_field->eval_jacobian(xi);
            const double weight = TriangleCentroidQuadrature::weights()[q] * J.col(0).cross(J.col(1)).norm();
            area += weight;
            integral += weight * scalar_field->eval_value(xi)[0];
        }
        EXPECT_NEAR(area, 5, 1e-14);
        EXPECT_NEAR(integral, 5 * nodal_values.mean(), 1e-14);
    }
}
} // namespace

} // namespace litepgo::fem::test
