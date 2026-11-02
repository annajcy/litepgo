#include "litepgo/fem/basis/2d/triangle_p1_basis.hpp"
#include "litepgo/fem/basis/basis.hpp"

#include <gtest/gtest.h>

#include <array>
#include <stdexcept>

namespace litepgo::fem::test {

namespace {
static_assert(litepgo::fem::Basis2D<TriangleP1Basis>);
static_assert(!litepgo::fem::Basis3D<TriangleP1Basis>);

const std::array<Eigen::Vector2d, 5> samples{{Eigen::Vector2d(0, 0), Eigen::Vector2d(1.0 / 3, 1.0 / 3),
                                              Eigen::Vector2d(0.2, 0.3), Eigen::Vector2d(0.4, 0.6),
                                              Eigen::Vector2d(1.2, -0.1)}};
} // namespace

TEST(TriangleP1Basis, InterpolatesReferenceNodes) {
    Eigen::Matrix<double, 3, 2> nodes;
    nodes << 0, 0, 1, 0, 0, 1;
    for (int a = 0; a < 3; ++a) {
        EXPECT_EQ(TriangleP1Basis::reference_node(a), nodes.row(a).transpose());
        EXPECT_EQ(TriangleP1Basis::eval_N(nodes.row(a).transpose()), Eigen::Vector3d::Unit(a));
    }
    EXPECT_THROW(TriangleP1Basis::reference_node(-1), std::out_of_range);
    EXPECT_THROW(TriangleP1Basis::reference_node(3), std::out_of_range);
}

TEST(TriangleP1Basis, ReproducesAffineFunctionsAndPartitionsUnity) {
    Eigen::Matrix<double, 3, 2> nodes;
    for (int a = 0; a < 3; ++a)
        nodes.row(a) = TriangleP1Basis::reference_node(a).transpose();
    const Eigen::Vector2d slope(0.3, -0.7);
    const Eigen::Vector3d coefficients = (nodes * slope).array() + 0.4;
    for (const auto& xi : samples) {
        const auto N = TriangleP1Basis::eval_N(xi);
        const auto dN_dxi = TriangleP1Basis::eval_dN_dxi(xi);
        const auto d2N_dxi2 = TriangleP1Basis::eval_d2N_dxi2(xi);
        EXPECT_NEAR(N.sum(), 1, 1e-15);
        EXPECT_TRUE(dN_dxi.colwise().sum().isZero());
        EXPECT_TRUE(d2N_dxi2.isZero());
        EXPECT_NEAR(coefficients.dot(N), 0.4 + slope.dot(xi), 1e-15);
        EXPECT_TRUE((dN_dxi.transpose() * coefficients).isApprox(slope, 1e-14));
    }
}

TEST(TriangleP1Basis, FirstAndSecondDerivativesMatchFiniteDifferences) {
    constexpr double h = 1e-6;
    for (const auto& xi : samples) {
        const auto dN_dxi = TriangleP1Basis::eval_dN_dxi(xi);
        const auto d2N_dxi2 = TriangleP1Basis::eval_d2N_dxi2(xi);
        for (int d = 0; d < 2; ++d) {
            Eigen::Vector2d plus = xi, minus = xi;
            plus[d] += h;
            minus[d] -= h;
            const Eigen::Vector3d fd = (TriangleP1Basis::eval_N(plus) - TriangleP1Basis::eval_N(minus)) / (2 * h);
            EXPECT_TRUE((fd - dN_dxi.col(d)).isZero(3e-10));
            const Eigen::Matrix<double, 3, 2> gradient_fd =
                (TriangleP1Basis::eval_dN_dxi(plus) - TriangleP1Basis::eval_dN_dxi(minus)) / (2 * h);
            for (int e = 0; e < 2; ++e)
                EXPECT_TRUE((gradient_fd.col(e) - d2N_dxi2.col(d + e)).isZero());
        }
    }
}

} // namespace litepgo::fem::test
