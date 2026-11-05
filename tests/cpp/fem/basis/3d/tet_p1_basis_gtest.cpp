#include "litepgo/fem/basis/3d/tet_p1_basis.hpp"
#include "litepgo/fem/basis/basis.hpp"

#include <gtest/gtest.h>

#include <array>
#include <stdexcept>

namespace litepgo::fem::test {

namespace {
static_assert(litepgo::fem::Basis3D<TetP1Basis>);

const std::array<Eigen::Vector3d, 4> samples{{Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0.25, 0.25, 0.25),
                                              Eigen::Vector3d(0.2, 0.3, 0.1), Eigen::Vector3d(1.2, -0.1, 0.4)}};
} // namespace

TEST(TetP1Basis, InterpolatesReferenceNodes) {
    Eigen::Matrix<double, 4, 3> nodes;
    nodes << 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1;
    for (int a = 0; a < 4; ++a) {
        EXPECT_EQ(TetP1Basis::coefficient_info(a).point, nodes.row(a).transpose());
        const auto N = TetP1Basis::eval_N(nodes.row(a).transpose());
        EXPECT_EQ(N, Eigen::Vector4d::Unit(a));
    }
    EXPECT_THROW(TetP1Basis::coefficient_info(-1), std::out_of_range);
    EXPECT_THROW(TetP1Basis::coefficient_info(4), std::out_of_range);
}

TEST(TetP1Basis, ReproducesAffineFunctionsAndPartitionsUnity) {
    Eigen::Matrix<double, 4, 3> nodes;
    for (int a = 0; a < 4; ++a)
        nodes.row(a) = TetP1Basis::coefficient_info(a).point.transpose();
    const Eigen::Vector3d slope(0.3, -0.7, 0.2);
    const Eigen::Vector4d coefficients = (nodes * slope).array() + 0.4;
    for (const auto& xi : samples) {
        const auto N = TetP1Basis::eval_N(xi);
        const auto dN_dxi = TetP1Basis::eval_dN_dxi(xi);
        const auto d2N_dxi2 = TetP1Basis::eval_d2N_dxi2(xi);
        EXPECT_NEAR(N.sum(), 1, 1e-15);
        EXPECT_TRUE(dN_dxi.colwise().sum().isZero());
        EXPECT_TRUE(d2N_dxi2.isZero());
        EXPECT_NEAR(coefficients.dot(N), 0.4 + slope.dot(xi), 1e-15);
        EXPECT_TRUE((dN_dxi.transpose() * coefficients).isApprox(slope, 1e-14));
    }
}

TEST(TetP1Basis, GradientsMatchFiniteDifferences) {
    constexpr double h = 1e-6;
    for (const auto& xi : samples) {
        const auto dN_dxi = TetP1Basis::eval_dN_dxi(xi);
        for (int d = 0; d < 3; ++d) {
            Eigen::Vector3d plus = xi, minus = xi;
            plus[d] += h;
            minus[d] -= h;
            const Eigen::Vector4d fd = (TetP1Basis::eval_N(plus) - TetP1Basis::eval_N(minus)) / (2 * h);
            EXPECT_TRUE((fd - dN_dxi.col(d)).isZero(3e-10));
            EXPECT_TRUE((TetP1Basis::eval_dN_dxi(plus) - TetP1Basis::eval_dN_dxi(minus)).isZero());
        }
    }
}

} // namespace litepgo::fem::test
