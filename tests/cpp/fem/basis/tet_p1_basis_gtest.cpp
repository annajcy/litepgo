#include "litepgo/fem/basis/basis.hpp"
#include "litepgo/fem/basis/tet_p1_basis.hpp"

#include <gtest/gtest.h>

#include <array>
#include <stdexcept>

namespace {
using litepgo::TetP1Basis;
static_assert(litepgo::Basis3D<TetP1Basis>);

const std::array<Eigen::Vector3d, 4> samples{{Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0.25, 0.25, 0.25),
                                              Eigen::Vector3d(0.2, 0.3, 0.1), Eigen::Vector3d(1.2, -0.1, 0.4)}};
} // namespace

TEST(TetP1Basis, InterpolatesReferenceNodes) {
    Eigen::Matrix<double, 4, 3> nodes;
    nodes << 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1;
    for (int a = 0; a < 4; ++a) {
        EXPECT_EQ(TetP1Basis::reference_node(a), nodes.row(a).transpose());
        const auto sample = TetP1Basis::evaluate(nodes.row(a).transpose());
        EXPECT_EQ(sample.N, Eigen::Vector4d::Unit(a));
    }
    EXPECT_THROW(TetP1Basis::reference_node(-1), std::out_of_range);
    EXPECT_THROW(TetP1Basis::reference_node(4), std::out_of_range);
}

TEST(TetP1Basis, ReproducesAffineFunctionsAndPartitionsUnity) {
    Eigen::Matrix<double, 4, 3> nodes;
    for (int a = 0; a < 4; ++a)
        nodes.row(a) = TetP1Basis::reference_node(a).transpose();
    const Eigen::Vector3d slope(0.3, -0.7, 0.2);
    const Eigen::Vector4d coefficients = (nodes * slope).array() + 0.4;
    for (const auto& xi : samples) {
        const auto sample = TetP1Basis::evaluate(xi);
        EXPECT_NEAR(sample.N.sum(), 1, 1e-15);
        EXPECT_TRUE(sample.dN_dxi.colwise().sum().isZero());
        EXPECT_NEAR(coefficients.dot(sample.N), 0.4 + slope.dot(xi), 1e-15);
        EXPECT_TRUE((sample.dN_dxi.transpose() * coefficients).isApprox(slope, 1e-14));
    }
}

TEST(TetP1Basis, GradientsMatchFiniteDifferences) {
    constexpr double h = 1e-6;
    for (const auto& xi : samples) {
        const auto sample = TetP1Basis::evaluate(xi);
        for (int d = 0; d < 3; ++d) {
            Eigen::Vector3d plus = xi, minus = xi;
            plus[d] += h;
            minus[d] -= h;
            const Eigen::Vector4d fd = (TetP1Basis::evaluate(plus).N - TetP1Basis::evaluate(minus).N) / (2 * h);
            EXPECT_TRUE((fd - sample.dN_dxi.col(d)).isZero(3e-10));
        }
    }
}
