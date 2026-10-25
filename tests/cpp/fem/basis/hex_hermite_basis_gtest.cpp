#include "litepgo/fem/basis/hex_hermite_basis.hpp"
#include "litepgo/fem/geometry/interpolation.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <stdexcept>

namespace {
using litepgo::HexHermiteBasis;
static_assert(litepgo::Basis3D<HexHermiteBasis>);
static_assert(HexHermiteBasis::num_basis_functions == 64);

Eigen::Matrix<double, 8, 3> corners() {
    Eigen::Matrix<double, 8, 3> result;
    result << 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1;
    return result;
}

const std::array<Eigen::Vector3d, 6> samples{{Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(1, 1, 1),
                                              Eigen::Vector3d(0.5, 0.5, 0.5), Eigen::Vector3d(0, 0.3, 0.7),
                                              Eigen::Vector3d(0.2, 0.3, 0.4), Eigen::Vector3d(1.2, -0.1, 0.8)}};

// The specified vertex slots, independent of the implementation's tensor permutation.
const std::array<Eigen::Vector3i, 8> derivatives{
    {Eigen::Vector3i(0, 0, 0), Eigen::Vector3i(1, 0, 0), Eigen::Vector3i(0, 1, 0), Eigen::Vector3i(0, 0, 1),
     Eigen::Vector3i(1, 1, 0), Eigen::Vector3i(1, 0, 1), Eigen::Vector3i(0, 1, 1), Eigen::Vector3i(1, 1, 1)}};

double monomial(const Eigen::Vector3i& powers, const Eigen::Vector3d& xi, const Eigen::Vector3i& derivative) {
    double result = 1;
    for (int d = 0; d < 3; ++d) {
        if (powers[d] < derivative[d])
            return 0;
        result *= (derivative[d] ? powers[d] : 1) * std::pow(xi[d], powers[d] - derivative[d]);
    }
    return result;
}
} // namespace

TEST(HexHermiteBasis, InterpolatesCornerValuesAndFirstDerivativesInVertexOrder) {
    const auto nodes = corners();
    for (int vertex = 0; vertex < 8; ++vertex) {
        const auto sample = HexHermiteBasis::evaluate(nodes.row(vertex).transpose());
        for (int slot = 0; slot < 8; ++slot)
            EXPECT_EQ(HexHermiteBasis::reference_node(8 * vertex + slot), nodes.row(vertex).transpose());
        EXPECT_EQ(sample.values, (Eigen::Matrix<double, 64, 1>::Unit(8 * vertex)));
        for (int d = 0; d < 3; ++d)
            EXPECT_EQ(sample.gradients.col(d), (Eigen::Matrix<double, 64, 1>::Unit(8 * vertex + d + 1)));
    }
    EXPECT_THROW(HexHermiteBasis::reference_node(-1), std::out_of_range);
    EXPECT_THROW(HexHermiteBasis::reference_node(64), std::out_of_range);
}

TEST(HexHermiteBasis, ReproducesAllTricubicMonomialsAndGradients) {
    const auto nodes = corners();
    for (int i = 0; i <= 3; ++i)
        for (int j = 0; j <= 3; ++j)
            for (int k = 0; k <= 3; ++k) {
                const Eigen::Vector3i powers(i, j, k);
                SCOPED_TRACE(powers.transpose());
                Eigen::Matrix<double, 64, 1> coefficients;
                for (int vertex = 0; vertex < 8; ++vertex)
                    for (int slot = 0; slot < 8; ++slot)
                        coefficients[8 * vertex + slot] = monomial(powers, nodes.row(vertex).transpose(),
                                                                   derivatives[static_cast<std::size_t>(slot)]);
                for (const auto& xi : samples) {
                    const auto sample = HexHermiteBasis::evaluate(xi);
                    EXPECT_NEAR(coefficients.dot(sample.values), monomial(powers, xi, Eigen::Vector3i::Zero()), 1e-13);
                    for (int d = 0; d < 3; ++d)
                        EXPECT_NEAR(coefficients.dot(sample.gradients.col(d)),
                                    monomial(powers, xi, Eigen::Vector3i::Unit(d)), 1e-13);
                }
            }
}

TEST(HexHermiteBasis, ReproducesAffineVectorFieldsWith192Dofs) {
    const auto nodes = corners();
    Eigen::Matrix3d A;
    A << 0.2, -0.1, 0.3, 0.4, 0.2, -0.3, -0.2, 0.5, 0.1;
    const Eigen::Vector3d b(-0.3, 0.4, 0.7);
    Eigen::Matrix<double, 64, 3> coefficients = Eigen::Matrix<double, 64, 3>::Zero();
    for (int vertex = 0; vertex < 8; ++vertex) {
        coefficients.row(8 * vertex) = nodes.row(vertex) * A.transpose() + b.transpose();
        for (int d = 0; d < 3; ++d)
            coefficients.row(8 * vertex + d + 1) = A.col(d).transpose();
    }
    const litepgo::Interpolation<HexHermiteBasis> interpolation(coefficients);
    EXPECT_EQ(interpolation.num_dofs(), 192);
    for (const auto& xi : samples) {
        EXPECT_TRUE(interpolation.value(xi).isApprox(A * xi + b, 1e-13));
        EXPECT_TRUE(interpolation.gradient(xi).isApprox(A, 1e-13));
    }
}

TEST(HexHermiteBasis, GradientsMatchFiniteDifferences) {
    constexpr double h = 1e-6;
    for (const auto& xi : samples) {
        const auto sample = HexHermiteBasis::evaluate(xi);
        for (int d = 0; d < 3; ++d) {
            Eigen::Vector3d plus = xi, minus = xi;
            plus[d] += h;
            minus[d] -= h;
            const Eigen::Matrix<double, 64, 1> fd =
                (HexHermiteBasis::evaluate(plus).values - HexHermiteBasis::evaluate(minus).values) / (2 * h);
            EXPECT_TRUE((fd - sample.gradients.col(d)).isZero(2e-9));
        }
    }
}
