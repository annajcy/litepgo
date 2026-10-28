#include "litepgo/fem/basis/2d/quad_hermite_basis.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>

namespace {
using namespace litepgo;
static_assert(Basis2D<QuadHermiteBasis>);
static_assert(SecondOrderBasis<QuadHermiteBasis, 2>);
static_assert(QuadHermiteBasis::num_basis_functions == 16);

double monomial(int i, int j, const Eigen::Vector2d& xi, int dx = 0, int dy = 0) {
    if (i < dx || j < dy)
        return 0;
    double scale = 1;
    for (int d = 0; d < dx; ++d)
        scale *= i - d;
    for (int d = 0; d < dy; ++d)
        scale *= j - d;
    return scale * std::pow(xi.x(), i - dx) * std::pow(xi.y(), j - dy);
}

const std::array<Eigen::Vector2d, 5> samples{Eigen::Vector2d(0, 0), Eigen::Vector2d(1, 1), Eigen::Vector2d(0, 0.4),
                                             Eigen::Vector2d(0.23, 0.61), Eigen::Vector2d(1.2, -0.3)};

TEST(QuadHermiteBasis, InterpolatesCornerValuesAndDerivativeSlots) {
    Eigen::Matrix<double, 4, 2> corners;
    corners << 0, 0, 1, 0, 1, 1, 0, 1;
    for (int v = 0; v < 4; ++v) {
        for (int slot = 0; slot < 4; ++slot)
            EXPECT_EQ(QuadHermiteBasis::reference_node(4 * v + slot), corners.row(v).transpose());
        const auto sample = QuadHermiteBasis::evaluate(corners.row(v).transpose());
        EXPECT_EQ(sample.N, (Eigen::Matrix<double, 16, 1>::Unit(4 * v)));
        EXPECT_EQ(sample.dN_dxi.col(0), (Eigen::Matrix<double, 16, 1>::Unit(4 * v + 1)));
        EXPECT_EQ(sample.dN_dxi.col(1), (Eigen::Matrix<double, 16, 1>::Unit(4 * v + 2)));
        EXPECT_EQ(sample.d2N_dxi2.col(1), (Eigen::Matrix<double, 16, 1>::Unit(4 * v + 3)));
    }
    EXPECT_THROW(QuadHermiteBasis::reference_node(-1), std::out_of_range);
    EXPECT_THROW(QuadHermiteBasis::reference_node(16), std::out_of_range);
}

TEST(QuadHermiteBasis, ReproducesEveryBicubicMonomialAndItsFirstAndSecondDerivatives) {
    const std::array<Eigen::Vector2i, 4> slots{Eigen::Vector2i(0, 0), Eigen::Vector2i(1, 0), Eigen::Vector2i(0, 1),
                                               Eigen::Vector2i(1, 1)};
    const std::array<Eigen::Vector2i, 3> second{Eigen::Vector2i(2, 0), Eigen::Vector2i(1, 1), Eigen::Vector2i(0, 2)};
    for (int i = 0; i <= 3; ++i)
        for (int j = 0; j <= 3; ++j) {
            SCOPED_TRACE(::testing::Message() << i << "," << j);
            Eigen::Matrix<double, 16, 1> coefficients;
            for (int v = 0; v < 4; ++v)
                for (int s = 0; s < 4; ++s) {
                    const auto& derivative = slots[static_cast<std::size_t>(s)];
                    coefficients[4 * v + s] =
                        monomial(i, j, QuadHermiteBasis::reference_node(4 * v), derivative.x(), derivative.y());
                }
            for (const auto& xi : samples) {
                const auto sample = QuadHermiteBasis::evaluate(xi);
                EXPECT_NEAR(coefficients.dot(sample.N), monomial(i, j, xi), 1e-13);
                EXPECT_NEAR(coefficients.dot(sample.dN_dxi.col(0)), monomial(i, j, xi, 1, 0), 1e-13);
                EXPECT_NEAR(coefficients.dot(sample.dN_dxi.col(1)), monomial(i, j, xi, 0, 1), 1e-13);
                for (int s = 0; s < 3; ++s) {
                    const auto& derivative = second[static_cast<std::size_t>(s)];
                    EXPECT_NEAR(coefficients.dot(sample.d2N_dxi2.col(s)),
                                monomial(i, j, xi, derivative.x(), derivative.y()), 2e-13);
                }
            }
        }
}

TEST(QuadHermiteBasis, FirstAndSecondDerivativesMatchFiniteDifferences) {
    constexpr double h = 1e-6;
    for (const auto& xi : samples) {
        const auto sample = QuadHermiteBasis::evaluate(xi);
        for (int d = 0; d < 2; ++d) {
            Eigen::Vector2d plus = xi, minus = xi;
            plus[d] += h;
            minus[d] -= h;
            const auto a = QuadHermiteBasis::evaluate(plus), b = QuadHermiteBasis::evaluate(minus);
            EXPECT_TRUE(((a.N - b.N) / (2 * h) - sample.dN_dxi.col(d)).isZero(2e-9));
            for (int e = 0; e < 2; ++e)
                EXPECT_TRUE(((a.dN_dxi.col(e) - b.dN_dxi.col(e)) / (2 * h) - sample.d2N_dxi2.col(d + e)).isZero(4e-9));
        }
    }
}
} // namespace
