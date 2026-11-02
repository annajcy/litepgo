#include "litepgo/fem/basis/1d/cubic_hermite_basis.hpp"
#include "litepgo/fem/basis/basis.hpp"

#include <gtest/gtest.h>

#include <cmath>

namespace litepgo::fem::test {

static_assert(litepgo::fem::Basis1D<CubicHermiteBasis>);
static_assert(CubicHermiteBasis::num_basis_functions == 4);

TEST(CubicHermiteBasis, InterpolatesEndpointValuesAndDerivatives) {
    for (int endpoint : {0, 1}) {
        const auto N = CubicHermiteBasis::eval_N(endpoint);
        const auto dN_dxi = CubicHermiteBasis::eval_dN_dxi(endpoint);
        EXPECT_EQ(N, Eigen::Vector4d::Unit(2 * endpoint));
        EXPECT_EQ(dN_dxi, Eigen::Vector4d::Unit(2 * endpoint + 1));
    }
}

TEST(CubicHermiteBasis, ReproducesAllCubicMonomialsAndDerivatives) {
    for (int degree = 0; degree <= 3; ++degree) {
        SCOPED_TRACE(degree);
        const Eigen::Vector4d coefficients(degree == 0 ? 1 : 0, degree == 1 ? 1 : 0, 1, degree);
        for (double t : {-0.5, 0.0, 0.13, 0.5, 0.61, 1.0, 1.5}) {
            SCOPED_TRACE(t);
            const auto N = CubicHermiteBasis::eval_N(t);
            const auto dN_dxi = CubicHermiteBasis::eval_dN_dxi(t);
            const auto d2N_dxi2 = CubicHermiteBasis::eval_d2N_dxi2(t);
            EXPECT_NEAR(coefficients.dot(N), std::pow(t, degree), 1e-14);
            EXPECT_NEAR(coefficients.dot(dN_dxi), degree == 0 ? 0 : degree * std::pow(t, degree - 1), 1e-14);
            EXPECT_NEAR(coefficients.dot(d2N_dxi2), degree < 2 ? 0 : degree * (degree - 1) * std::pow(t, degree - 2),
                        1e-14);
        }
    }
}

TEST(CubicHermiteBasis, GradientsMatchFiniteDifferences) {
    constexpr double h = 1e-6;
    for (double t : {-0.5, 0.0, 0.13, 0.5, 0.61, 1.0, 1.5}) {
        const Eigen::Vector4d fd = (CubicHermiteBasis::eval_N(t + h) - CubicHermiteBasis::eval_N(t - h)) / (2 * h);
        EXPECT_TRUE((fd - CubicHermiteBasis::eval_dN_dxi(t)).isZero(2e-9));
        const Eigen::Vector4d second_fd =
            (CubicHermiteBasis::eval_dN_dxi(t + h) - CubicHermiteBasis::eval_dN_dxi(t - h)) / (2 * h);
        EXPECT_TRUE((second_fd - CubicHermiteBasis::eval_d2N_dxi2(t)).isZero(3e-9));
    }
}

} // namespace litepgo::fem::test
