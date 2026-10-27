#include "litepgo/fem/basis/basis.hpp"
#include "litepgo/fem/basis/1d/cubic_hermite_basis.hpp"

#include <gtest/gtest.h>

#include <cmath>

using litepgo::CubicHermiteBasis;

static_assert(litepgo::Basis1D<CubicHermiteBasis>);
static_assert(CubicHermiteBasis::num_basis_functions == 4);

TEST(CubicHermiteBasis, InterpolatesEndpointValuesAndDerivatives) {
    for (int endpoint : {0, 1}) {
        const auto basis_sample = CubicHermiteBasis::evaluate(endpoint);
        EXPECT_EQ(basis_sample.N, Eigen::Vector4d::Unit(2 * endpoint));
        EXPECT_EQ(basis_sample.dN_dxi, Eigen::Vector4d::Unit(2 * endpoint + 1));
    }
}

TEST(CubicHermiteBasis, ReproducesAllCubicMonomialsAndDerivatives) {
    for (int degree = 0; degree <= 3; ++degree) {
        SCOPED_TRACE(degree);
        const Eigen::Vector4d coefficients(degree == 0 ? 1 : 0, degree == 1 ? 1 : 0, 1, degree);
        for (double t : {-0.5, 0.0, 0.13, 0.5, 0.61, 1.0, 1.5}) {
            SCOPED_TRACE(t);
            const auto basis_sample = CubicHermiteBasis::evaluate(t);
            EXPECT_NEAR(coefficients.dot(basis_sample.N), std::pow(t, degree), 1e-14);
            EXPECT_NEAR(coefficients.dot(basis_sample.dN_dxi), degree == 0 ? 0 : degree * std::pow(t, degree - 1),
                        1e-14);
        }
    }
}

TEST(CubicHermiteBasis, GradientsMatchFiniteDifferences) {
    constexpr double h = 1e-6;
    for (double t : {-0.5, 0.0, 0.13, 0.5, 0.61, 1.0, 1.5}) {
        const Eigen::Vector4d fd =
            (CubicHermiteBasis::evaluate(t + h).N - CubicHermiteBasis::evaluate(t - h).N) / (2 * h);
        EXPECT_TRUE((fd - CubicHermiteBasis::evaluate(t).dN_dxi).isZero(2e-9));
    }
}
