#include "litepgo/fem/basis/1d/cubic_hermite_basis.hpp"
#include "litepgo/fem/basis/3d/hex_q1_basis.hpp"
#include "litepgo/fem/response/interpolated_response.hpp"
#include <gtest/gtest.h>
#include <limits>
namespace litepgo::fem::test {
TEST(InterpolatedResponse, CubicHermiteSpatialDerivatives) {
    // f(t)=t^3, coefficients f(0), f'(0), f(1), f'(1).
    Eigen::MatrixXd C(4, 1);
    C << 0, 0, 1, 3;
    InterpolatedResponse<CubicHermiteBasis> r(C);
    C.setZero();
    Vec x = Vec::Constant(1, .5), w = Vec::Constant(1, -2);
    auto evaluation_0 = r.evaluate(x);
    auto jacobian_0 = evaluation_0->jacobian();
    auto [value, J] = std::pair{litepgo::Vec(evaluation_0->value()), std::move(jacobian_0)};
    EXPECT_NEAR(value[0], .125, 1e-14);
    EXPECT_NEAR(J.coeff(0, 0), .75, 1e-14);
    EXPECT_NEAR(r.evaluate(x)->hessian(w).coeff(0, 0), -6, 1e-14);
    EXPECT_NEAR(r.evaluate(x)->jvp(w)[0], -1.5, 1e-14);
    EXPECT_NEAR(r.evaluate(x)->vjp(w)[0], -1.5, 1e-14);
}
TEST(InterpolatedResponse, MixedHessianIsFullAndUnscaled) {
    Eigen::MatrixXd C(8, 2);
    for (int a = 0; a < 8; ++a) {
        const auto p = HexQ1Basis::coefficient_info(a).point;
        C(a, 0) = p.x() * p.y();
        C(a, 1) = p.y() * p.z();
    }
    InterpolatedResponse<HexQ1Basis> r(C);
    Vec x(3), w(2);
    x << .2, .3, .4;
    w << 2, -3;
    Eigen::Matrix3d expected;
    expected << 0, 2, 0, 2, 0, -3, 0, -3, 0;
    const auto H = r.evaluate(x)->hessian(w);
    EXPECT_TRUE(Eigen::MatrixXd(H).isApprox(expected, 1e-14));
    EXPECT_EQ(r.evaluate(x)->hessian(Vec::Zero(2)).nonZeros(), H.nonZeros());
    EXPECT_THROW(r.evaluate(Vec::Zero(2))->value(), std::invalid_argument);
    C(0, 0) = std::numeric_limits<double>::infinity();
    EXPECT_THROW((InterpolatedResponse<HexQ1Basis>(C)), std::invalid_argument);
    EXPECT_THROW((InterpolatedResponse<HexQ1Basis>(Eigen::MatrixXd(8, 0))), std::invalid_argument);
}
} // namespace litepgo::fem::test
