#include "litepgo/fem/basis/1d/cubic_hermite_basis.hpp"
#include "litepgo/fem/basis/3d/hex_q1_basis.hpp"
#include "litepgo/fem/response/sampling_response.hpp"
#include <gtest/gtest.h>
#include <limits>
#include <type_traits>

namespace litepgo::fem::test {
TEST(SamplingResponse, CoefficientMapsReturnAffineResponses) {
    Eigen::MatrixXd points(2, 1);
    points << .5, 2;
    auto value = value_sampling_response<CubicHermiteBasis>(points, 1);
    auto gradient = gradient_sampling_response<CubicHermiteBasis>(points, 1);
    auto hessian = hessian_sampling_response<CubicHermiteBasis>(points, 1);
    static_assert(std::is_same_v<decltype(value), energy::AffineResponse>);
    Vec c(4);
    c << 0, 0, 1, 3; // t^3
    points.setZero();
    EXPECT_TRUE(value.eval_value(c).isApprox(Eigen::Vector2d(.125, 8)));
    EXPECT_TRUE(gradient.eval_value(c).isApprox(Eigen::Vector2d(.75, 12)));
    EXPECT_TRUE(hessian.eval_value(c).isApprox(Eigen::Vector2d(3, 12)));
    EXPECT_EQ(hessian.eval_hessian(c, Eigen::Vector2d(2, -3)).nonZeros(), 0);
}
TEST(SamplingResponse, PointDerivativeComponentOrdering) {
    Eigen::MatrixXd points(2, 3);
    points << .2, .3, .4, .6, .7, .8;
    Eigen::MatrixXd C(8, 2);
    for (int a = 0; a < 8; ++a) {
        const auto p = HexQ1Basis::coefficient_info(a).point;
        C(a, 0) = p.x() * p.y();
        C(a, 1) = p.y() * p.z();
    }
    const Vec c = C.reshaped<Eigen::RowMajor>();
    auto g = gradient_sampling_response<HexQ1Basis>(points, 2);
    Vec expected_g(12);
    expected_g << .3, 0, .2, .4, 0, .3, .7, 0, .6, .8, 0, .7;
    EXPECT_TRUE(g.eval_value(c).isApprox(expected_g));
    auto h = hessian_sampling_response<HexQ1Basis>(points, 2);
    Vec expected_h(24);
    expected_h << 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0;
    EXPECT_TRUE(h.eval_value(c).isApprox(expected_h));
}
TEST(SamplingResponse, InvalidAndEmptyInputs) {
    Eigen::MatrixXd points = Eigen::MatrixXd::Zero(0, 3);
    auto empty = value_sampling_response<HexQ1Basis>(points, 2);
    EXPECT_EQ(empty.num_inputs(), 16);
    EXPECT_EQ(empty.num_outputs(), 0);
    EXPECT_EQ(empty.eval_jacobian(Vec::Zero(16)).rows(), 0);
    EXPECT_THROW(value_sampling_response<HexQ1Basis>(points, 0), std::invalid_argument);
    EXPECT_THROW(gradient_sampling_response<HexQ1Basis>(points, std::numeric_limits<int>::max()),
                 std::invalid_argument);
    EXPECT_THROW(hessian_sampling_response<HexQ1Basis>(Eigen::MatrixXd::Zero(1, 2), 1), std::invalid_argument);
    points = Eigen::MatrixXd::Constant(1, 3, std::numeric_limits<double>::quiet_NaN());
    EXPECT_THROW(value_sampling_response<HexQ1Basis>(points, 1), std::invalid_argument);
}
} // namespace litepgo::fem::test
