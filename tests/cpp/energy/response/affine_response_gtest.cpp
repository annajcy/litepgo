#include "litepgo/energy/response/affine_response.hpp"
#include "litepgo/energy/response/composed_response.hpp"
#include "litepgo/energy/response/constant_response.hpp"
#include "litepgo/energy/response/exp_response.hpp"
#include "litepgo/energy/response/identity_response.hpp"
#include "litepgo/linalg/sparse_pattern.hpp"
#include <memory>

#include <gtest/gtest.h>
#include <limits>

namespace litepgo::energy::test {
namespace {
Eigen::Matrix<double, 2, 3> matrix() {
    return (Eigen::Matrix<double, 2, 3>() << 1, 0, -2, .5, 3, 0).finished();
}
} // namespace

TEST(AffineResponse, RectangularValuesDerivativesAndZeroHessian) {
    const auto A = matrix();
    const Vec b = Eigen::Vector2d(4, -1), x = Eigen::Vector3d(1, 2, -3);
    const Vec dx = Eigen::Vector3d(.2, -.5, .7), w = Eigen::Vector2d(2, -3);
    AffineResponse r(A.sparseView(), b);
    EXPECT_EQ(r.num_inputs(), 3);
    EXPECT_EQ(r.num_outputs(), 2);
    EXPECT_TRUE(r.has_hessian());
    SpMat J;
    EXPECT_TRUE(([&] {
                    auto evaluation_0 = r.evaluate(x);
                    auto jacobian_0 = evaluation_0->jacobian();
                    auto result = std::pair{litepgo::Vec(evaluation_0->value()), std::move(jacobian_0)};
                    J = std::move(result.second);
                    return std::move(result.first);
                }())
                    .isApprox(Eigen::Vector2d(11, 5.5)));
    EXPECT_TRUE(Eigen::MatrixXd(J).isApprox(A));
    EXPECT_TRUE(r.evaluate(x)->jvp(dx).isApprox(A * dx));
    EXPECT_TRUE(r.evaluate(x)->vjp(w).isApprox(A.transpose() * w));
    EXPECT_LT(
        (r.evaluate(x)->jvp(dx) - (r.evaluate(x + 1e-6 * dx)->value() - r.evaluate(x - 1e-6 * dx)->value()) / 2e-6)
            .norm(),
        1e-8);
    for (const Vec& weights : {w, Vec::Zero(2).eval()}) {
        const SpMat H = r.evaluate(x)->hessian(weights);
        EXPECT_EQ(H.rows(), 3);
        EXPECT_EQ(H.cols(), 3);
        EXPECT_EQ(H.nonZeros(), 0);
    }
    EXPECT_TRUE(r.evaluate(-x)->value().isApprox(-A * x + b));
}

TEST(AffineResponse, OwnsSnapshotsAndPreservesStoredZeros) {
    SpMat A(2, 3);
    A.insert(0, 0) = 2;
    A.insert(1, 1) = 0;
    A.insert(0, 2) = -1;
    Vec offset = Eigen::Vector2d(4, 5);
    AffineResponse r(A, offset);
    A.setZero();
    offset.setConstant(std::numeric_limits<double>::quiet_NaN());
    const Vec x = Eigen::Vector3d(1, 2, 3);
    EXPECT_TRUE(r.evaluate(x)->value().isApprox(Eigen::Vector2d(3, 5)));
    SpMat J = r.evaluate(x)->jacobian();
    EXPECT_EQ(J.nonZeros(), 3);
    EXPECT_TRUE(linalg::same_pattern(J, r.evaluate(x)->jacobian()));
    J.coeffs().setConstant(99);
    EXPECT_DOUBLE_EQ(r.evaluate(x)->jacobian().coeff(0, 0), 2);
    EXPECT_DOUBLE_EQ(r.evaluate(x)->jacobian().coeff(1, 1), 0);
}

TEST(AffineResponse, DefaultOffsetIdentityRankDeficientAndEmptyMaps) {
    const Vec x = Eigen::Vector3d(1, -2, 3);
    AffineResponse identity(Eigen::Matrix3d::Identity().sparseView());
    EXPECT_TRUE(identity.evaluate(x)->value().isApprox(x));
    AffineResponse rank_deficient(Eigen::Matrix<double, 2, 3>::Ones().sparseView(), Vec());
    EXPECT_TRUE(rank_deficient.evaluate(x)->value().isApprox(Vec::Constant(2, x.sum())));
    AffineResponse constant(SpMat(2, 3), Eigen::Vector2d(2, -1));
    EXPECT_TRUE(constant.evaluate(x)->value().isApprox(Eigen::Vector2d(2, -1)));
    EXPECT_TRUE(constant.evaluate(x)->jvp(x).isZero());
    EXPECT_TRUE(constant.evaluate(x)->vjp(Vec::Ones(2)).isZero());
    AffineResponse no_inputs(SpMat(2, 0), Eigen::Vector2d(2, 3));
    EXPECT_TRUE(no_inputs.evaluate(Vec(0))->value().isApprox(Eigen::Vector2d(2, 3)));
    EXPECT_EQ(no_inputs.evaluate(Vec(0))->hessian(Vec::Ones(2)).rows(), 0);
    AffineResponse no_outputs(SpMat(0, 3));
    EXPECT_EQ(no_outputs.evaluate(x)->value().size(), 0);
    EXPECT_TRUE(no_outputs.evaluate(x)->vjp(Vec(0)).isZero());
    EXPECT_EQ(no_outputs.evaluate(x)->hessian(Vec(0)).rows(), 3);
}

TEST(AffineResponse, ComposesWithNonlinearResponsesInBothDirections) {
    const auto A = matrix();
    const Vec offset = Eigen::Vector2d(.1, -.2), x = Eigen::Vector3d(-.3, .1, .2);
    const Vec w = Eigen::Vector2d(.4, -.7);
    auto affine = std::make_shared<AffineResponse>(SpMat(A.sparseView()), offset);
    ComposedResponse r(std::make_shared<ExpResponse>(2), affine);
    const Vec y = (A * x + offset).array().exp();
    EXPECT_TRUE(r.evaluate(x)->value().isApprox(y));
    const Eigen::Matrix3d H = A.transpose() * w.cwiseProduct(y).asDiagonal() * A;
    EXPECT_TRUE(Eigen::MatrixXd(r.evaluate(x)->hessian(w)).isApprox(H, 1e-12));
    const Vec dx = Eigen::Vector3d(.2, -.3, .4);
    EXPECT_LT((H * dx - (r.evaluate(x + 1e-6 * dx)->vjp(w) - r.evaluate(x - 1e-6 * dx)->vjp(w)) / 2e-6).norm(), 1e-8);
    ComposedResponse outer(affine, std::make_shared<ExpResponse>(3));
    const Vec diagonal = (A.transpose() * w).array() * x.array().exp();
    EXPECT_TRUE(Eigen::MatrixXd(outer.evaluate(x)->hessian(w)).isApprox(Eigen::MatrixXd(diagonal.asDiagonal())));
}

TEST(AffineResponse, RejectsInvalidConfigurationInputsAndNumericalOverflow) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    for (double bad : {nan, inf, -inf}) {
        SpMat A = matrix().sparseView();
        A.valuePtr()[0] = bad;
        EXPECT_THROW((AffineResponse(A)), std::invalid_argument);
        EXPECT_THROW((AffineResponse(matrix().sparseView(), Vec::Constant(2, bad))), std::invalid_argument);
    }
    EXPECT_THROW((AffineResponse(matrix().sparseView(), Vec::Zero(3))), std::invalid_argument);
    AffineResponse r(matrix().sparseView());
    const Vec x = Vec::Zero(3), w = Vec::Ones(2);
    EXPECT_THROW(r.evaluate(Vec::Zero(2))->value(), std::invalid_argument);
    EXPECT_THROW(r.evaluate(Vec::Constant(3, nan))->value(), std::invalid_argument);
    EXPECT_THROW(r.evaluate(x)->jvp(Vec::Ones(2)), std::invalid_argument);
    EXPECT_THROW(r.evaluate(x)->vjp(Vec::Ones(3)), std::invalid_argument);
    EXPECT_THROW(r.evaluate(x)->jvp(Vec::Constant(3, inf)), std::invalid_argument);
    EXPECT_THROW(r.evaluate(x)->vjp(Vec::Constant(2, inf)), std::invalid_argument);
    EXPECT_THROW(r.evaluate(x)->hessian(Vec::Zero(3)), std::invalid_argument);
    EXPECT_THROW(r.evaluate(x)->hessian(Vec::Constant(2, nan)), std::invalid_argument);
    SpMat wrong(2, 3);
    const double huge = std::numeric_limits<double>::max();
    AffineResponse overflow(Eigen::Matrix<double, 1, 1>::Constant(huge).sparseView());
    EXPECT_THROW(overflow.evaluate(Vec::Constant(1, 2))->value(), std::domain_error);
    EXPECT_THROW(overflow.evaluate(Vec::Zero(1))->jvp(Vec::Constant(1, 2)), std::domain_error);
    EXPECT_THROW(overflow.evaluate(Vec::Zero(1))->vjp(Vec::Constant(1, 2)), std::domain_error);
    EXPECT_TRUE(overflow.evaluate(Vec::Zero(1))->value().isZero());
}

TEST(IdentityResponse, ProductsJacobianAndHessian) {
    IdentityResponse r(3);
    const Vec x = Eigen::Vector3d(1, 2, 3), dx = Eigen::Vector3d(-1, 4, 5), w = Eigen::Vector3d(7, 8, -9);
    EXPECT_TRUE(r.evaluate(x)->value().isApprox(x));
    EXPECT_TRUE(r.evaluate(x)->jvp(dx).isApprox(dx));
    EXPECT_TRUE(r.evaluate(x)->vjp(w).isApprox(w));
    EXPECT_TRUE(Eigen::MatrixXd(r.evaluate(x)->jacobian()).isIdentity());
    EXPECT_EQ(r.evaluate(x)->hessian(w).nonZeros(), 0);
}

TEST(ConstantResponse, ProductsJacobianAndHessian) {
    const Vec x = Eigen::Vector3d(1, 2, 3), value = Eigen::Vector2d(4, -5);
    ConstantResponse r(3, value);
    auto evaluation_1 = r.evaluate(x);
    auto jacobian_1 = evaluation_1->jacobian();
    const auto [result, J] = std::pair{litepgo::Vec(evaluation_1->value()), std::move(jacobian_1)};
    EXPECT_TRUE(result.isApprox(value));
    EXPECT_EQ(J.rows(), 2);
    EXPECT_EQ(J.cols(), 3);
    EXPECT_EQ(J.nonZeros(), 0);
    const Vec jvp = r.evaluate(x)->jvp(x), vjp = r.evaluate(x)->vjp(value);
    EXPECT_EQ(jvp.size(), 2);
    EXPECT_EQ(vjp.size(), 3);
    EXPECT_TRUE(jvp.isZero());
    EXPECT_TRUE(vjp.isZero());
    EXPECT_EQ(r.evaluate(x)->hessian(value).nonZeros(), 0);
}
} // namespace litepgo::energy::test
