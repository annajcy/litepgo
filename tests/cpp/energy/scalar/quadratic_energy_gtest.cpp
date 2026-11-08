#include "litepgo/energy/scalar/quadratic_energy.hpp"

#include "../../testing/finite_difference.hpp"

#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>
#include <vector>

namespace litepgo::energy::test {

namespace {

SpMat symmetric_matrix() {
    const std::vector<Triplet> entries = {{0, 0, 4.0}, {1, 1, 3.0}, {2, 2, 5.0}, {0, 2, -1.0}, {2, 0, -1.0}};
    SpMat A(3, 3);
    A.setFromTriplets(entries.begin(), entries.end());
    return A;
}

} // namespace

TEST(QuadraticEnergy, EvaluatesValueGradientAndHessian) {
    const SpMat A = symmetric_matrix();
    const Vec b = Vec::LinSpaced(3, 1.0, 3.0);
    const QuadraticEnergy energy(A, b);
    const Vec x = Vec::LinSpaced(3, -1.0, 2.0);
    const Eigen::MatrixXd dense = Eigen::MatrixXd(A);
    EXPECT_DOUBLE_EQ(energy.eval_value(x), 0.5 * x.dot(dense * x) + b.dot(x));
    EXPECT_EQ(energy.eval_gradient(x), Vec(dense * x + b));
    EXPECT_EQ(Eigen::MatrixXd(energy.eval_hessian(x)), dense);
    EXPECT_EQ(Eigen::MatrixXd(energy.eval_hessian(x, HessianKind::Projected)), dense);
    expect_derivatives_match(energy, x);
}

TEST(QuadraticEnergy, DefaultsToNoLinearTerm) {
    const QuadraticEnergy energy(symmetric_matrix());
    const Vec x = Vec::LinSpaced(3, -1.0, 2.0);
    EXPECT_EQ(energy.eval_gradient(x), Vec(symmetric_matrix() * x));
}

TEST(QuadraticEnergy, RejectsInvalidInput) {
    for (double bad : {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
        SpMat A = symmetric_matrix();
        A.coeffRef(0, 0) = bad;
        EXPECT_THROW(QuadraticEnergy{A}, std::invalid_argument);
        Vec b = Vec::Zero(3);
        b[0] = bad;
        EXPECT_THROW(QuadraticEnergy(symmetric_matrix(), b), std::invalid_argument);
    }

    SpMat asymmetric_values = symmetric_matrix();
    asymmetric_values.coeffRef(0, 2) = -2.0;
    EXPECT_THROW(QuadraticEnergy{asymmetric_values}, std::invalid_argument);

    SpMat asymmetric_pattern = symmetric_matrix();
    asymmetric_pattern.insert(1, 0) = 0.0;
    EXPECT_THROW(QuadraticEnergy{asymmetric_pattern}, std::invalid_argument);

    EXPECT_THROW(QuadraticEnergy(SpMat(3, 2)), std::invalid_argument);
    EXPECT_THROW(QuadraticEnergy(symmetric_matrix(), Vec::Ones(2)), std::invalid_argument);
}

} // namespace litepgo::energy::test
