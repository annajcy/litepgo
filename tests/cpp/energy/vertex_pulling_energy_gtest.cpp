#include "litepgo/energy/vertex_pulling_energy.hpp"

#include "../testing/finite_difference.hpp"

#include <gtest/gtest.h>

#include <array>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

using litepgo::HessianKind;
using litepgo::SpMat;
using litepgo::Vec;
using litepgo::VertexPullingEnergy;

// Two pulled vertices among eight DOFs, with position DOFs out of order.
std::vector<std::array<int, 3>> position_dofs() {
    return {{0, 1, 2}, {7, 5, 6}};
}

Vec targets() {
    Vec t(6);
    t << 1.0, -2.0, 0.5, 3.0, 0.0, -1.0;
    return t;
}

} // namespace

TEST(VertexPullingEnergy, EvaluatesValueGradientAndHessian) {
    const double k = 2.5;
    const VertexPullingEnergy energy(8, position_dofs(), targets(), k);
    const Vec x = Vec::LinSpaced(8, -1.0, 2.5);
    const std::vector<int> dofs = {0, 1, 2, 7, 5, 6};
    double expected = 0.0;
    Vec expected_gradient = Vec::Zero(8);
    Eigen::MatrixXd expected_hessian = Eigen::MatrixXd::Zero(8, 8);
    for (std::size_t i = 0; i < dofs.size(); ++i) {
        const double gap = x[dofs[i]] - targets()[static_cast<Eigen::Index>(i)];
        expected += 0.5 * k * gap * gap;
        expected_gradient[dofs[i]] = k * gap;
        expected_hessian(dofs[i], dofs[i]) = k;
    }
    EXPECT_DOUBLE_EQ(energy.eval_value(x), expected);
    EXPECT_TRUE(energy.eval_gradient(x).isApprox(expected_gradient, 1e-15));
    EXPECT_EQ(Eigen::MatrixXd(energy.eval_hessian(x)), expected_hessian);
    EXPECT_EQ(Eigen::MatrixXd(energy.eval_hessian(x, HessianKind::Projected)), expected_hessian);
    EXPECT_EQ(energy.hessian_pattern().nonZeros(), 6);
    EXPECT_TRUE(energy.has_fixed_pattern());
    litepgo::testing::expect_derivatives_match(energy, x);
}

TEST(VertexPullingEnergy, VanishesAtTargets) {
    const VertexPullingEnergy energy(8, position_dofs(), targets(), 1.0);
    Vec x = Vec::Constant(8, 4.0);
    x.segment<3>(0) = targets().head<3>();
    x[7] = targets()[3];
    x[5] = targets()[4];
    x[6] = targets()[5];
    EXPECT_EQ(energy.eval_value(x), 0.0);
    EXPECT_TRUE(energy.eval_gradient(x).isZero(0.0));
}

TEST(VertexPullingEnergy, AllowsNoPulledVertices) {
    const VertexPullingEnergy energy(3, {}, Vec(), 1.0);
    EXPECT_EQ(energy.eval_value(Vec::Ones(3)), 0.0);
    EXPECT_EQ(energy.hessian_pattern().nonZeros(), 0);
}

TEST(VertexPullingEnergy, RejectsInvalidInput) {
    EXPECT_THROW(VertexPullingEnergy(-1, {}, Vec(), 1.0), std::invalid_argument);
    EXPECT_THROW(VertexPullingEnergy(8, position_dofs(), targets(), 0.0), std::invalid_argument);
    EXPECT_THROW(VertexPullingEnergy(8, position_dofs(), targets(), -1.0), std::invalid_argument);
    EXPECT_THROW(VertexPullingEnergy(8, position_dofs(), targets(), std::numeric_limits<double>::infinity()),
                 std::invalid_argument);
    EXPECT_THROW(VertexPullingEnergy(8, position_dofs(), Vec::Zero(5), 1.0), std::invalid_argument);
    Vec nan_targets = targets();
    nan_targets[2] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(VertexPullingEnergy(8, position_dofs(), nan_targets, 1.0), std::invalid_argument);
    EXPECT_THROW(VertexPullingEnergy(7, position_dofs(), targets(), 1.0), std::invalid_argument);
    EXPECT_THROW(VertexPullingEnergy(8, {{0, 1, 2}, {-1, 5, 6}}, targets(), 1.0), std::invalid_argument);
    EXPECT_THROW(VertexPullingEnergy(8, {{0, 1, 2}, {2, 5, 6}}, targets(), 1.0), std::invalid_argument);
}
