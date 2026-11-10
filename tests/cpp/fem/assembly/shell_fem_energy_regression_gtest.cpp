#include "litepgo/fem/material/shell/pointwise_shell_adapter.hpp"
#include <memory>

#include "litepgo/dof/stencil_dof_map.hpp"
#include "litepgo/fem/assembly/shell_fem_energy.hpp"
#include "litepgo/fem/material/shell/koiter_stvk.hpp"
#include "litepgo/fem/quadrature/2d/triangle_centroid_quadrature.hpp"
#include "litepgo/mesh/triangle_mid_edge_stencils.hpp"

#include <stdexcept>
#include <utility>
#include <vector>

#include "../../testing/finite_difference.hpp"
#include "litepgo/dof/dof_reduction.hpp"
#include "litepgo/linalg/sparse_solver/eigen_ldlt.hpp"
#include "litepgo/solver/newton.hpp"
#include <Eigen/Geometry>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>

namespace litepgo::fem::test {

inline mesh::TriangleTopology shell_stvk_topology() {
    // Face 0 has three neighbors; boundary faces each have one neighbor.
    return mesh::TriangleTopology(6, {{0, 1, 2}, {2, 1, 3}, {0, 2, 4}, {1, 0, 5}});
}

inline Vec shell_stvk_rest_positions() {
    // A flat patch with face areas 1, 2, 1/2, 2 and consistent winding.
    Eigen::Matrix<double, 6, 3> X_nodes;
    X_nodes << 0, 0, 0, 2, 0, 0, 0, 1, 0, 2, 2, 0, -1, 0, 0, 0, -2, 0;
    return X_nodes.reshaped<Eigen::RowMajor>();
}

inline auto make_shell_energy(const Vec& X_nodes) {
    const auto topology = shell_stvk_topology();
    const auto stencils = mesh::make_triangle_mid_edge_stencils(topology);
    const dof::StencilDofMap map(topology.num_vertices(), stencils.vertices, 3);
    std::vector<fem::TriangleMidEdgeRestData> rest;
    for (int f = 0; f < map.num_elements(); ++f)
        rest.emplace_back(map.gather(f, X_nodes), stencils.opposite[static_cast<std::size_t>(f)]);

    const double thickness = 0.1, Y = 100, nu = 0.3;
    // Plane-stress Lame parameters for a shell.
    const Eigen::Vector2d theta(Y * nu / (1 - nu * nu), Y / (2 * (1 + nu)));
    return fem::ShellFemEnergy(std::make_shared<litepgo::fem::PointwiseShellAdapter<fem::KoiterStVK>>(), map,
                               std::move(rest), std::vector<double>(map.num_elements(), thickness),
                               std::vector<Eigen::VectorXd>(map.num_elements(), theta));
}

namespace {

Vec bent_displacements() {
    Vec u = Vec::Zero(18);
    u.segment<3>(9) << 0.02, -0.01, 0.2;
    u.segment<3>(12) << -0.01, 0.02, -0.15;
    u.segment<3>(15) << 0.01, -0.02, 0.18;
    return u;
}

void expect_safe_segment(const energy::Energy& energy, const Vec& u, const Vec& du, double alpha) {
    ASSERT_GE(alpha, 0);
    ASSERT_LE(alpha, 1);
    for (int i = 0; i <= 32; ++i)
        EXPECT_NO_THROW(energy.evaluate(u + (alpha * i / 32) * du)->value());
}

TEST(ShellFemEnergyRegression, UniformStretchAndRigidMotionMatchReferenceAreaAndMaterialParameters) {
    const Vec X_nodes = shell_stvk_rest_positions();
    auto energy = make_shell_energy(X_nodes);
    EXPECT_EQ(energy.num_dofs(), 18);
    EXPECT_DOUBLE_EQ(energy.evaluate(Vec::Zero(18))->value(), 0);
    const double Y = 100, nu = 0.3;
    const double lambda_ps = Y * nu / (1 - nu * nu), mu = Y / (2 * (1 + nu));
    const double density = 0.1 / 2 * (lambda_ps + mu) * 0.21 * 0.21;
    EXPECT_NEAR(energy.evaluate(0.1 * X_nodes)->value(), 5.5 * density, 1e-12);
    const Eigen::Matrix3d R = Eigen::AngleAxisd(0.4, Eigen::Vector3d(1, 2, -1).normalized()).toRotationMatrix();
    Vec rigid(18);
    for (int v = 0; v < 6; ++v) {
        const Eigen::Vector3d X = X_nodes.segment<3>(3 * v);
        rigid.segment<3>(3 * v) = R * X + Eigen::Vector3d(0.3, -0.4, 0.7) - X;
    }
    EXPECT_NEAR(energy.evaluate(rigid)->value(), 0, 1e-24);
    EXPECT_TRUE(energy.evaluate(rigid)->gradient().isZero(1e-11));
}

TEST(ShellFemEnergyRegression, BentStateHasCorrectGradientHessianAndNeighborCouplings) {
    auto energy = make_shell_energy(shell_stvk_rest_positions());
    const Vec u = bent_displacements();
    EXPECT_GT(energy.evaluate(u)->value(), 0);
    energy::test::expect_derivatives_match(energy, u, 2e-6);
    const SpMat H = energy.evaluate(u)->hessian();
    EXPECT_GT(std::abs(H.coeff(3 * 3 + 2, 3 * 4 + 2)), 1e-8);
    EXPECT_NO_THROW(energy.evaluate(u)->hessian());
}

TEST(ShellFemEnergyRegression, FeasibleStepPreventsInteriorCollapseEvenWhenEndpointIsFeasible) {
    const Vec X_nodes = shell_stvk_rest_positions();
    auto energy = make_shell_energy(X_nodes);
    const Vec u = Vec::Zero(18), du = -2 * X_nodes;
    EXPECT_NO_THROW(energy.evaluate(u + du)->value());
    EXPECT_THROW(energy.evaluate(u + 0.5 * du)->value(), std::domain_error);
    const double alpha = energy.evaluate(u)->max_feasible_step(du);
    EXPECT_GT(alpha, 0);
    EXPECT_LT(alpha, 0.5);
    expect_safe_segment(energy, u, du, alpha);
    Vec translation(18);
    for (int v = 0; v < 6; ++v)
        translation.segment<3>(3 * v) = Eigen::Vector3d(0.1, -0.2, 0.3);
    EXPECT_DOUBLE_EQ(energy.evaluate(u)->max_feasible_step(translation), 1);
    EXPECT_DOUBLE_EQ(energy.evaluate(u)->max_feasible_step(Vec::Zero(18)), 1);
    EXPECT_DOUBLE_EQ(energy.evaluate(u)->max_feasible_step(-1e200 * X_nodes), 0);
}

TEST(ShellFemEnergyRegression, FeasibleStepKeepsAdjacentNormalsFromCancelling) {
    auto energy = make_shell_energy(shell_stvk_rest_positions());
    Vec u = Vec::Zero(18), du = Vec::Zero(18);
    u.segment<3>(9) << -1, -1.5, 1;
    du.segment<3>(9) << -1, -0.5, -1;
    EXPECT_NO_THROW(energy.evaluate(u)->value());
    EXPECT_THROW(energy.evaluate(u + du)->value(), std::domain_error);
    const double alpha = energy.evaluate(u)->max_feasible_step(du);
    EXPECT_GT(alpha, 0);
    EXPECT_LT(alpha, 1);
    expect_safe_segment(energy, u, du, alpha);
    EXPECT_THROW(energy.evaluate(u + du)->max_feasible_step(du), std::domain_error);
}

TEST(ShellFemEnergyRegression, NewtonRelaxesBentPatchWhileKeepingCentralTriangleFixed) {
    auto energy = make_shell_energy(shell_stvk_rest_positions());
    const Vec u0 = bent_displacements();
    const double initial_energy = energy.evaluate(u0)->value();
    const dof::DofReduction fixed(18, {0, 1, 2, 3, 4, 5, 6, 7, 8});
    solver::NewtonOptions options;
    options.gradient_tolerance = 1e-9;
    linalg::EigenLDLT solver;
    const auto result = solver::newton_solve(energy, fixed, u0, options, solver);
    ASSERT_EQ(result.status, solver::NewtonStatus::Converged);
    EXPECT_GT(result.iterations, 0);
    EXPECT_LT(result.energy, initial_energy);
    EXPECT_LT(result.energy, 1e-14);
    EXPECT_LE(result.gradient_norm, options.gradient_tolerance);
    EXPECT_EQ(result.x.head(9), u0.head(9));
    EXPECT_LT(result.x.tail(9).norm(), 1e-5);
}

TEST(ShellFemEnergyRegression, OwnsReferenceDataAndRejectsInvalidRestAndTrialGeometry) {
    Vec X_nodes = shell_stvk_rest_positions();
    auto energy = make_shell_energy(X_nodes);
    const Vec u = bent_displacements();
    const double W = energy.evaluate(u)->value();
    X_nodes.setZero();
    EXPECT_DOUBLE_EQ(energy.evaluate(u)->value(), W);
    EXPECT_THROW(make_shell_energy(Vec::Zero(17)), std::invalid_argument);
    EXPECT_THROW(make_shell_energy(Vec::Zero(18)), std::domain_error);
    X_nodes = shell_stvk_rest_positions();
    X_nodes[0] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(make_shell_energy(X_nodes), std::invalid_argument);
    Vec collapsed = Vec::Zero(18);
    collapsed.segment<3>(3) = -shell_stvk_rest_positions().segment<3>(3);
    // Repeat cancellation across concurrent batches and verify the object remains usable.
    for (int attempt = 0; attempt < 25; ++attempt) {
        EXPECT_THROW(energy.evaluate(collapsed)->value(), std::domain_error);
        EXPECT_THROW(energy.evaluate(collapsed)->gradient(), std::domain_error);
        EXPECT_THROW(energy.evaluate(collapsed)->hessian(), std::domain_error);
        EXPECT_THROW(energy.evaluate(collapsed)->max_feasible_step(Vec::Zero(18)), std::domain_error);
        EXPECT_DOUBLE_EQ(energy.evaluate(u)->value(), W);
    }

    const auto stencils = mesh::make_triangle_mid_edge_stencils(shell_stvk_topology());
    const fem::TriangleMidEdgeRestData rest(shell_stvk_rest_positions().reshaped<Eigen::RowMajor>(6, 3),
                                            stencils.opposite[0]);
    EXPECT_THROW(
        fem::eval_triangle_mid_edge_feasible_step(rest.X_local.topRows(5), Eigen::MatrixXd::Zero(6, 3), rest.opposite),
        std::invalid_argument);
    Eigen::MatrixXd bad_direction = Eigen::MatrixXd::Zero(6, 3);
    bad_direction(0, 0) = std::numeric_limits<double>::infinity();
    EXPECT_THROW(fem::eval_triangle_mid_edge_feasible_step(rest.X_local, bad_direction, rest.opposite),
                 std::invalid_argument);
}
} // namespace

} // namespace litepgo::fem::test
