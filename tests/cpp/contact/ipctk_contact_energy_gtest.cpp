#include "../testing/evaluate_energy.hpp"
#include "litepgo/contact/ipctk_contact_energy.hpp"

#include "../testing/finite_difference.hpp"
#include "litepgo/dof/surface_kinematics.hpp"
#include "litepgo/energy/scalar/affine_mapped_energy.hpp"

#include <Eigen/Eigenvalues>
#include <gtest/gtest.h>

#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace litepgo::contact::test {

namespace {

// Triangle A spans [0, 10]^2 in the plane z = 0. Triangle B points vertex 3 down at
// (2, 2, height); its other vertices rise away from A, so with dhat = 1 and height
// below 1 the only active pair is vertex 3 against face A.
mesh::TriangleMesh two_triangles(double height) {
    Vec positions(18);
    positions << 0, 0, 0, 10, 0, 0, 0, 10, 0, //
        2, 2, height, 6, 2, height + 4, 2, 6, height + 4;
    return {positions, mesh::TriangleTopology(6, {{0, 1, 2}, {3, 4, 5}})};
}

IPCTKContactOptions options(double dhat, double min_distance = 0.0) {
    IPCTKContactOptions result;
    result.dhat = dhat;
    result.min_distance = min_distance;
    return result;
}

// Moves triangle B (vertices 3..5) by delta.
Vec translate_b(Vec x, const Eigen::Vector3d& delta) {
    for (int v = 3; v < 6; ++v) {
        x.segment<3>(3 * v) += delta;
    }
    return x;
}

} // namespace

TEST(IPCTKContactEnergy, RejectsInvalidConfiguration) {
    const mesh::TriangleMesh surface = two_triangles(0.5);
    EXPECT_THROW(IPCTKContactEnergy(surface, IPCTKContactOptions{}), std::invalid_argument); // dhat unset
    EXPECT_THROW(IPCTKContactEnergy(surface, options(0.0)), std::invalid_argument);
    EXPECT_THROW(IPCTKContactEnergy(surface, options(1.0, -1.0)), std::invalid_argument);
    IPCTKContactOptions no_ccd_budget = options(1.0);
    no_ccd_budget.ccd_max_iterations = 0;
    EXPECT_THROW(IPCTKContactEnergy(surface, no_ccd_budget), std::invalid_argument);
    EXPECT_THROW(IPCTKContactEnergy(mesh::TriangleMesh{Vec(), mesh::TriangleTopology(0, {})}, options(1.0)),
                 std::invalid_argument);
}

TEST(IPCTKContactEnergy, VanishesBeyondDhat) {
    const mesh::TriangleMesh surface = two_triangles(1.5);
    IPCTKContactEnergy energy(surface, options(1.0));
    EXPECT_EQ(energy.num_dofs(), 18);

    EXPECT_EQ(energy.num_active_pairs(surface.positions), 0);
    EXPECT_EQ(energy.min_active_distance(surface.positions), std::numeric_limits<double>::infinity());
    EXPECT_NO_THROW(energy.eval_hessian(surface.positions));
    EXPECT_EQ(energy.eval_value(surface.positions), 0.0);
    EXPECT_EQ(energy.eval_gradient(surface.positions), Vec::Zero(18));
    EXPECT_EQ(energy.eval_hessian(surface.positions).nonZeros(), 0);
}

TEST(IPCTKContactEnergy, ReportsTheActivePairAndItsDistance) {
    const mesh::TriangleMesh surface = two_triangles(0.5);
    IPCTKContactEnergy energy(surface, options(1.0));
    EXPECT_TRUE(energy.options().use_area_weighting);
    EXPECT_EQ(energy.num_active_pairs(surface.positions), 1);
    EXPECT_NEAR(energy.min_active_distance(surface.positions), 0.5, 1e-14);

    const auto snapshot = energy.contact_snapshot(surface.positions);
    EXPECT_EQ(snapshot.types, std::vector<std::string>{"VF"});
    EXPECT_EQ(snapshot.positions.reshaped<Eigen::RowMajor>(), surface.positions);
    EXPECT_EQ(snapshot.vertex_ids.row(0), (Eigen::RowVector4i(3, 0, 1, 2)));
    EXPECT_DOUBLE_EQ(snapshot.distances[0], 0.5);
    EXPECT_GT(snapshot.weights[0], 0);
    EXPECT_DOUBLE_EQ(snapshot.mollifiers[0], 1);
    EXPECT_EQ(energy.eval_hessian(surface.positions).nonZeros(), 144);

    // Pushes vertex 3 away from A and A's vertices the other way.
    const Vec gradient = energy.eval_gradient(surface.positions);
    EXPECT_GT(energy.eval_value(surface.positions), 0.0);
    EXPECT_LT(gradient[3 * 3 + 2], 0.0);
    EXPECT_GT(gradient[2] + gradient[5] + gradient[8], 0.0);
    EXPECT_NEAR(gradient.segment<3>(0).sum() + gradient.segment<3>(3).sum() + gradient.segment<3>(6).sum() +
                    gradient.segment<3>(9).sum(),
                0.0, 1e-12); // translation invariance
}

TEST(IPCTKContactEnergy, PatternFollowsTheActiveSet) {
    const mesh::TriangleMesh surface = two_triangles(1.5);
    IPCTKContactEnergy energy(surface, options(1.0));
    const Vec near = translate_b(surface.positions, Eigen::Vector3d(0, 0, -1));

    // The first Hessian request prepares the active structure.
    EXPECT_NO_THROW(energy.eval_hessian(near));
    EXPECT_NO_THROW(energy.eval_gradient(near));

    EXPECT_NO_THROW(energy.eval_hessian(near));
    // Vertex 3 and face A's vertices 0, 1, 2: a dense 12 x 12 block.
    EXPECT_EQ(energy.eval_hessian(near).nonZeros(), 144);
    EXPECT_NO_THROW(energy.eval_hessian(translate_b(near, Eigen::Vector3d(0, 0, 0.1))));
    EXPECT_NO_THROW(energy.eval_hessian(surface.positions));
    EXPECT_EQ(energy.eval_hessian(surface.positions).nonZeros(), 0);
}

TEST(IPCTKContactEnergy, DerivativesMatchFiniteDifferences) {
    // A single vertex-face pair, then several pairs of every edge and vertex kind.
    for (const double dhat : {1.0, 3.0}) {
        const mesh::TriangleMesh surface = two_triangles(0.5);
        IPCTKContactEnergy energy(surface, options(dhat));
        // Tilt B so that no pair is degenerate.
        Vec x = surface.positions;
        x.segment<3>(12) += Eigen::Vector3d(0.3, -0.2, 0.1);
        ASSERT_GE(energy.num_active_pairs(x), dhat == 1.0 ? 1 : 4);
        energy::test::expect_derivatives_match(energy, x, 1e-5);
    }
}

TEST(IPCTKContactEnergy, ProjectedHessianIsPositiveSemidefinite) {
    const mesh::TriangleMesh surface = two_triangles(0.5);
    IPCTKContactEnergy energy(surface, options(3.0));
    const Eigen::MatrixXd exact = Eigen::MatrixXd(energy.eval_hessian(surface.positions));
    const Eigen::MatrixXd projected =
        Eigen::MatrixXd(energy.eval_hessian(surface.positions, energy::HessianKind::Projected));
    const Eigen::VectorXd exact_eigenvalues = Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(exact).eigenvalues();
    const Eigen::VectorXd eigenvalues = Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(projected).eigenvalues();
    EXPECT_LT(exact_eigenvalues.minCoeff(), 0.0); // the projection has work to do
    EXPECT_GE(eigenvalues.minCoeff(), -1e-10 * eigenvalues.cwiseAbs().maxCoeff());
}

TEST(IPCTKContactEnergy, EvaluationIsDeterministic) {
    const mesh::TriangleMesh surface = two_triangles(0.5);
    IPCTKContactEnergy energy(surface, options(3.0));
    Vec gradient;
    SpMat hessian;
    const double value = litepgo::testing::evaluate_energy(energy, surface.positions, &gradient, &hessian);
    for (int repeat = 0; repeat < 20; ++repeat) {
        Vec again_gradient;
        SpMat again_hessian;
        EXPECT_EQ(litepgo::testing::evaluate_energy(energy, surface.positions, &again_gradient, &again_hessian), value);
        EXPECT_EQ(again_gradient, gradient);
        EXPECT_EQ(Eigen::MatrixXd(again_hessian), Eigen::MatrixXd(hessian));
    }
}

TEST(IPCTKContactEnergy, FeasibleStepStopsBeforeImpact) {
    const mesh::TriangleMesh surface = two_triangles(0.5);
    IPCTKContactEnergy energy(surface, options(1.0));
    const Vec x = surface.positions;

    // Dropping B by 2 would pass vertex 3 through A at t = 0.25.
    const Vec down = translate_b(Vec::Zero(18), Eigen::Vector3d(0, 0, -2));
    const double alpha = energy.max_feasible_step(x, down);
    EXPECT_GT(alpha, 0.0);
    EXPECT_LT(alpha, 0.25);
    EXPECT_GT(energy.min_active_distance(x + alpha * down), 0.0);
    EXPECT_NO_THROW(energy.eval_value(x + alpha * down));

    // Moving apart, or sliding sideways above A, is unrestricted.
    EXPECT_EQ(energy.max_feasible_step(x, -down), 1.0);
    EXPECT_EQ(energy.max_feasible_step(x, translate_b(Vec::Zero(18), Eigen::Vector3d(1, 1, 0))), 1.0);
}

TEST(IPCTKContactEnergy, FeasibleStepKeepsMinDistance) {
    const mesh::TriangleMesh surface = two_triangles(0.5);
    IPCTKContactEnergy energy(surface, options(1.0, 0.2));
    const Vec down = translate_b(Vec::Zero(18), Eigen::Vector3d(0, 0, -2));
    const double alpha = energy.max_feasible_step(surface.positions, down);
    EXPECT_GT(alpha, 0.0);
    EXPECT_GT(energy.min_active_distance(surface.positions + alpha * down), 0.2);
}

TEST(IPCTKContactEnergy, RejectsInfeasibleStates) {
    const mesh::TriangleMesh surface = two_triangles(0.5);
    IPCTKContactEnergy thick(surface, options(1.0, 0.6));
    EXPECT_THROW(thick.eval_value(surface.positions), std::domain_error);
    EXPECT_THROW(thick.max_feasible_step(surface.positions, Vec::Zero(18)), std::domain_error);

    // B's vertex 3 pokes through A, but sits 0.5 below it: eval does not see the
    // intersection, max_feasible_step does.
    IPCTKContactEnergy energy(surface, options(1.0));
    const Vec crossing = translate_b(surface.positions, Eigen::Vector3d(0, 0, -1));
    EXPECT_NO_THROW(energy.eval_value(crossing));
    EXPECT_THROW(energy.max_feasible_step(crossing, Vec::Zero(18)), std::domain_error);
}

// Displacement DOFs u drive the surface through E(u) = B(X + W u), with X and W
// from a SurfaceKinematics. Triangle A is fixed by the displacement map; B's vertices 3, 4, 5 take
// simulation vertices 2, 1, 0, before an unrelated DOF u[9].
TEST(IPCTKContactEnergy, ComposesWithASurfaceKinematics) {
    auto barrier = std::make_shared<IPCTKContactEnergy>(two_triangles(0.5), options(1.0));
    SpMat W(18, 10);
    for (int v = 0; v < 3; ++v)
        for (int c = 0; c < 3; ++c)
            W.insert(9 + 3 * v + c, 3 * (2 - v) + c) = 1;
    const dof::SurfaceKinematics kinematics(barrier->rest_positions(), W);
    energy::AffineMappedEnergy mapped(barrier, kinematics.displacement_map(), kinematics.rest_positions());

    Vec u = Vec::Zero(10);
    u[9] = 7.0;  // drives nothing
    u[8] = -0.2; // lowers vertex 3
    u[3] = 0.1;  // moves vertex 4 in x
    const Vec x = kinematics.positions(u);
    EXPECT_EQ(mapped.eval_value(u), barrier->eval_value(x));
    const Vec g = barrier->eval_gradient(x);
    const Vec gradient = mapped.eval_gradient(u);
    EXPECT_EQ(gradient[9], 0.0);
    EXPECT_EQ(gradient.segment<3>(6), g.segment<3>(9));
    EXPECT_EQ(gradient.segment<3>(3), g.segment<3>(12));
    EXPECT_EQ(gradient.segment<3>(0), g.segment<3>(15));
    EXPECT_NO_THROW(mapped.eval_hessian(u));
    energy::test::expect_derivatives_match(mapped, u, 1e-5);

    // Dropping B through the obstacle is caught on the mapped positions.
    Vec du = Vec::Zero(10);
    du[2] = du[5] = du[8] = -2.0;
    const double alpha = mapped.max_feasible_step(Vec::Zero(10), du);
    EXPECT_GT(alpha, 0.0);
    EXPECT_LT(alpha, 0.25);
}

} // namespace litepgo::contact::test
