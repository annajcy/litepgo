#include "../testing/evaluate_energy.hpp"
#include "litepgo/contact/broad_phase/spatial_hash.hpp"
#include "litepgo/contact/detail/distance.hpp"
#include "litepgo/contact/detail/distance_derivatives.hpp"
#include "litepgo/contact/ipc_contact_energy.hpp"

#include "../testing/finite_difference.hpp"
#include "litepgo/dof/surface_kinematics.hpp"
#include "litepgo/energy/scalar/affine_mapped_energy.hpp"

#include <Eigen/Eigenvalues>
#include <cstring>
#include <gtest/gtest.h>
#include <tbb/global_control.h>

#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
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

IPCContactOptions options(double dhat, double min_distance = 0.0) {
    IPCContactOptions result;
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

TEST(IPCContactEnergy, RejectsInvalidConfiguration) {
    static_assert(!std::is_constructible_v<IPCContactEnergy, const mesh::TriangleMesh&, IPCContactOptions>);
    const mesh::TriangleMesh surface = two_triangles(0.5);
    EXPECT_THROW(IPCContactEnergy(surface, options(1), nullptr), std::invalid_argument);
    EXPECT_THROW(IPCContactEnergy(surface, IPCContactOptions{}, std::make_shared<SpatialHash>()),
                 std::invalid_argument); // dhat unset
    EXPECT_THROW(IPCContactEnergy(surface, options(0.0), std::make_shared<SpatialHash>()), std::invalid_argument);
    EXPECT_THROW(IPCContactEnergy(surface, options(1.0, -1.0), std::make_shared<SpatialHash>()), std::invalid_argument);
    IPCContactOptions no_ccd_budget = options(1.0);
    no_ccd_budget.ccd_max_iterations = 0;
    EXPECT_THROW(IPCContactEnergy(surface, no_ccd_budget, std::make_shared<SpatialHash>()), std::invalid_argument);
    EXPECT_THROW(IPCContactEnergy(mesh::TriangleMesh{Vec(), mesh::TriangleTopology(0, {})}, options(1.0),
                                  std::make_shared<SpatialHash>()),
                 std::invalid_argument);
}

TEST(IPCContactEnergy, VanishesBeyondDhat) {
    const mesh::TriangleMesh surface = two_triangles(1.5);
    IPCContactEnergy energy(surface, options(1.0), std::make_shared<SpatialHash>());
    EXPECT_EQ(energy.num_dofs(), 18);

    EXPECT_EQ(energy.num_active_pairs(surface.positions), 0);
    EXPECT_EQ(energy.min_active_distance(surface.positions), std::numeric_limits<double>::infinity());
    EXPECT_NO_THROW(energy.evaluate(surface.positions)->hessian());
    EXPECT_EQ(energy.evaluate(surface.positions)->value(), 0.0);
    EXPECT_EQ(energy.evaluate(surface.positions)->gradient(), Vec::Zero(18));
    EXPECT_EQ(energy.evaluate(surface.positions)->hessian().nonZeros(), 0);
}

TEST(IPCContactEnergy, ReportsTheActivePairAndItsDistance) {
    const mesh::TriangleMesh surface = two_triangles(0.5);
    IPCContactEnergy energy(surface, options(1.0), std::make_shared<SpatialHash>());
    EXPECT_EQ(energy.num_active_pairs(surface.positions), 1);
    EXPECT_NEAR(energy.min_active_distance(surface.positions), 0.5, 1e-14);

    const auto snapshot = energy.contact_snapshot(surface.positions);
    EXPECT_EQ(snapshot.types, std::vector<std::string>{"VF"});
    EXPECT_EQ(snapshot.positions.reshaped<Eigen::RowMajor>(), surface.positions);
    EXPECT_EQ(snapshot.vertex_ids.row(0), (Eigen::RowVector4i(3, 0, 1, 2)));
    EXPECT_DOUBLE_EQ(snapshot.distances[0], 0.5);
    EXPECT_GT(snapshot.weights[0], 0);
    EXPECT_DOUBLE_EQ(snapshot.mollifiers[0], 1);
    EXPECT_EQ(energy.evaluate(surface.positions)->hessian().nonZeros(), 144);

    // Pushes vertex 3 away from A and A's vertices the other way.
    const Vec gradient = energy.evaluate(surface.positions)->gradient();
    EXPECT_GT(energy.evaluate(surface.positions)->value(), 0.0);
    EXPECT_LT(gradient[3 * 3 + 2], 0.0);
    EXPECT_GT(gradient[2] + gradient[5] + gradient[8], 0.0);
    EXPECT_NEAR(gradient.segment<3>(0).sum() + gradient.segment<3>(3).sum() + gradient.segment<3>(6).sum() +
                    gradient.segment<3>(9).sum(),
                0.0, 1e-12); // translation invariance
}

TEST(IPCContactEnergy, PatternFollowsTheActiveSet) {
    const mesh::TriangleMesh surface = two_triangles(1.5);
    IPCContactEnergy energy(surface, options(1.0), std::make_shared<SpatialHash>());
    const Vec near = translate_b(surface.positions, Eigen::Vector3d(0, 0, -1));

    // The first Hessian request prepares the active structure.
    EXPECT_NO_THROW(energy.evaluate(near)->hessian());
    EXPECT_NO_THROW(energy.evaluate(near)->gradient());

    EXPECT_NO_THROW(energy.evaluate(near)->hessian());
    // Vertex 3 and face A's vertices 0, 1, 2: a dense 12 x 12 block.
    EXPECT_EQ(energy.evaluate(near)->hessian().nonZeros(), 144);
    EXPECT_NO_THROW(energy.evaluate(translate_b(near, Eigen::Vector3d(0, 0, 0.1)))->hessian());
    EXPECT_NO_THROW(energy.evaluate(surface.positions)->hessian());
    EXPECT_EQ(energy.evaluate(surface.positions)->hessian().nonZeros(), 0);
}

TEST(IPCContactEnergy, DerivativesMatchFiniteDifferences) {
    // A single vertex-face pair, then several pairs of every edge and vertex kind.
    for (const double dhat : {1.0, 3.0}) {
        const mesh::TriangleMesh surface = two_triangles(0.5);
        IPCContactEnergy energy(surface, options(dhat), std::make_shared<SpatialHash>());
        // Tilt B so that no pair is degenerate.
        Vec x = surface.positions;
        x.segment<3>(12) += Eigen::Vector3d(0.3, -0.2, 0.1);
        ASSERT_GE(energy.num_active_pairs(x), dhat == 1.0 ? 1 : 4);
        energy::test::expect_derivatives_match(energy, x, 1e-5);
    }
}

TEST(IPCContactEnergy, AreaWeightScalesValueAndDerivativesUsingRestGeometry) {
    const auto surface = two_triangles(0.5);
    IPCContactEnergy weighted(surface, options(1), std::make_shared<SpatialHash>());
    EXPECT_TRUE(weighted.options().use_area_weighting);
    auto unweighted_options = options(1);
    unweighted_options.use_area_weighting = false;
    IPCContactEnergy unweighted(surface, unweighted_options, std::make_shared<SpatialHash>());
    Vec x = surface.positions;
    // Change B's current area without moving its only active vertex-face pair.
    x[14] += 2;
    const Eigen::Vector3d a = surface.positions.segment<3>(12) - surface.positions.segment<3>(9),
                          b = surface.positions.segment<3>(15) - surface.positions.segment<3>(9);
    const double weight = a.cross(b).norm() / 24; // quarter of vertex 3's one-third face area
    ASSERT_EQ(weighted.num_active_pairs(x), 1);
    EXPECT_DOUBLE_EQ(weighted.evaluate(x)->value(), weight * unweighted.evaluate(x)->value());
    EXPECT_TRUE(weighted.evaluate(x)->gradient().isApprox(weight * unweighted.evaluate(x)->gradient(), 1e-12));
    EXPECT_TRUE(Eigen::MatrixXd(weighted.evaluate(x)->hessian())
                    .isApprox(weight * Eigen::MatrixXd(unweighted.evaluate(x)->hessian()), 1e-12));
    EXPECT_TRUE(Eigen::MatrixXd(weighted.evaluate(x)->projected_hessian())
                    .isApprox(weight * Eigen::MatrixXd(unweighted.evaluate(x)->projected_hessian()), 1e-12));
    Vec dx = Vec::Zero(x.size());
    dx[11] = -2;
    EXPECT_EQ(weighted.evaluate(x)->max_feasible_step(dx), unweighted.evaluate(x)->max_feasible_step(dx));
}

TEST(IPCContactEnergy, AreaWeightedDerivativesMatchFiniteDifferences) {
    const auto surface = two_triangles(0.5);
    auto weighted_options = options(3, 0.1);
    weighted_options.use_area_weighting = true;
    IPCContactEnergy weighted(surface, weighted_options, std::make_shared<SpatialHash>());
    Vec x = surface.positions;
    x.segment<3>(12) += Eigen::Vector3d(0.3, -0.2, 0.1);
    ASSERT_GE(weighted.num_active_pairs(x), 4);
    energy::test::expect_derivatives_match(weighted, x, 1e-5);
}

TEST(IPCContactEnergy, ProjectedHessianIsPositiveSemidefinite) {
    const mesh::TriangleMesh surface = two_triangles(0.5);
    IPCContactEnergy energy(surface, options(3.0), std::make_shared<SpatialHash>());
    const Eigen::MatrixXd exact = Eigen::MatrixXd(energy.evaluate(surface.positions)->hessian());
    const Eigen::MatrixXd projected = Eigen::MatrixXd(energy.evaluate(surface.positions)->projected_hessian());
    const Eigen::VectorXd exact_eigenvalues = Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(exact).eigenvalues();
    const Eigen::VectorXd eigenvalues = Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(projected).eigenvalues();
    EXPECT_LT(exact_eigenvalues.minCoeff(), 0.0); // the projection has work to do
    EXPECT_GE(eigenvalues.minCoeff(), -1e-10 * eigenvalues.cwiseAbs().maxCoeff());
}

TEST(IPCContactEnergy, EvaluationIsDeterministic) {
    const mesh::TriangleMesh surface = two_triangles(0.5);
    IPCContactEnergy energy(surface, options(3.0), std::make_shared<SpatialHash>());
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

TEST(IPCContactEnergy, FeasibleStepStopsBeforeImpact) {
    const mesh::TriangleMesh surface = two_triangles(0.5);
    IPCContactEnergy energy(surface, options(1.0), std::make_shared<SpatialHash>());
    const Vec x = surface.positions;

    // Dropping B by 2 would pass vertex 3 through A at t = 0.25.
    const Vec down = translate_b(Vec::Zero(18), Eigen::Vector3d(0, 0, -2));
    const double alpha = energy.evaluate(x)->max_feasible_step(down);
    EXPECT_GT(alpha, 0.0);
    EXPECT_LT(alpha, 0.25);
    EXPECT_GT(energy.min_active_distance(x + alpha * down), 0.0);
    EXPECT_NO_THROW(energy.evaluate(x + alpha * down)->value());

    // Moving apart, or sliding sideways above A, is unrestricted.
    EXPECT_EQ(energy.evaluate(x)->max_feasible_step(-down), 1.0);
    EXPECT_EQ(energy.evaluate(x)->max_feasible_step(translate_b(Vec::Zero(18), Eigen::Vector3d(1, 1, 0))), 1.0);
}

TEST(IPCContactEnergy, FeasibleStepKeepsMinDistance) {
    const mesh::TriangleMesh surface = two_triangles(0.5);
    IPCContactEnergy energy(surface, options(1.0, 0.2), std::make_shared<SpatialHash>());
    const Vec down = translate_b(Vec::Zero(18), Eigen::Vector3d(0, 0, -2));
    const double alpha = energy.evaluate(surface.positions)->max_feasible_step(down);
    EXPECT_GT(alpha, 0.0);
    EXPECT_GT(energy.min_active_distance(surface.positions + alpha * down), 0.2);
}

TEST(IPCContactEnergy, RejectsInfeasibleStates) {
    const mesh::TriangleMesh surface = two_triangles(0.5);
    IPCContactEnergy thick(surface, options(1.0, 0.6), std::make_shared<SpatialHash>());
    EXPECT_THROW(thick.evaluate(surface.positions)->value(), std::domain_error);
    EXPECT_THROW(thick.evaluate(surface.positions)->max_feasible_step(Vec::Zero(18)), std::domain_error);

    // B's vertex 3 pokes through A, but sits 0.5 below it: eval does not see the
    // intersection, max_feasible_step does.
    IPCContactEnergy energy(surface, options(1.0), std::make_shared<SpatialHash>());
    const Vec crossing = translate_b(surface.positions, Eigen::Vector3d(0, 0, -1));
    EXPECT_NO_THROW(energy.evaluate(crossing)->value());
    EXPECT_THROW(energy.evaluate(crossing)->max_feasible_step(Vec::Zero(18)), std::domain_error);
}

// Displacement DOFs u drive the surface through E(u) = B(X + W u), with X and W
// from a SurfaceKinematics. Triangle A is fixed by the displacement map; B's vertices 3, 4, 5 take
// simulation vertices 2, 1, 0, before an unrelated DOF u[9].
TEST(IPCContactEnergy, ComposesWithASurfaceKinematics) {
    auto barrier =
        std::make_shared<IPCContactEnergy>(two_triangles(0.5), options(1.0), std::make_shared<SpatialHash>());
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
    EXPECT_EQ(mapped.evaluate(u)->value(), barrier->evaluate(x)->value());
    const Vec g = barrier->evaluate(x)->gradient();
    const Vec gradient = mapped.evaluate(u)->gradient();
    EXPECT_EQ(gradient[9], 0.0);
    EXPECT_EQ(gradient.segment<3>(6), g.segment<3>(9));
    EXPECT_EQ(gradient.segment<3>(3), g.segment<3>(12));
    EXPECT_EQ(gradient.segment<3>(0), g.segment<3>(15));
    EXPECT_NO_THROW(mapped.evaluate(u)->hessian());
    energy::test::expect_derivatives_match(mapped, u, 1e-5);

    // Dropping B through the obstacle is caught on the mapped positions.
    Vec du = Vec::Zero(10);
    du[2] = du[5] = du[8] = -2.0;
    const double alpha = mapped.evaluate(Vec::Zero(10))->max_feasible_step(du);
    EXPECT_GT(alpha, 0.0);
    EXPECT_LT(alpha, 0.25);
}

TEST(IPCContactEnergy, OriginalSquaredBarrierAndThickness) {
    for (double thickness : {0.0, 0.2}) {
        auto surface = two_triangles(0.5);
        auto unweighted_options = options(1, thickness);
        unweighted_options.use_area_weighting = false;
        IPCContactEnergy e(surface, unweighted_options, std::make_shared<SpatialHash>());
        double s = 0.25 - thickness * thickness, h = 1 + 2 * thickness;
        EXPECT_NEAR(e.evaluate(surface.positions)->value(), -(s - h) * (s - h) * std::log(s / h), 1e-13);
        energy::test::expect_derivatives_match(e, surface.positions, 1e-5);
        Vec boundary = translate_b(surface.positions, Eigen::Vector3d(0, 0, 0.5 + thickness));
        EXPECT_NEAR(e.evaluate(boundary)->value(), 0, 1e-14);
    }
}

TEST(IPCContactEnergy, PhysicalBarrierMatchesToolkitNormalization) {
    const auto surface = two_triangles(.5);
    for (double minimum : {0.0, .2}) {
        for (bool area : {false, true}) {
            auto config = options(.7, minimum);
            config.use_area_weighting = area;
            EXPECT_FALSE(config.use_physical_barrier);
            IPCContactEnergy raw(surface, config, std::make_shared<SpatialHash>());
            config.use_physical_barrier = true;
            IPCContactEnergy physical(surface, config, std::make_shared<SpatialHash>());
            const double h = .7 * (.7 + 2 * minimum), scale = .7 / (h * h);
            EXPECT_NEAR(physical.evaluate(surface.positions)->value(), scale * raw.evaluate(surface.positions)->value(),
                        1e-12);
            EXPECT_TRUE(physical.evaluate(surface.positions)
                            ->gradient()
                            .isApprox(scale * raw.evaluate(surface.positions)->gradient(), 1e-12));
            for (auto kind : {false, true})
                EXPECT_TRUE(
                    Eigen::MatrixXd((kind ? physical.evaluate(surface.positions)->projected_hessian()
                                          : physical.evaluate(surface.positions)->hessian()))
                        .isApprox(scale * Eigen::MatrixXd((kind ? raw.evaluate(surface.positions)->projected_hessian()
                                                                : raw.evaluate(surface.positions)->hessian())),
                                  1e-10));
            energy::test::expect_derivatives_match(physical, surface.positions, 1e-5);
            const Vec direction = translate_b(Vec::Zero(18), Eigen::Vector3d(0, 0, -2));
            EXPECT_EQ(physical.evaluate(surface.positions)->max_feasible_step(direction),
                      raw.evaluate(surface.positions)->max_feasible_step(direction));
        }
    }
}

TEST(IPCContactEnergy, RejectsUnrepresentablePhysicalNormalization) {
    for (double dhat : {1e-100, 1e100}) {
        auto config = options(dhat);
        config.use_physical_barrier = true;
        EXPECT_THROW(IPCContactEnergy(two_triangles(.5), config, std::make_shared<SpatialHash>()),
                     std::invalid_argument);
    }
}

TEST(IPCContactEnergy, RejectsDegenerateDuplicateAndUnusedGeometry) {
    auto surface = two_triangles(0.5);
    surface.positions.segment<3>(3) = surface.positions.segment<3>(0);
    EXPECT_THROW(IPCContactEnergy(surface, options(1), std::make_shared<SpatialHash>()), std::invalid_argument);
    auto valid = two_triangles(0.5);
    mesh::TriangleMesh duplicate{valid.positions, mesh::TriangleTopology(6, {{0, 1, 2}, {2, 1, 0}, {3, 4, 5}})};
    EXPECT_THROW(IPCContactEnergy(duplicate, options(1), std::make_shared<SpatialHash>()), std::invalid_argument);
    mesh::TriangleMesh unused{valid.positions, mesh::TriangleTopology(6, {{0, 1, 2}})};
    EXPECT_THROW(IPCContactEnergy(unused, options(1), std::make_shared<SpatialHash>()), std::invalid_argument);
    auto bad = options(1);
    bad.dhat = std::numeric_limits<double>::max();
    EXPECT_THROW(IPCContactEnergy(valid, bad, std::make_shared<SpatialHash>()), std::invalid_argument);
    IPCContactEnergy e(valid, options(1), std::make_shared<SpatialHash>());
    Vec x = valid.positions;
    x[0] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(e.evaluate(x)->value(), std::invalid_argument);
    EXPECT_THROW(e.num_active_pairs(Vec::Zero(3)), std::invalid_argument);
    EXPECT_THROW(e.evaluate(valid.positions)->max_feasible_step(x), std::invalid_argument);
}

TEST(IPCContactEnergy, ACCDBudgetToleranceAndFastCrossing) {
    auto surface = two_triangles(0.5);
    Vec dx = translate_b(Vec::Zero(18), Eigen::Vector3d(0, 0, -200));
    for (long budget : {1L, 2L, 1000L})
        for (double tolerance : {1e-2, 1e-12}) {
            auto o = options(1, 0.1);
            o.ccd_max_iterations = budget;
            o.ccd_tolerance = tolerance;
            IPCContactEnergy e(surface, o, std::make_shared<SpatialHash>());
            double t = e.evaluate(surface.positions)->max_feasible_step(dx);
            EXPECT_GT(t, 0);
            EXPECT_LT(t, 0.002);
            EXPECT_GT(e.min_active_distance(surface.positions + t * dx), 0.1);
            EXPECT_EQ(t, e.evaluate(surface.positions)->max_feasible_step(dx));
        }
    IPCContactEnergy e(surface, options(1), std::make_shared<SpatialHash>());
    Vec common = Vec::Zero(18);
    for (int i = 0; i < 6; ++i)
        common.segment<3>(3 * i) = Eigen::Vector3d(100, -40, 3);
    EXPECT_EQ(e.evaluate(surface.positions)->max_feasible_step(common), 1);
}

TEST(IPCContactEnergy, NearParallelMollifierDerivatives) {
    Vec x(18);
    x << -1, 0, 0, 1, 0, 0, 0, -4, -2, -0.7, -0.004, 0.2, 0.8, 0.008, 0.2, 0, 4, 2;
    mesh::TriangleMesh surface{x, mesh::TriangleTopology(6, {{0, 1, 2}, {3, 4, 5}})};
    IPCContactEnergy e(surface, options(0.5), std::make_shared<SpatialHash>());
    energy::test::expect_derivatives_match(e, x, 2e-5);
    EXPECT_TRUE(e.evaluate(x)->gradient().allFinite());
    const Eigen::MatrixXd h = e.evaluate(x)->hessian();
    EXPECT_LE((h - h.transpose()).norm(), 1e-10 * (1 + h.norm()));
}

// Brute-force original candidate sum checks broadphase coverage AND multiplicity.
// Independently enumerate every nonincident VF and EE, without AABBs or dedup.
TEST(IPCContactEnergy, MatchesBruteForceOriginalCandidateSum) {
    using namespace detail;
    auto surface = two_triangles(0.5);
    const Vec rest = surface.positions;
    std::array<std::array<int, 3>, 2> faces{{{0, 1, 2}, {3, 4, 5}}};
    std::array<std::array<int, 2>, 6> edges{{{0, 1}, {0, 2}, {1, 2}, {3, 4}, {3, 5}, {4, 5}}};
    auto local = [](const Vec& x, std::array<int, 4> ids) {
        Points p{};
        for (int i = 0; i < 4; ++i)
            for (int k = 0; k < 3; ++k)
                p[i][k] = x[3 * ids[i] + k];
        return p;
    };
    // Avoid exact closest-feature transitions: the distance is C1, not C2 there.
    for (double shift : {-0.31, 0.0, 1.2, 8.13}) {
        SCOPED_TRACE(shift);
        Vec x = translate_b(rest, Eigen::Vector3d(shift, 0.13, 0.07));
        x[13] += 0.173;
        auto unweighted_options = options(3, 0.1);
        unweighted_options.use_area_weighting = false;
        IPCContactEnergy e(surface, unweighted_options, std::make_shared<SpatialHash>());
        auto barrier = [](double d2) {
            double s = d2 - 0.01, h = 9.6;
            return s < h ? -(s - h) * (s - h) * std::log(s / h) : 0;
        };
        double expected = 0;
        for (int v = 0; v < 6; ++v)
            for (auto f : faces) {
                if (v == f[0] || v == f[1] || v == f[2])
                    continue;
                auto p = local(x, {v, f[0], f[1], f[2]});
                expected += barrier(distance(p, point_triangle(p)));
            }
        for (int i = 0; i < 6; ++i)
            for (int j = i + 1; j < 6; ++j) {
                auto a = edges[i], b = edges[j];
                if (a[0] == b[0] || a[0] == b[1] || a[1] == b[0] || a[1] == b[1])
                    continue;
                auto p = local(x, {a[0], a[1], b[0], b[1]});
                double eps = 1e-3 * (rest.segment<3>(3 * a[1]) - rest.segment<3>(3 * a[0])).squaredNorm() *
                             (rest.segment<3>(3 * b[1]) - rest.segment<3>(3 * b[0])).squaredNorm();
                auto n = cross(sub(p[1], p[0]), sub(p[3], p[2]));
                double z = dot(n, n) / eps;
                expected += barrier(distance(p, edge_edge(p))) * (z < 1 ? z * (2 - z) : 1);
            }
        EXPECT_NEAR(e.evaluate(x)->value(), expected, 1e-11 * (1 + expected));
        energy::test::expect_derivatives_match(e, x, 2e-5);
    }
}

TEST(IPCContactDistance, KnownClosestFeaturesAndAnalyticDerivatives) {
    using namespace detail;
    std::vector<std::pair<Points, Feature>> cases;
    Points p{{{0, 0, 1}, {0, 0, 0}, {2, 0, 0}, {0, 2, 0}}};
    EXPECT_EQ(point_triangle(p).n, 2);
    EXPECT_DOUBLE_EQ(distance(p, point_triangle(p)), 1);
    p[0] = {1, 0, 1};
    EXPECT_EQ(point_triangle(p).n, 3);
    EXPECT_DOUBLE_EQ(distance(p, point_triangle(p)), 1);
    cases.emplace_back(p, point_triangle(p));
    p[0] = {0.3, 0.4, 1};
    EXPECT_EQ(point_triangle(p).n, 4);
    EXPECT_DOUBLE_EQ(distance(p, point_triangle(p)), 1);
    cases.emplace_back(p, point_triangle(p));
    p = {{{-1, 0, 0}, {1, 0, 0}, {0, -1, 2}, {0, 1, 2}}};
    EXPECT_TRUE(edge_edge(p).line_line);
    EXPECT_DOUBLE_EQ(distance(p, edge_edge(p)), 4);
    cases.emplace_back(p, edge_edge(p));
    p = {{{0, 0, 0}, {1, 0, 0}, {2, 1, 0}, {3, 1, 0}}};
    EXPECT_EQ(edge_edge(p).n, 2);
    EXPECT_DOUBLE_EQ(distance(p, edge_edge(p)), 2);
    cases.emplace_back(p, edge_edge(p));
    for (auto [positions, feature] : cases) {
        LocalGradient gradient;
        LocalHessian hessian;
        distance_derivatives(positions, feature, gradient, &hessian);
        for (int i = 0; i < 12; ++i) {
            auto forward = positions, backward = positions;
            forward[i / 3][i % 3] += 1e-5;
            backward[i / 3][i % 3] -= 1e-5;
            EXPECT_NEAR(gradient[i], (distance(forward, feature) - distance(backward, feature)) / 2e-5, 1e-8);
            LocalGradient gf, gb;
            distance_derivatives(forward, feature, gf, nullptr);
            distance_derivatives(backward, feature, gb, nullptr);
            const LocalGradient column = (gf - gb) / 2e-5;
            EXPECT_LE((hessian.col(i) - column).norm(), 1e-7);
        }
    }
}

TEST(IPCContactEnergy, CoplanarOverlapAndEdgeCrossing) {
    Vec x(18);
    x << -2, -2, 0, 2, -2, 0, 0, 2, 0, -0.3, 0, 0, 0.3, 0, 0, 0, 0.5, 0;
    mesh::TriangleMesh surface{x, mesh::TriangleTopology(6, {{0, 1, 2}, {3, 4, 5}})};
    IPCContactEnergy overlap(surface, options(0.1), std::make_shared<SpatialHash>());
    EXPECT_THROW(overlap.evaluate(x)->max_feasible_step(Vec::Zero(18)), std::domain_error);
    // Perpendicular edges sweep through one another while all vertices remain
    // separated. No endpoint-only CCD check can detect this motion.
    x << -1, 0, 0, 1, 0, 0, 0, -3, -1, 0, -1, 1, 0, 1, 1, 3, 0, 2;
    surface.positions = x;
    IPCContactEnergy e(surface, options(0.2), std::make_shared<SpatialHash>());
    Vec dx = translate_b(Vec::Zero(18), Eigen::Vector3d(0, 0, -2));
    double t = e.evaluate(x)->max_feasible_step(dx);
    EXPECT_GT(t, 0);
    EXPECT_LT(t, 0.5);
    for (int i = 0; i <= 20; ++i)
        EXPECT_NO_THROW(e.evaluate(x + (t * i / 20) * dx)->max_feasible_step(Vec::Zero(18)));
}

// Many disjoint copies supply enough independent VF, EE, and mollifier
// work to exercise scheduling, while contacts within each copy share vertices.
TEST(IPCContactEnergy, BitwiseDeterministicAcrossThreadCounts) {
    for (int fixture = 0; fixture < 3; ++fixture) {
        auto base = two_triangles(0.5);
        if (fixture == 1)
            base.positions << -1, 0, 0, 1, 0, 0, 0, -4, -2, -0.7, -0.004, 0.2, 0.8, 0.008, 0.2, 0, 4, 2;
        if (fixture == 2)
            base.positions << -1, 0, 0, 1, 0, 0, 0, -3, -1, 0, -1, 1, 0, 1, 1, 3, 0, 2;
        constexpr int copies = 24;
        Vec x(18 * copies), dx = Vec::Zero(18 * copies);
        std::vector<std::array<int, 3>> faces;
        for (int c = 0; c < copies; ++c) {
            x.segment(18 * c, 18) = base.positions;
            for (int v = 0; v < 6; ++v)
                x[18 * c + 3 * v] += 30 * c;
            for (int v = 3; v < 6; ++v)
                dx[18 * c + 3 * v + 2] = -2 - 0.01 * c;
            faces.push_back({6 * c, 6 * c + 1, 6 * c + 2});
            faces.push_back({6 * c + 3, 6 * c + 4, 6 * c + 5});
        }
        mesh::TriangleMesh surface{x, mesh::TriangleTopology(6 * copies, faces)};
        IPCContactEnergy e(surface, options(3, 0.05), std::make_shared<SpatialHash>());
        ASSERT_GT(e.num_active_pairs(x), copies);
        for (auto kind : {false, true}) {
            double value, value_only, step;
            Vec gradient, gradient_only;
            SpMat hessian, hessian_only;
            for (int threads : {1, 4, 8}) {
                tbb::global_control limit(tbb::global_control::max_allowed_parallelism, threads);
                for (int repeat = 0; repeat < 4; ++repeat) {
                    Vec g;
                    SpMat h;
                    double v = litepgo::testing::evaluate_energy(e, x, &g, &h, kind);
                    double vo = e.evaluate(x)->value(), t = e.evaluate(x)->max_feasible_step(dx);
                    Vec go = e.evaluate(x)->gradient();
                    SpMat ho = (kind ? e.evaluate(x)->projected_hessian() : e.evaluate(x)->hessian());
                    ASSERT_GT(t, 0);
                    ASSERT_LT(t, 0.5);
                    if (threads == 1 && repeat == 0) {
                        value = v;
                        value_only = vo;
                        step = t;
                        gradient = g;
                        gradient_only = go;
                        hessian = h;
                        hessian_only = ho;
                    }
                    EXPECT_EQ(std::memcmp(&v, &value, sizeof(double)), 0);
                    EXPECT_EQ(std::memcmp(&vo, &value_only, sizeof(double)), 0);
                    EXPECT_EQ(std::memcmp(&t, &step, sizeof(double)), 0);
                    EXPECT_EQ(std::memcmp(g.data(), gradient.data(), g.size() * sizeof(double)), 0);
                    EXPECT_EQ(std::memcmp(go.data(), gradient_only.data(), go.size() * sizeof(double)), 0);
                    ASSERT_EQ(h.nonZeros(), hessian.nonZeros());
                    EXPECT_EQ(std::memcmp(h.outerIndexPtr(), hessian.outerIndexPtr(), (h.cols() + 1) * sizeof(int)), 0);
                    EXPECT_EQ(std::memcmp(h.innerIndexPtr(), hessian.innerIndexPtr(), h.nonZeros() * sizeof(int)), 0);
                    EXPECT_EQ(std::memcmp(h.valuePtr(), hessian.valuePtr(), h.nonZeros() * sizeof(double)), 0);
                    ASSERT_EQ(ho.nonZeros(), hessian_only.nonZeros());
                    EXPECT_EQ(std::memcmp(ho.valuePtr(), hessian_only.valuePtr(), ho.nonZeros() * sizeof(double)), 0);
                }
            }
        }
    }
}

TEST(IPCContactEnergy, OwnedResultsFollowDisappearingAndReturningActiveSets) {
    const auto surface = two_triangles(0.5);
    const Vec far = translate_b(surface.positions, Eigen::Vector3d(0, 0, 10));
    for (int threads : {1, 4, 8}) {
        tbb::global_control limit(tbb::global_control::max_allowed_parallelism, threads);
        IPCContactEnergy e(surface, options(1), std::make_shared<SpatialHash>());
        for (auto kind : {false, true}) {
            Vec expected_g;
            SpMat expected_h;
            const double expected_value =
                litepgo::testing::evaluate_energy(e, surface.positions, &expected_g, &expected_h, kind);
            Vec g = Vec::Constant(18, 123);
            SpMat h = expected_h;
            // Each result follows the active set as it disappears, then returns.
            EXPECT_EQ(litepgo::testing::evaluate_energy(e, far, &g, &h, kind), 0);
            EXPECT_TRUE(g.isZero(0));
            EXPECT_TRUE(Eigen::Map<const Vec>(h.valuePtr(), h.nonZeros()).isZero(0));
            EXPECT_EQ(litepgo::testing::evaluate_energy(e, surface.positions, &g, &h, kind), expected_value);
            EXPECT_EQ(std::memcmp(g.data(), expected_g.data(), g.size() * sizeof(double)), 0);
            EXPECT_EQ(std::memcmp(h.valuePtr(), expected_h.valuePtr(), h.nonZeros() * sizeof(double)), 0);
        }
        ASSERT_EQ(e.evaluate(far)->hessian().nonZeros(), 0);
        EXPECT_NO_THROW(e.evaluate(surface.positions)->hessian());
        EXPECT_TRUE(e.evaluate(surface.positions)->hessian().toDense().allFinite());
    }
}

TEST(IPCContactEnergy, ParallelErrorsPropagateAndAllowRecovery) {
    for (int threads : {1, 4, 8}) {
        tbb::global_control limit(tbb::global_control::max_allowed_parallelism, threads);
        auto surface = two_triangles(0.5);
        IPCContactEnergy e(surface, options(3, 0.1), std::make_shared<SpatialHash>());
        Vec bad = translate_b(surface.positions, Eigen::Vector3d(0, 0, -0.45));
        // These fail inside the parallel potential evaluation, not input validation.
        EXPECT_THROW(e.evaluate(bad)->value(), std::domain_error);
        EXPECT_THROW(e.evaluate(bad)->gradient(), std::domain_error);
        for (auto kind : {false, true})
            EXPECT_THROW((kind ? e.evaluate(bad)->projected_hessian() : e.evaluate(bad)->hessian()), std::domain_error);
        // Finite swept bounds, but the CCD speed norm overflows inside a worker.
        Vec huge = translate_b(Vec::Zero(18), Eigen::Vector3d(0, 0, -1e160));
        EXPECT_THROW(e.evaluate(surface.positions)->max_feasible_step(huge), std::domain_error);
        IPCContactEnergy empty(surface, options(3), std::make_shared<SpatialHash>());
        EXPECT_NO_THROW(empty.evaluate(surface.positions)->hessian());
        EXPECT_TRUE(e.evaluate(surface.positions)->gradient().allFinite());
        EXPECT_GT(
            e.evaluate(surface.positions)->max_feasible_step(translate_b(Vec::Zero(18), Eigen::Vector3d(0, 0, -2))), 0);
    }
}

TEST(IPCContactEnergy, RequestedDerivativeOrdersAgree) {
    for (int fixture = 0; fixture < 2; ++fixture) {
        auto surface = two_triangles(0.5);
        if (fixture == 1)
            surface.positions << -1, 0, 0, 1, 0, 0, 0, -4, -2, -0.7, -0.004, 0.2, 0.8, 0.008, 0.2, 0, 4, 2;
        IPCContactEnergy e(surface, options(0.7, 0.05), std::make_shared<SpatialHash>());
        const Vec x = surface.positions;
        Vec g;
        SpMat h;
        const double value = litepgo::testing::evaluate_energy(e, x, &g, &h, false);
        const Vec g_only = e.evaluate(x)->gradient();
        const SpMat h_only = e.evaluate(x)->hessian();
        EXPECT_EQ(value, e.evaluate(x)->value());
        EXPECT_EQ(std::memcmp(g.data(), g_only.data(), sizeof(double) * g.size()), 0);
        ASSERT_EQ(h.nonZeros(), h_only.nonZeros());
        EXPECT_EQ(std::memcmp(h.valuePtr(), h_only.valuePtr(), sizeof(double) * h.nonZeros()), 0);
    }
}

} // namespace litepgo::contact::test
