#include "litepgo/energy/response/static_equilibrium_response.hpp"
#include "litepgo/energy/scalar/quadratic_energy.hpp"
#include "litepgo/fem/material/volumetric/neo_hookean.hpp"
#include "litepgo/fem/quadrature/3d/tet_centroid_quadrature.hpp"
#include "litepgo/fem/volumetric_fem_energy.hpp"
#include "litepgo/fem/volumetric_fem_force_material_response.hpp"
#include <gtest/gtest.h>

namespace litepgo::energy::test {
TEST(StaticEquilibriumResponse, FemDisplacementAndReactionMatchResolveFiniteDifferences) {
    using namespace fem;
    const mesh::TetTopology topology(4, {{0, 1, 2, 3}});
    const dof::VertexDofMap map(topology, 3);
    using XField = InterpolatedField3D<TetP1Basis>;
    using Field = IdentityField3D;
    Eigen::Matrix<double, 4, 3> X;
    X << 0, 0, 0, 1.2, 0, 0, 0, .8, 0, 0, 0, 1.5;
    const std::vector<XField> geometry{XField(X)};
    Eigen::Matrix3d Fp = Eigen::Matrix3d::Identity();
    Fp(0, 1) = .08;
    Fp(2, 2) = 1.1;
    const std::vector<Field> plastic{Field(Vec(Fp.reshaped()))};
    using Elastic = VolumetricFemEnergy<NeoHookean, TetP1Basis, TetCentroidQuadrature, XField, Field, Field>;
    using Force = VolumetricFemForceMaterialResponse<NeoHookean, TetP1Basis, TetCentroidQuadrature, XField, Field>;
    const dof::DofReduction fixed(12, {0, 1, 2, 3, 4, 5, 6, 7, 8});
    Vec initial = Vec::Zero(12);
    initial[3] = .15;
    initial[7] = -.04;
    const auto make_energy = [&](const Vec& p) -> std::shared_ptr<Energy> {
        return std::make_shared<Elastic>(map, geometry, plastic, std::vector<Field>{Field(p)});
    };
    int partial_calls = 0;
    const auto partial = [&](const Vec& p, const Vec& u) -> SpMat {
        ++partial_calls;
        return Force(dof::ConstantDofMap(topology, 2), map, geometry, u, plastic).eval_jacobian(p);
    };
    solver::NewtonOptions options;
    options.gradient_tolerance = 1e-12;
    options.max_iterations = 100;
    StaticEquilibriumResponse response(2, fixed, initial, options, make_energy, partial);
    const Vec p = Eigen::Vector2d(2, 3);
    SpMat J = response.jacobian_pattern();
    const Vec u = response.eval(p, &J);
    ASSERT_EQ(partial_calls, 1);
    EXPECT_GT(J.norm(), 1e-4);
    for (int r : fixed.fixed_dofs()) {
        EXPECT_EQ(u[r], initial[r]);
        EXPECT_EQ(Eigen::MatrixXd(J).row(r).norm(), 0);
    }
    const auto potential = make_energy(p);
    const Eigen::MatrixXd reaction_J = Eigen::MatrixXd(partial(p, u)) + potential->eval_hessian(u) * J;
    EXPECT_LT(reaction_J.bottomRows(3).norm(), 1e-10);
    const int calls_before = partial_calls;
    for (double h : {1e-3, 1e-4}) {
        SCOPED_TRACE(h);
        for (int c = 0; c < 2; ++c) {
            SCOPED_TRACE(c);
            Vec plus = p, minus = p;
            plus[c] += h;
            minus[c] -= h;
            const Vec up = response.eval_value(plus), um = response.eval_value(minus);
            EXPECT_LT((Vec(J.col(c)) - (up - um) / (2 * h)).norm(), 2e-7);
            const Vec numerical =
                (make_energy(plus)->eval_gradient(up) - make_energy(minus)->eval_gradient(um)) / (2 * h);
            EXPECT_LT((reaction_J.col(c) - numerical).norm(), 2e-7);
        }
    }
    EXPECT_EQ(partial_calls, calls_before); // Value-only evaluations never request partial derivatives.
    EXPECT_EQ(response.eval_value(p), u);   // Perturbed evaluations do not warm-start the next solve.
}

TEST(StaticEquilibriumResponse, ValidationFailureAndFullyFixedStates) {
    solver::NewtonOptions options;
    options.gradient_tolerance = 1e-12;
    const auto make_energy = [](const Vec& p) -> std::shared_ptr<Energy> {
        return std::make_shared<QuadraticEnergy>(Eigen::Matrix2d::Identity().sparseView(), -p);
    };
    const auto partial = [](const Vec&, const Vec&) -> SpMat { return (-Eigen::Matrix2d::Identity()).sparseView(); };
    const Vec p = Eigen::Vector2d(2, 3), initial = Eigen::Vector2d(.3, .4);
    const dof::DofReduction free(2, {}), fixed(2, {0, 1});
    EXPECT_THROW(StaticEquilibriumResponse(0, free, initial, options, make_energy, partial), std::invalid_argument);
    EXPECT_THROW(StaticEquilibriumResponse(2, free, Vec::Zero(3), options, make_energy, partial),
                 std::invalid_argument);
    StaticEquilibriumResponse all_fixed(2, fixed, initial, options, make_energy, partial);
    EXPECT_EQ(all_fixed.eval_value(p), initial);
    EXPECT_EQ(all_fixed.eval_jacobian(p).nonZeros(), 0);
    StaticEquilibriumResponse valid(2, free, initial, options, make_energy, partial);
    EXPECT_TRUE(valid.eval_value(p).isApprox(p, 1e-12));
    EXPECT_TRUE(Eigen::MatrixXd(valid.eval_jacobian(p)).isApprox(Eigen::Matrix2d::Identity(), 1e-12));
    options.max_iterations = 0;
    StaticEquilibriumResponse unfinished(2, free, initial, options, make_energy, partial);
    EXPECT_THROW(unfinished.eval_value(p), std::domain_error);
    options.max_iterations = 50;
    StaticEquilibriumResponse bad_partial(2, free, initial, options, make_energy,
                                          [](const Vec&, const Vec&) { return SpMat(1, 2); });
    EXPECT_THROW(bad_partial.eval_jacobian(p), std::invalid_argument);
    StaticEquilibriumResponse null_energy(
        2, free, initial, options, [](const Vec&) { return std::shared_ptr<Energy>{}; }, partial);
    EXPECT_THROW(null_energy.eval_value(p), std::invalid_argument);
    StaticEquilibriumResponse singular(
        2, free, initial, options,
        [](const Vec&) -> std::shared_ptr<Energy> {
            Eigen::Matrix2d K = Eigen::Vector2d(1, 0).asDiagonal();
            return std::make_shared<QuadraticEnergy>(K.sparseView());
        },
        partial);
    EXPECT_THROW(singular.eval_jacobian(p), std::exception);
}
} // namespace litepgo::energy::test
