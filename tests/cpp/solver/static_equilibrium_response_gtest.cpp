#include "../testing/equilibrium_force.hpp"
#include "litepgo/energy/scalar/quadratic_energy.hpp"
#include "litepgo/fem/material/volumetric/neo_hookean.hpp"
#include "litepgo/fem/quadrature/3d/tet_centroid_quadrature.hpp"
#include "litepgo/fem/volumetric_fem_energy.hpp"
#include "litepgo/fem/volumetric_fem_force_material_response.hpp"
#include "litepgo/solver/static_equilibrium_response.hpp"
#include <gtest/gtest.h>

namespace litepgo::solver::test {
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
    const auto make_energy = [&](const Vec& p) -> std::shared_ptr<energy::Energy> {
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
    StaticEquilibriumResponse response(2, fixed, initial, options, make_energy,
                                       testing::force_factory(2, make_energy, partial));
    const Vec p = Eigen::Vector2d(2, 3);
    SpMat J = response.jacobian_pattern();
    const Vec u = response.eval_value_and_jacobian_into(p, J);
    ASSERT_EQ(partial_calls, 1);
    EXPECT_GT(J.norm(), 1e-4);
    const Vec v = Eigen::Vector2d(.4, -.7);
    const Vec w = Vec::LinSpaced(12, -.8, 1.1);
    const Vec jv = response.eval_jvp(p, v), jtw = response.eval_vjp(p, w);
    EXPECT_LT((jv - J * v).norm(), 1e-10);
    EXPECT_LT((jtw - J.transpose() * w).norm(), 1e-10);
    EXPECT_NEAR(w.dot(jv), v.dot(jtw), 1e-10);
    const double eps = 1e-4;
    const Vec numerical_direction = (response.eval_value(p + eps * v) - response.eval_value(p - eps * v)) / (2 * eps);
    EXPECT_LT((jv - numerical_direction).norm(), 2e-7);
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
    const auto make_energy = [](const Vec& p) -> std::shared_ptr<energy::Energy> {
        return std::make_shared<energy::QuadraticEnergy>(Eigen::Matrix2d::Identity().sparseView(), -p);
    };
    const auto partial = [](const Vec&, const Vec&) -> SpMat { return (-Eigen::Matrix2d::Identity()).sparseView(); };
    const Vec p = Eigen::Vector2d(2, 3), initial = Eigen::Vector2d(.3, .4);
    const dof::DofReduction free(2, {}), fixed(2, {0, 1});
    EXPECT_THROW(StaticEquilibriumResponse(0, free, initial, options, make_energy,
                                           testing::force_factory(0, make_energy, partial)),
                 std::invalid_argument);
    EXPECT_THROW(StaticEquilibriumResponse(2, free, Vec::Zero(3), options, make_energy,
                                           testing::force_factory(2, make_energy, partial)),
                 std::invalid_argument);
    StaticEquilibriumResponse all_fixed(2, fixed, initial, options, make_energy,
                                        testing::force_factory(2, make_energy, partial));
    EXPECT_EQ(all_fixed.eval_value(p), initial);
    EXPECT_EQ(all_fixed.eval_jacobian(p).nonZeros(), 0);
    EXPECT_TRUE(all_fixed.eval_jvp(p, p).isZero());
    EXPECT_TRUE(all_fixed.eval_vjp(p, p).isZero());
    StaticEquilibriumResponse valid(2, free, initial, options, make_energy,
                                    testing::force_factory(2, make_energy, partial));
    EXPECT_TRUE(valid.eval_value(p).isApprox(p, 1e-12));
    EXPECT_TRUE(Eigen::MatrixXd(valid.eval_jacobian(p)).isApprox(Eigen::Matrix2d::Identity(), 1e-12));
    options.max_iterations = 0;
    StaticEquilibriumResponse unfinished(2, free, initial, options, make_energy,
                                         testing::force_factory(2, make_energy, partial));
    EXPECT_THROW(unfinished.eval_value(p), std::domain_error);
    EXPECT_THROW(unfinished.eval_jvp(p, p), std::domain_error);
    EXPECT_THROW(unfinished.eval_vjp(p, p), std::domain_error);
    options.max_iterations = 50;
    StaticEquilibriumResponse bad_partial(
        2, free, initial, options, make_energy,
        testing::force_factory(2, make_energy, [](const Vec&, const Vec&) { return SpMat(1, 2); }));
    EXPECT_THROW(bad_partial.eval_jacobian(p), std::invalid_argument);
    EXPECT_THROW(bad_partial.eval_jvp(p, p), std::invalid_argument);
    EXPECT_THROW(bad_partial.eval_vjp(p, p), std::invalid_argument);
    StaticEquilibriumResponse null_energy(
        2, free, initial, options, [](const Vec&) { return std::shared_ptr<energy::Energy>{}; },
        testing::force_factory(2, [](const Vec&) { return std::shared_ptr<energy::Energy>{}; }, partial));
    EXPECT_THROW(null_energy.eval_value(p), std::invalid_argument);
    EXPECT_THROW(null_energy.eval_jvp(p, p), std::invalid_argument);
    EXPECT_THROW(null_energy.eval_vjp(p, p), std::invalid_argument);
    StaticEquilibriumResponse singular(
        2, free, initial, options,
        [](const Vec&) -> std::shared_ptr<energy::Energy> {
            Eigen::Matrix2d K = Eigen::Vector2d(1, 0).asDiagonal();
            return std::make_shared<energy::QuadraticEnergy>(K.sparseView());
        },
        testing::force_factory(
            2,
            [](const Vec&) -> std::shared_ptr<energy::Energy> {
                Eigen::Matrix2d K = Eigen::Vector2d(1, 0).asDiagonal();
                return std::make_shared<energy::QuadraticEnergy>(K.sparseView());
            },
            partial));
    EXPECT_THROW(singular.eval_jacobian(p), std::exception);
    EXPECT_THROW(singular.eval_jvp(p, p), std::exception);
    EXPECT_THROW(singular.eval_vjp(p, p), std::exception);
}
TEST(StaticEquilibriumResponse, InterleavedFixedDofsMatchResolveFiniteDifferences) {
    Eigen::Matrix3d K;
    K << 4, 1, .5, 1, 3, -.4, .5, -.4, 2;
    Eigen::Matrix<double, 3, 2> A;
    A << 1, -.5, .3, 2, -.7, 1.2;
    const dof::DofReduction fixed(3, {1});
    const Vec initial = Eigen::Vector3d(0, .8, 0);
    solver::NewtonOptions options;
    options.gradient_tolerance = 1e-12;
    StaticEquilibriumResponse response(
        2, fixed, initial, options,
        [&](const Vec& p) -> std::shared_ptr<energy::Energy> {
            return std::make_shared<energy::QuadraticEnergy>(K.sparseView(), -A * p);
        },
        testing::force_factory(
            2,
            [&](const Vec& p) -> std::shared_ptr<energy::Energy> {
                return std::make_shared<energy::QuadraticEnergy>(K.sparseView(), -A * p);
            },
            [&](const Vec&, const Vec&) -> SpMat { return (-A).sparseView(); }));
    const Vec p = Eigen::Vector2d(.6, -1.1);
    SpMat J = response.jacobian_pattern();
    const Vec u = response.eval_value_and_jacobian_into(p, J);
    EXPECT_EQ(u[1], initial[1]);
    EXPECT_EQ(Eigen::MatrixXd(J).row(1).norm(), 0);
    EXPECT_LT(fixed.reduce(Vec(K * u - A * p)).norm(), 1e-12);
    EXPECT_GT(std::abs(J.coeff(0, 0) - J.coeff(2, 0)), .1);
    for (double h : {1e-3, 1e-5}) {
        SCOPED_TRACE(h);
        for (int c = 0; c < 2; ++c) {
            SCOPED_TRACE(c);
            Vec plus = p, minus = p;
            plus[c] += h;
            minus[c] -= h;
            const Vec up = response.eval_value(plus), um = response.eval_value(minus);
            EXPECT_EQ(up[1], initial[1]);
            EXPECT_EQ(um[1], initial[1]);
            EXPECT_LT((Vec(J.col(c)) - (up - um) / (2 * h)).norm(), 1e-9);
        }
    }
}
TEST(StaticEquilibriumResponse, ResponseProductsAvoidExplicitParameterJacobian) {
    class Force final : public energy::FixedPatternResponse {
      public:
        Force() : FixedPatternResponse(Eigen::Matrix<double, 2, 3>::Ones().sparseView()) {}
        Eigen::Matrix<double, 2, 3> A = (Eigen::Matrix<double, 2, 3>() << 1, 2, -1, 3, -2, .5).finished();

      private:
        Vec eval_impl(const ConstVecRef&, SpMat*) const override {
            throw std::logic_error("explicit B requested");
        }
        Vec eval_jvp_impl(const ConstVecRef&, const ConstVecRef& v) override {
            return -A * v;
        }
        Vec eval_vjp_impl(const ConstVecRef&, const ConstVecRef& w) override {
            return -A.transpose() * w;
        }
    };
    auto force = std::make_shared<Force>();
    bool invalid = false;
    int calls = 0;
    StaticEquilibriumResponse eq(
        3, dof::DofReduction(2, {1}), Eigen::Vector2d(0, .4), {.gradient_tolerance = 1e-12},
        [force](const Vec& p) -> std::shared_ptr<energy::Energy> {
            return std::make_shared<energy::QuadraticEnergy>(Eigen::Matrix2d::Identity().sparseView(), -force->A * p);
        },
        [&](const Vec& u) -> std::shared_ptr<energy::Response> {
            ++calls;
            EXPECT_DOUBLE_EQ(u[1], .4);
            return invalid ? nullptr : force;
        });
    const Vec p = Eigen::Vector3d(.2, -.7, .4), w = Eigen::Vector2d(.3, .6);
    eq.eval_value(p);
    EXPECT_EQ(calls, 0);
    Eigen::Matrix<double, 2, 3> J = force->A;
    J.row(1).setZero();
    EXPECT_TRUE(eq.eval_jvp(p, p).isApprox(J * p));
    EXPECT_TRUE(eq.eval_vjp(p, w).isApprox(J.transpose() * w));
    EXPECT_THROW(eq.eval_jacobian(p), std::logic_error);
    invalid = true;
    EXPECT_THROW(eq.eval_vjp(p, w), std::invalid_argument);
    invalid = false;
    EXPECT_TRUE(eq.eval_vjp(p, w).isApprox(J.transpose() * w));
}
} // namespace litepgo::solver::test
