#include "../../../testing/element_formula_energy.hpp"
#include "../../../testing/finite_difference.hpp"
#include "litepgo/dof/stencil_dof_map.hpp"
#include "litepgo/energy/scalar/mesh_energy.hpp"
#include "litepgo/fem/material/shell/koiter_stvk.hpp"
#include "litepgo/fem/material/shell/shell_material_local_energy.hpp"
#include "litepgo/fem/quadrature/2d/triangle_centroid_quadrature.hpp"
#include "litepgo/mesh/triangle_mid_edge_stencils.hpp"

#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace litepgo::fem::test {

namespace {
using Shell = ShellMaterialLocalEnergy2D<KoiterStVK>;
static_assert(std::is_base_of_v<LocalEnergy2D, Shell>);

Eigen::MatrixXd coordinates() {
    Eigen::Matrix<double, 6, 3> X;
    X << 0, 0, 0, 1.2, 0.1, 0.2, -0.1, 0.9, -0.1, 1.1, 1.3, 0.4, -1, 0.2, -0.3, 0.2, -1.1, 0.3;
    return X;
}

Eigen::MatrixXd displacements(int vertices) {
    Eigen::MatrixXd u(vertices, 3);
    for (int v = 0; v < vertices; ++v)
        for (int c = 0; c < 3; ++c)
            u(v, c) = 0.03 * std::sin(0.7 * (3 * v + c + 1));
    return u;
}

struct NonFiniteGradientMaterial {
    static constexpr int num_parameters = KoiterStVK::num_parameters;
    static constexpr auto eval_W = KoiterStVK::eval_W;
    static constexpr auto eval_d2W_dab2 = KoiterStVK::eval_d2W_dab2;
    static Eigen::VectorXd eval_dW_dab(const ShellMaterialInputs&) {
        return Eigen::VectorXd::Constant(6, std::numeric_limits<double>::infinity());
    }
};
struct NonFiniteHessianMaterial {
    static constexpr int num_parameters = KoiterStVK::num_parameters;
    static constexpr auto eval_W = KoiterStVK::eval_W;
    static constexpr auto eval_dW_dab = KoiterStVK::eval_dW_dab;
    static Eigen::MatrixXd eval_d2W_dab2(const ShellMaterialInputs&) {
        return Eigen::MatrixXd::Constant(6, 6, std::numeric_limits<double>::quiet_NaN());
    }
};

TEST(ShellMaterialLocalEnergy2D, NonFiniteMaterialDerivativesThrowDomainError) {
    const TriangleMidEdgeRestData rest(coordinates(), {3, 4, 5});
    const auto u = displacements(6);
    const ShellMaterialLocalEnergy2D<NonFiniteGradientMaterial> bad_gradient(rest, u, 0.2, Eigen::Vector2d(2, 3));
    const ShellMaterialLocalEnergy2D<NonFiniteHessianMaterial> bad_hessian(rest, u, 0.2, Eigen::Vector2d(2, 3));
    EXPECT_TRUE(std::isfinite(bad_gradient.eval_value(Eigen::Vector2d::Zero())));
    EXPECT_THROW(bad_gradient.eval_gradient(Eigen::Vector2d::Zero()), std::domain_error);
    EXPECT_THROW(bad_gradient.eval_hessian(Eigen::Vector2d::Zero()), std::domain_error);
    EXPECT_THROW(bad_hessian.eval_hessian(Eigen::Vector2d::Zero()), std::domain_error);
}

void check_derivatives(const TriangleMidEdgeRestData& rest) {
    const Eigen::MatrixXd u = displacements(static_cast<int>(rest.X_local.rows()));
    const Shell energy(rest, u, 0.2, Eigen::Vector2d(2, 3));
    const Eigen::Vector2d xi(1.0 / 3, 1.0 / 3);
    const auto g = energy.eval_gradient(xi);
    const auto H = energy.eval_hessian(xi);
    EXPECT_EQ(g.size(), u.size());
    EXPECT_EQ(H.rows(), u.size());
    EXPECT_EQ(H.cols(), u.size());
    EXPECT_TRUE(H.isApprox(H.transpose(), 1e-12));
    constexpr double step = 2e-6;
    for (int j = 0; j < u.size(); ++j) {
        auto plus = u, minus = u;
        plus(j / 3, j % 3) += step;
        minus(j / 3, j % 3) -= step;
        const Shell a(rest, plus, 0.2, Eigen::Vector2d(2, 3)), b(rest, minus, 0.2, Eigen::Vector2d(2, 3));
        EXPECT_NEAR(g[j], (a.eval_value(xi) - b.eval_value(xi)) / (2 * step), 2e-8);
        const Vec fd = (a.eval_gradient(xi) - b.eval_gradient(xi)) / (2 * step);
        EXPECT_LT((fd - H.col(j)).norm(), 2e-7);
    }
    Eigen::MatrixXd dab_dx;
    const auto k = eval_triangle_mid_edge_kinematics(rest.X_local + u, rest.opposite, &dab_dx);
    const ShellMaterialInputs p{k.a, k.b, rest.a_bar, rest.b_bar, 0.2, Eigen::Vector2d(2, 3)};
    const Eigen::MatrixXd gauss_newton = dab_dx.transpose() * KoiterStVK::eval_d2W_dab2(p) * dab_dx;
    EXPECT_GT((H - gauss_newton).norm(), 1e-5);
}

TEST(ShellMaterialLocalEnergy2D, CachesFixedReferenceFormsAndHasStressFreeRigidStates) {
    const auto X = coordinates();
    const TriangleMidEdgeRestData rest(X, {3, 4, 5});
    const auto reference = eval_triangle_mid_edge_kinematics(X, {3, 4, 5});
    EXPECT_EQ(rest.a_bar, reference.a);
    EXPECT_EQ(rest.b_bar, reference.b);
    const Eigen::Vector3d e1 = X.row(1) - X.row(0), e2 = X.row(2) - X.row(0);
    EXPECT_NEAR(rest.sqrt_det_a_bar, e1.cross(e2).norm(), 1e-14);
    const Shell zero(rest, Eigen::MatrixXd::Zero(6, 3), 0.2, Eigen::Vector2d(2, 3));
    const LocalEnergy2D& base = zero;
    const Eigen::Vector2d xi(0.2, 0.3);
    EXPECT_TRUE(base.has_gradient());
    EXPECT_TRUE(base.has_hessian());
    EXPECT_EQ(base.num_dofs(), 18);
    EXPECT_DOUBLE_EQ(base.eval_value(xi), 0);
    EXPECT_TRUE(base.eval_gradient(xi).isZero(1e-13));
    const Eigen::MatrixXd H = base.eval_hessian(xi);
    EXPECT_GE(Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(H).eigenvalues().minCoeff(), -1e-11);
    const Eigen::Matrix3d R = Eigen::AngleAxisd(0.4, Eigen::Vector3d(1, 2, -1).normalized()).toRotationMatrix();
    const Eigen::MatrixXd moved = (X * R.transpose()).rowwise() + Eigen::RowVector3d(0.3, -0.7, 1.1);
    const Shell rigid(rest, moved - X, 0.2, Eigen::Vector2d(2, 3));
    EXPECT_NEAR(rigid.eval_value(xi), 0, 1e-25);
    EXPECT_TRUE(rigid.eval_gradient(xi).isZero(1e-12));
}

TEST(ShellMaterialLocalEnergy2D, FullStencilDerivativesIncludeGeometricSecondDerivatives) {
    check_derivatives(TriangleMidEdgeRestData(coordinates(), {3, 4, 5}));
}

TEST(ShellMaterialLocalEnergy2D, BoundaryAndDeduplicatedStencilsUseTheirActualDofCounts) {
    check_derivatives(TriangleMidEdgeRestData(coordinates().topRows(3), {-1, -1, -1}));
    Eigen::Matrix<double, 4, 3> tetrahedron;
    tetrahedron << 0, 0, 0, 1.2, 0, 0, 0, 0.9, 0, 0.1, 0.2, 1.1;
    check_derivatives(TriangleMidEdgeRestData(tetrahedron, {3, 3, 3}));
}

TEST(ShellMaterialLocalEnergy2D, OwnsInputSnapshotsAndIsConstantInParameterCoordinates) {
    auto X = coordinates();
    TriangleMidEdgeRestData rest(X, {3, 4, 5});
    auto u = displacements(6);
    Vec theta = Eigen::Vector2d(2, 3);
    const Shell energy(rest, u, 0.2, theta);
    const Eigen::Vector2d xi(0.2, 0.3);
    const double W = energy.eval_value(xi);
    X.setZero();
    u.setZero();
    theta.setZero();
    rest.X_local.setZero();
    rest.a_bar.setZero();
    rest.b_bar.setZero();
    EXPECT_DOUBLE_EQ(energy.eval_value(xi), W);
    EXPECT_DOUBLE_EQ(energy.eval_value({1.2, -0.4}), W);
    const Shell copy = energy;
    EXPECT_EQ(copy.eval_gradient(xi), energy.eval_gradient(xi));
}

TEST(ShellMaterialLocalEnergy2D, AccumulatesDensityWithCallerAreaAndIndependentNullableOutputs) {
    const TriangleMidEdgeRestData rest(coordinates(), {3, 4, 5});
    const Shell energy(rest, displacements(6), 0.2, Eigen::Vector2d(2, 3));
    const Eigen::Vector2d xi(0.2, 0.3);
    const double W = energy.eval_value(xi);
    const auto g = energy.eval_gradient(xi);
    const auto H = energy.eval_hessian(xi);
    const double weight = 0.5 * rest.sqrt_det_a_bar;
    EXPECT_NEAR(energy.accumulate(xi, weight, nullptr, nullptr), weight * W, 1e-14);
    for (int request = 0; request < 4; ++request) {
        const bool want_g = request & 1, want_H = request & 2;
        Vec accumulated_g = Vec::Constant(18, 2);
        Eigen::MatrixXd accumulated_H = Eigen::MatrixXd::Constant(18, 18, 3);
        const double result =
            energy.accumulate(xi, -0.4, want_g ? &accumulated_g : nullptr, want_H ? &accumulated_H : nullptr);
        EXPECT_NEAR(result, -0.4 * W, 1e-14);
        const Vec expected_g = Vec::Constant(18, 2) + (want_g ? -0.4 : 0) * g;
        const Eigen::MatrixXd expected_H = Eigen::MatrixXd::Constant(18, 18, 3) + (want_H ? -0.4 : 0) * H;
        EXPECT_TRUE(accumulated_g.isApprox(expected_g, 1e-13));
        EXPECT_TRUE(accumulated_H.isApprox(expected_H, 1e-13));
    }
}

TEST(ShellMaterialLocalEnergy2D, RejectsMalformedInputsInvalidCoordinatesAndOutputShapes) {
    const TriangleMidEdgeRestData rest(coordinates(), {3, 4, 5});
    for (const Eigen::MatrixXd& u : {Eigen::MatrixXd::Zero(3, 3).eval(), Eigen::MatrixXd::Zero(6, 2).eval()})
        EXPECT_THROW((Shell(rest, u, 0.2, Eigen::Vector2d(2, 3))), std::invalid_argument);
    auto u = displacements(6);
    u(0, 0) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW((Shell(rest, u, 0.2, Eigen::Vector2d(2, 3))), std::invalid_argument);
    for (double t : {0., -1., std::numeric_limits<double>::infinity()})
        EXPECT_THROW((Shell(rest, displacements(6), t, Eigen::Vector2d(2, 3))), std::invalid_argument);
    for (int count : {0, 1, 3})
        EXPECT_THROW((Shell(rest, displacements(6), 0.2, Vec::Ones(count))), std::invalid_argument);
    const Shell energy(rest, displacements(6), 0.2, Eigen::Vector2d(2, 3));
    const Eigen::Vector2d bad_xi(0.2, std::numeric_limits<double>::quiet_NaN());
    EXPECT_THROW(energy.eval_value(bad_xi), std::invalid_argument);
    EXPECT_THROW(energy.eval_gradient(bad_xi), std::invalid_argument);
    EXPECT_THROW(energy.eval_hessian(bad_xi), std::invalid_argument);
    Vec g = Vec::Ones(18);
    Eigen::MatrixXd H = Eigen::MatrixXd::Ones(17, 17);
    EXPECT_THROW(energy.accumulate({0.2, 0.3}, 1, &g, &H), std::invalid_argument);
    EXPECT_TRUE(g.isOnes());
    EXPECT_TRUE(H.isOnes());
    g = Vec::Ones(17);
    EXPECT_THROW(energy.accumulate({0.2, 0.3}, 1, &g, nullptr), std::invalid_argument);
    EXPECT_THROW(energy.accumulate({0.2, 0.3}, std::numeric_limits<double>::infinity(), nullptr, nullptr),
                 std::invalid_argument);
    EXPECT_THROW((TriangleMidEdgeRestData(coordinates(), {0, 4, 5})), std::invalid_argument);
    EXPECT_THROW((TriangleMidEdgeRestData(Eigen::MatrixXd::Zero(3, 3), {-1, -1, -1})), std::domain_error);
}

TEST(ShellMaterialLocalEnergy2D, RejectsDegenerateCurrentGeometryAndInvalidMaterialDomain) {
    const TriangleMidEdgeRestData rest(coordinates(), {3, 4, 5});
    Eigen::MatrixXd u = Eigen::MatrixXd::Zero(6, 3);
    u.row(1) = rest.X_local.row(0) - rest.X_local.row(1);
    const Shell collapsed(rest, u, 0.2, Eigen::Vector2d(2, 3));
    EXPECT_THROW(collapsed.eval_value({0.2, 0.3}), std::domain_error);
    EXPECT_THROW(collapsed.eval_gradient({0.2, 0.3}), std::domain_error);
    EXPECT_THROW(collapsed.eval_hessian({0.2, 0.3}), std::domain_error);
    const Shell bad_material(rest, displacements(6), 0.2, Eigen::Vector2d(-3, 2));
    EXPECT_THROW(bad_material.eval_value({0.2, 0.3}), std::invalid_argument);
}

TEST(ShellMaterialLocalEnergy2D, MeshEnergyAssemblesIntegratedStencilDerivatives) {
    const mesh::TriangleTopology topology(6, {{0, 1, 2}, {2, 1, 3}, {0, 2, 4}, {1, 0, 5}});
    const auto stencils = mesh::make_triangle_mid_edge_stencils(topology);
    const dof::StencilDofMap map(topology.num_vertices(), stencils.vertices, 3);
    Eigen::Matrix<double, 6, 3> X;
    X << 0, 0, 0, 2, 0, 0, 0, 1, 0, 2, 2, 0, -1, 0, 0, 0, -2, 0;
    const Vec X_packed = X.reshaped<Eigen::RowMajor>();
    std::vector<TriangleMidEdgeRestData> rest;
    for (int f = 0; f < map.num_elements(); ++f)
        rest.emplace_back(map.gather(f, X_packed), stencils.opposite[static_cast<std::size_t>(f)]);
    litepgo::testing::ElementFormulaEnergy energy(
        map,
        [rest](int f, const Eigen::MatrixXd& u_local, Vec* g, Eigen::MatrixXd* H) {
            const auto& r = rest[static_cast<std::size_t>(f)];
            const Shell local(r, u_local, 0.2, Eigen::Vector2d(2, 3));
            if (g)
                g->setZero(local.num_dofs());
            if (H)
                H->setZero(local.num_dofs(), local.num_dofs());
            using Q = TriangleCentroidQuadrature;
            return local.accumulate(Q::points().row(0).transpose(), Q::weights()[0] * r.sqrt_det_a_bar, g, H);
        },
        [](int, const auto&, const auto&) -> double {
            throw std::logic_error("This evaluation test does not provide geometric feasible-step bounds");
        });
    const Vec u = displacements(6).reshaped<Eigen::RowMajor>();
    energy::test::expect_derivatives_match(energy, u, 2e-6);
    const auto H = energy.eval_hessian(u);
    EXPECT_GT(std::abs(H.coeff(3 * 3 + 2, 3 * 4 + 2)), 1e-8);
    const double uniform_density = 0.2 / 2 * 5 * 0.21 * 0.21;
    EXPECT_NEAR(energy.eval_value(0.1 * X_packed), 5.5 * uniform_density, 1e-13);
    EXPECT_DOUBLE_EQ(energy.eval_value(Vec::Zero(18)), 0);
    EXPECT_FALSE(energy.update_pattern(u));
}
} // namespace

} // namespace litepgo::fem::test
