
#include "litepgo/dof/vertex_dof_map.hpp"
#include "litepgo/fem/basis/3d/hex_hermite_basis.hpp"
#include "litepgo/fem/basis/3d/hex_q1_basis.hpp"
#include "litepgo/fem/field/deformation_gradient_field.hpp"
#include "litepgo/fem/field/identity_field.hpp"
#include "litepgo/fem/field/interpolated_field.hpp"
#include "litepgo/fem/kinematics/basis_dof_transform.hpp"
#include "litepgo/fem/kinematics/volumetric_element.hpp"
#include "litepgo/fem/material/volumetric/neo_hookean.hpp"
#include "litepgo/fem/material/volumetric/volumetric_material_local_energy.hpp"
#include "litepgo/fem/quadrature/3d/gauss_legendre_3d.hpp"
#include "litepgo/fem/volumetric_fem_energy.hpp"

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "litepgo/fem/field/deformation_gradient_field.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>
#include <vector>

namespace litepgo::example::test {

inline mesh::HexTopology hex_hermite_topology() {
    // Shared face at X = 1: vertices 1, 2, 5, 6 reuse all 24 physical DOFs per vertex.
    return mesh::HexTopology(12, {{0, 1, 2, 3, 4, 5, 6, 7}, {1, 8, 9, 2, 5, 10, 11, 6}});
}

inline Vec hex_hermite_rest_positions() {
    // Two boxes of sizes (1,2,3) and (2,2,3) share the face at X = 1.
    Eigen::Matrix<double, 12, 3> X_nodes;
    X_nodes << 0, 0, 0, 1, 0, 0, 1, 2, 0, 0, 2, 0, 0, 0, 3, 1, 0, 3, 1, 2, 3, 0, 2, 3, 3, 0, 0, 3, 2, 0, 3, 0, 3, 3, 2,
        3;
    return X_nodes.reshaped<Eigen::RowMajor>();
}

inline auto make_hex_hermite_mesh_energy(const Vec& rest_positions) {
    const auto topology = hex_hermite_topology();
    const dof::VertexDofMap rest_position_map(topology, 3), displacement_map(topology, 24);
    using XField = fem::InterpolatedField3D<fem::HexQ1Basis>;
    std::vector<XField> rest_position_fields;
    rest_position_fields.reserve(topology.num_cells());
    for (int e = 0; e < topology.num_cells(); ++e)
        rest_position_fields.emplace_back(rest_position_map.gather(e, rest_positions));
    const fem::IdentityField3D Fp(Eigen::Matrix3d::Identity().reshaped());
    const fem::IdentityField3D theta(Eigen::Vector2d(2, 3));
    return fem::VolumetricFemEnergy<fem::NeoHookean, fem::HexHermiteBasis, fem::GaussLegendre3D<4>, XField,
                                    fem::IdentityField3D, fem::IdentityField3D>(
        displacement_map, std::move(rest_position_fields), std::vector<fem::IdentityField3D>(topology.num_cells(), Fp),
        std::vector<fem::IdentityField3D>(topology.num_cells(), theta));
}

namespace {

Vec affine_dofs(const Vec& X_nodes, const Eigen::Matrix3d& A, const Eigen::Vector3d& b = Eigen::Vector3d::Zero()) {
    Vec x = Vec::Zero(8 * X_nodes.size());
    for (int v = 0; v < X_nodes.size() / 3; ++v) {
        x.segment<3>(24 * v) = A * X_nodes.segment<3>(3 * v) + b;
        for (int d = 0; d < 3; ++d)
            x.segment<3>(24 * v + 3 * (d + 1)) = A.col(d);
    }
    return x;
}
} // namespace

TEST(HexHermiteExample, ReproducesAffineDisplacementAndVolumeWeightedEnergy) {
    const Vec X_nodes = hex_hermite_rest_positions();
    const auto topology = hex_hermite_topology();
    const dof::VertexDofMap geometry_map(topology, 3), displacement_map(topology, 24);
    Eigen::Matrix3d A;
    A << 0.1, 0.02, -0.03, 0.04, -0.05, 0.02, 0.01, 0.03, 0.08;
    const Eigen::Vector3d b(0.2, -0.1, 0.3);
    const Vec x = affine_dofs(X_nodes, A, b);
    auto energy = make_hex_hermite_mesh_energy(X_nodes);
    EXPECT_DOUBLE_EQ(energy.eval_value(Vec::Zero(x.size())), 0);
    EXPECT_NEAR(energy.eval_value(x),
                18 * fem::NeoHookean::eval_W(
                         {Eigen::Matrix3d::Identity() + A, Eigen::Matrix3d::Identity(), Eigen::Vector2d(2, 3)}),
                1e-11);
    const std::array<Eigen::Vector3d, 3> points{
        {Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(1, 1, 1), Eigen::Vector3d(0.2, 0.3, 0.4)}};
    for (int e = 0; e < 2; ++e) {
        const auto X_local = geometry_map.gather(e, X_nodes);
        const fem::InterpolatedField3D<fem::HexQ1Basis> X_field(X_local);
        const auto coefficients =
            fem::BasisDofTransform<fem::HexHermiteBasis>(X_field).to_coefficients(displacement_map.gather(e, x));
        const fem::InterpolatedField3D<fem::HexHermiteBasis> displacement(coefficients);
        const auto F = fem::DeformationGradientField3D(displacement, X_field);
        for (const auto& xi : points) {
            EXPECT_TRUE(displacement.eval_value(xi).isApprox(A * X_field.eval_value(xi) + b, 1e-13));
            EXPECT_TRUE(F.eval_value(xi).reshaped(3, 3).isApprox(Eigen::Matrix3d::Identity() + A, 1e-13));
        }
    }
}

TEST(HexHermiteExample, SharedFacesHaveContinuousValuesAndPhysicalGradientsInAllDirections) {
    for (int normal = 0; normal < 3; ++normal) {
        SCOPED_TRACE(normal);
        std::vector<mesh::HexTopology::Cell> cells(2);
        std::vector<Eigen::Vector3d> vertices;
        for (int e = 0; e < 2; ++e) {
            Eigen::Vector3d lengths(2, 3, 4), origin = Eigen::Vector3d::Zero();
            lengths[normal] = e + 1;
            origin[normal] = e;
            for (int v = 0; v < 8; ++v) {
                const Eigen::Vector3d X = origin + lengths.cwiseProduct(fem::HexQ1Basis::reference_node(v));
                const auto found = std::find(vertices.begin(), vertices.end(), X);
                cells[static_cast<std::size_t>(e)][static_cast<std::size_t>(v)] =
                    static_cast<int>(found - vertices.begin());
                if (found == vertices.end())
                    vertices.push_back(X);
            }
        }
        ASSERT_EQ(vertices.size(), 12);
        Vec X_nodes(36), x(288);
        for (int v = 0; v < 12; ++v)
            X_nodes.segment<3>(3 * v) = vertices[static_cast<std::size_t>(v)];
        for (int j = 0; j < x.size(); ++j)
            x[j] = 0.01 * std::sin(0.7 * j + 0.3);
        const mesh::HexTopology topology(12, cells);
        const dof::VertexDofMap geometry_map(topology, 3), displacement_map(topology, 24);
        const std::array<Eigen::MatrixXd, 2> X_local{geometry_map.gather(0, X_nodes), geometry_map.gather(1, X_nodes)};
        const std::array<fem::InterpolatedField3D<fem::HexQ1Basis>, 2> X_fields{
            fem::InterpolatedField3D<fem::HexQ1Basis>(X_local[0]),
            fem::InterpolatedField3D<fem::HexQ1Basis>(X_local[1])};
        const auto c0 =
            fem::BasisDofTransform<fem::HexHermiteBasis>(X_fields[0]).to_coefficients(displacement_map.gather(0, x));
        const auto c1 =
            fem::BasisDofTransform<fem::HexHermiteBasis>(X_fields[1]).to_coefficients(displacement_map.gather(1, x));
        const fem::InterpolatedField3D<fem::HexHermiteBasis> u0(c0), u1(c1);
        const auto F0 = fem::DeformationGradientField3D(u0, X_fields[0]);
        const auto F1 = fem::DeformationGradientField3D(u1, X_fields[1]);
        for (double s : {0.0, 0.23, 0.67, 1.0})
            for (double t : {0.0, 0.31, 0.8, 1.0}) {
                Eigen::Vector3d xi = Eigen::Vector3d::Zero();
                xi[(normal + 1) % 3] = s;
                xi[(normal + 2) % 3] = t;
                xi[normal] = 1;
                const Eigen::Vector3d p0_xi = xi;
                xi[normal] = 0;
                const Eigen::Vector3d p1_xi = xi;
                EXPECT_TRUE(X_fields[0].eval_value(p0_xi).isApprox(X_fields[1].eval_value(p1_xi), 1e-14));
                EXPECT_LT((u0.eval_value(p0_xi) - u1.eval_value(p1_xi)).norm(), 1e-13);
                EXPECT_LT((F0.eval_value(p0_xi).reshaped(3, 3) - F1.eval_value(p1_xi).reshaped(3, 3)).norm(), 1e-13);
            }
    }
}

TEST(HexHermiteExample, PhysicalDofGradientAndHessianMatchFiniteDifferences) {
    const Vec X_nodes = hex_hermite_rest_positions();
    auto energy = make_hex_hermite_mesh_energy(X_nodes);
    Vec x = affine_dofs(X_nodes, 0.1 * Eigen::Matrix3d::Identity());
    for (int j = 0; j < x.size(); ++j)
        x[j] += 0.005 * std::sin(0.7 * j + 0.3);
    ASSERT_EQ(x.size(), 288);
    Vec g;
    SpMat sparse_h = energy.hessian_pattern();
    const double value = energy.eval(x, &g, &sparse_h);
    const Eigen::MatrixXd H(sparse_h);
    ASSERT_EQ(g.size(), x.size());
    ASSERT_EQ(H.rows(), x.size());
    ASSERT_EQ(H.cols(), x.size());
    EXPECT_LT((H - H.transpose()).norm(), 1e-12 * (1 + H.norm()));
    EXPECT_NEAR(energy.eval_value(x), value, 1e-13);

    constexpr double h = 1e-4;
    Vec fd_g(x.size());
    Eigen::MatrixXd fd_h(x.size(), x.size());
    for (int j = 0; j < x.size(); ++j) {
        Vec plus = x, minus = x, plus_g, minus_g;
        plus[j] += h;
        minus[j] -= h;
        fd_g[j] = (energy.eval(plus, &plus_g, nullptr) - energy.eval(minus, &minus_g, nullptr)) / (2 * h);
        fd_h.col(j) = (plus_g - minus_g) / (2 * h);
    }
    EXPECT_LT((g - fd_g).norm() / (1 + g.norm()), 2e-7);
    EXPECT_LT((H - fd_h).norm() / (1 + H.norm()), 2e-7);

    Vec d(x.size()), v(x.size());
    for (int j = 0; j < x.size(); ++j) {
        d[j] = std::sin(j + 0.5);
        v[j] = std::cos(j + 0.2);
    }
    d.normalize();
    v.normalize();
    for (double step : {1e-3, 3e-4, 1e-4}) {
        SCOPED_TRACE(step);
        const double mixed = (energy.eval_value(x + step * d + step * v) - energy.eval_value(x + step * d - step * v) -
                              energy.eval_value(x - step * d + step * v) + energy.eval_value(x - step * d - step * v)) /
                             (4 * step * step);
        EXPECT_NEAR(d.dot(H * v), mixed, (10 * step * step + 1e-7) * (1 + H.norm()));
    }
    SpMat h_only = energy.hessian_pattern();
    EXPECT_NEAR(energy.eval(x, nullptr, &h_only), value, 1e-13);
    EXPECT_LT((Eigen::MatrixXd(h_only) - H).norm(), 1e-13 * (1 + H.norm()));
}

TEST(HexHermiteExample, RejectsUnsupportedGeometryAndIncorrectDofShapes) {
    for (int which = 0; which < 6; ++which) {
        SCOPED_TRACE(which);
        Vec X_nodes = hex_hermite_rest_positions();
        for (int v = 0; v < 12; ++v) {
            if (which == 0)
                X_nodes[3 * v] += 0.1 * X_nodes[3 * v + 1];
            if (which == 2)
                X_nodes[3 * v + 2] = 0;
            if (which == 3 || which == 4)
                X_nodes[3 * v] *= -1;
            if (which == 4)
                X_nodes[3 * v + 1] *= -1;
        }
        if (which == 1)
            X_nodes[3 * 6] += 0.1;
        if (which == 5)
            X_nodes[0] = std::numeric_limits<double>::quiet_NaN();
        EXPECT_THROW(make_hex_hermite_mesh_energy(X_nodes), std::invalid_argument);
    }
    EXPECT_THROW(make_hex_hermite_mesh_energy(Vec::Zero(35)), std::invalid_argument);
    const dof::VertexDofMap map(hex_hermite_topology(), 3);
    Eigen::MatrixXd nodes = map.gather(0, hex_hermite_rest_positions());
    const fem::BasisDofTransform<fem::HexHermiteBasis> transform{fem::InterpolatedField3D<fem::HexQ1Basis>(nodes)};
    EXPECT_THROW(transform.to_coefficients(Eigen::MatrixXd::Zero(8, 3)), std::invalid_argument);
    EXPECT_THROW(transform.to_coefficients(Eigen::MatrixXd::Constant(8, 24, std::numeric_limits<double>::infinity())),
                 std::invalid_argument);
    EXPECT_THROW(transform.to_coefficients(Eigen::MatrixXd::Constant(8, 24, std::numeric_limits<double>::quiet_NaN())),
                 std::invalid_argument);
    nodes.col(0) *= 1e100;
    const fem::BasisDofTransform<fem::HexHermiteBasis> large{fem::InterpolatedField3D<fem::HexQ1Basis>(nodes)};
    EXPECT_THROW(large.to_coefficients(Eigen::MatrixXd::Constant(8, 24, 1e250)), std::domain_error);
}

TEST(HexHermiteExample, FeasibleStepMapsPhysicalDerivativeDofsAndChecksAllQuadraturePoints) {
    const Vec X = hex_hermite_rest_positions(), zero = Vec::Zero(288);
    auto energy = make_hex_hermite_mesh_energy(X);
    EXPECT_EQ(energy.max_feasible_step(zero, zero), 1);
    EXPECT_EQ(energy.max_feasible_step(zero, affine_dofs(X, 0.1 * Eigen::Matrix3d::Identity())), 1);
    const Eigen::Matrix3d dF = Eigen::Vector3d(-4, -4, 0).asDiagonal();
    const Vec direction = affine_dofs(X, dF);
    EXPECT_TRUE(std::isfinite(energy.eval_value(direction)));
    const double alpha = energy.max_feasible_step(zero, direction);
    EXPECT_NEAR(alpha, 0.225, 1e-6);
    for (int i = 0; i <= 12; ++i)
        EXPECT_TRUE(std::isfinite(energy.eval_value((alpha * i / 12) * direction)));
    const Vec current = affine_dofs(X, 0.1 * Eigen::Matrix3d::Identity());
    EXPECT_NEAR(energy.max_feasible_step(current, direction), 0.2475, 1e-6);
    EXPECT_THROW(energy.max_feasible_step(affine_dofs(X, -Eigen::Matrix3d::Identity()), zero), std::domain_error);
}

TEST(HexHermiteExample, BasisTransformsUsePhysicalDofLayoutAndIndependentPullbacks) {
    static_assert(fem::BasisDofTransform<fem::TetP1Basis>::dofs_per_vertex == 3);
    static_assert(fem::BasisDofTransform<fem::HexQ1Basis>::dofs_per_vertex == 3);
    static_assert(fem::BasisDofTransform<fem::HexHermiteBasis>::dofs_per_vertex == 24);
    const dof::VertexDofMap map(hex_hermite_topology(), 3);
    const Eigen::MatrixXd nodes = map.gather(0, hex_hermite_rest_positions());
    const fem::InterpolatedField3D<fem::HexQ1Basis> X(nodes);
    const fem::BasisDofTransform<fem::HexQ1Basis> nodal(X);
    EXPECT_EQ(nodal.to_coefficients(nodes), nodes);
    const fem::BasisDofTransform<fem::HexHermiteBasis> hermite(X);
    Vec scales(192);
    const std::array<double, 8> slots{1, 1, 2, 3, 2, 3, 6, 6};
    for (int v = 0; v < 8; ++v)
        for (int slot = 0; slot < 8; ++slot)
            scales.segment<3>(24 * v + 3 * slot).setConstant(slots[slot]);
    const auto coeffs = hermite.to_coefficients(Eigen::MatrixXd::Ones(8, 24));
    EXPECT_EQ(Vec(coeffs.reshaped<Eigen::RowMajor>()), scales);
    Vec gradient = Vec::Ones(192);
    hermite.pullback(&gradient, nullptr);
    EXPECT_EQ(gradient, scales);
    Eigen::MatrixXd hessian = Eigen::MatrixXd::Ones(192, 192);
    hermite.pullback(nullptr, &hessian);
    EXPECT_TRUE(hessian.isApprox(scales * scales.transpose(), 0));
    EXPECT_NO_THROW(hermite.pullback(nullptr, nullptr));
    EXPECT_THROW(nodal.to_coefficients(Eigen::MatrixXd::Zero(8, 24)), std::invalid_argument);
}

TEST(HexHermiteExample, AcceptsHermiteReferenceFieldsRepresentingAffineBoxes) {
    const auto topology = hex_hermite_topology();
    const dof::VertexDofMap rest_position_map(topology, 3), displacement_map(topology, 24);
    const Vec positions = hex_hermite_rest_positions();
    using XField = fem::InterpolatedField3D<fem::HexHermiteBasis>;
    std::vector<XField> fields;
    for (int e = 0; e < topology.num_cells(); ++e) {
        const auto nodes = rest_position_map.gather(e, positions);
        const fem::InterpolatedField3D<fem::HexQ1Basis> q1(nodes);
        Eigen::MatrixXd coefficients = Eigen::MatrixXd::Zero(64, 3);
        for (int v = 0; v < 8; ++v) {
            coefficients.row(8 * v) = nodes.row(v);
            const auto jacobian = q1.eval_dvalue_dxi(fem::HexQ1Basis::reference_node(v));
            for (int axis = 0; axis < 3; ++axis)
                coefficients.row(8 * v + 1 + axis) = jacobian.col(axis).transpose();
        }
        fields.emplace_back(coefficients);
        const fem::BasisDofTransform<fem::HexHermiteBasis> from_hermite(fields.back()), from_q1(q1);
        EXPECT_TRUE(from_hermite.to_coefficients(Eigen::MatrixXd::Ones(8, 24))
                        .isApprox(from_q1.to_coefficients(Eigen::MatrixXd::Ones(8, 24)), 1e-12));
    }
    const fem::IdentityField3D Fp(Eigen::Matrix3d::Identity().reshaped()), theta(Eigen::Vector2d(2, 3));
    fem::VolumetricFemEnergy<fem::NeoHookean, fem::HexHermiteBasis, fem::GaussLegendre3D<4>, XField,
                             fem::IdentityField3D, fem::IdentityField3D>
        energy(displacement_map, std::move(fields), {Fp, Fp}, {theta, theta});
    auto reference = make_hex_hermite_mesh_energy(positions);
    const Vec u = affine_dofs(positions, 0.1 * Eigen::Matrix3d::Identity());
    EXPECT_NEAR(energy.eval_value(u), reference.eval_value(u), 1e-11);
    EXPECT_TRUE(energy.eval_gradient(u).isApprox(reference.eval_gradient(u), 1e-11));
    EXPECT_TRUE(Eigen::MatrixXd(energy.eval_hessian(u)).isApprox(Eigen::MatrixXd(reference.eval_hessian(u)), 1e-11));
    const Vec direction = affine_dofs(positions, -4 * Eigen::Matrix3d::Identity());
    EXPECT_NEAR(energy.max_feasible_step(u, direction), reference.max_feasible_step(u, direction), 1e-12);
}

TEST(HexHermiteExample, CachedKinematicsMatchesUncachedAcrossChangingDisplacements) {
    const Vec positions = hex_hermite_rest_positions();
    const auto topology = hex_hermite_topology();
    const dof::VertexDofMap rest_map(topology, 3), map(topology, 24);
    auto cached = make_hex_hermite_mesh_energy(positions);
    energy::MeshEnergy uncached(map, [&](int e, const Eigen::MatrixXd& u, Vec* g, Eigen::MatrixXd* H) {
        const fem::InterpolatedField3D<fem::HexQ1Basis> X(rest_map.gather(e, positions));
        const fem::BasisDofTransform<fem::HexHermiteBasis> transform(X);
        const fem::InterpolatedField3D<fem::HexHermiteBasis> displacement(transform.to_coefficients(u));
        const fem::DeformationGradientField3D F(displacement, X);
        const fem::IdentityField3D Fp(Eigen::Matrix3d::Identity().reshaped()), theta(Eigen::Vector2d(2, 3));
        const fem::VolumetricMaterialLocalEnergyF3D<fem::NeoHookean> local(F, Fp, theta);
        const double value = fem::VolumetricElement3D<fem::GaussLegendre3D<4>>(X).eval(local, g, H);
        transform.pullback(g, H);
        return value;
    });
    for (int pose : {0, 1, 2, 0}) {
        Vec u = affine_dofs(positions, 0.03 * pose * Eigen::Matrix3d::Identity());
        for (int j = 0; j < u.size(); ++j)
            u[j] += 0.002 * std::sin(j + pose);
        EXPECT_NEAR(cached.eval_value(u), uncached.eval_value(u), 1e-11);
        EXPECT_TRUE(cached.eval_gradient(u).isApprox(uncached.eval_gradient(u), 1e-11));
        EXPECT_TRUE(Eigen::MatrixXd(cached.eval_hessian(u)).isApprox(Eigen::MatrixXd(uncached.eval_hessian(u)), 1e-11));
    }
}

} // namespace litepgo::example::test
