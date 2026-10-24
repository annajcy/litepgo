#include "litepgo/fem/hex_q1_basis.h"
#include "litepgo/fem/interpolated_field.h"
#include "litepgo/fem/vertex_element_dof_map.h"

#include <gtest/gtest.h>

#include <array>
#include <concepts>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace {

using litepgo::HexTopology;
using litepgo::Vec;
using litepgo::VertexDofLayout;
using litepgo::VertexElementDofMap;
using Map3 = VertexElementDofMap<8, 3>;
using Map2 = VertexElementDofMap<8, 2>;
using Basis = litepgo::HexQ1Basis;
using Reference = litepgo::VolumeReferenceGeometry<Basis>;
using VectorField = litepgo::InterpolatedField<Basis, 3>;
using ParameterField = litepgo::InterpolatedField<Basis, 2>;

template <int N, int D>
concept CanFormMap = requires { typename VertexElementDofMap<N, D>; };

static_assert(CanFormMap<8, 3>);
static_assert(!CanFormMap<0, 3>);
static_assert(!CanFormMap<8, 0>);
static_assert(!CanFormMap<8, -1>);
static_assert(!CanFormMap<std::numeric_limits<int>::max(), 2>);
static_assert(
    std::same_as<decltype(VertexElementDofMap(std::declval<const HexTopology&>(), VertexDofLayout<3>(0))), Map3>);
static_assert(std::same_as<Map3::LocalCoefficients, VectorField::LocalCoefficients>);
static_assert(std::same_as<Map2::LocalCoefficients, ParameterField::LocalCoefficients>);
static_assert(!std::is_constructible_v<Map3, HexTopology&&, VertexDofLayout<3>>);
static_assert(!std::is_constructible_v<Map3, const HexTopology&, VertexDofLayout<2>>);

// Adjacent cells with a shared face, in Q1 corner order and shuffled global numbering.
HexTopology two_hexes() {
    return HexTopology(12, {{1, 3, 9, 4, 7, 6, 0, 10}, {3, 8, 5, 9, 6, 2, 11, 0}});
}

Eigen::Matrix3d reference_jacobian() {
    Eigen::Matrix3d J;
    J << 2, 0.3, 0.1, 0.2, 1.5, 0.1, 0, 0.2, 1.1;
    return J;
}

Eigen::Matrix<double, 12, 3> rest_positions() {
    Eigen::Matrix<double, 12, 3> nodes;
    nodes << 1, 1, 1, 0, 0, 0, 2, 0, 1, 1, 0, 0, 0, 1, 0, 2, 1, 0, 1, 0, 1, 0, 0, 1, 2, 0, 0, 1, 1, 0, 0, 1, 1, 2, 1, 1;
    return (nodes * reference_jacobian().transpose()).rowwise() + Eigen::RowVector3d(1.2, -0.7, 2.1);
}

} // namespace

TEST(VertexElementDofMap, GathersThreeIndependentFieldsInCellAndSlotOrder) {
    const auto topology = two_hexes();
    const VertexDofLayout<3> u_layout(12);
    const VertexDofLayout<3> up_layout(12);
    const VertexDofLayout<2> theta_layout(12);
    const VertexElementDofMap u_map(topology, u_layout);
    const VertexElementDofMap up_map(topology, up_layout);
    const VertexElementDofMap theta_map(topology, theta_layout);
    EXPECT_EQ(u_map.num_field_dofs(), 36);
    EXPECT_EQ(up_map.num_field_dofs(), 36);
    EXPECT_EQ(theta_map.num_field_dofs(), 24);

    Vec a(36), b(36), theta(24);
    for (int i = 0; i < 36; ++i) {
        a[i] = 100.0 + i;
        b[i] = -1000.0 - 2.0 * i;
    }
    for (int i = 0; i < 24; ++i) {
        theta[i] = 2.0 + 0.1 * i;
    }

    const std::array<Map3::Indices, 2> expected3{{
        {3, 4, 5, 9, 10, 11, 27, 28, 29, 12, 13, 14, 21, 22, 23, 18, 19, 20, 0, 1, 2, 30, 31, 32},
        {9, 10, 11, 24, 25, 26, 15, 16, 17, 27, 28, 29, 18, 19, 20, 6, 7, 8, 33, 34, 35, 0, 1, 2},
    }};
    const std::array<Map2::Indices, 2> expected2{{
        {2, 3, 6, 7, 18, 19, 8, 9, 14, 15, 12, 13, 0, 1, 20, 21},
        {6, 7, 16, 17, 10, 11, 18, 19, 12, 13, 4, 5, 22, 23, 0, 1},
    }};
    for (int e = 0; e < 2; ++e) {
        const auto index = static_cast<std::size_t>(e);
        EXPECT_EQ(u_map.dof_indices(e), expected3[index]);
        EXPECT_EQ(up_map.dof_indices(e), expected3[index]);
        EXPECT_EQ(theta_map.dof_indices(e), expected2[index]);
        const auto a_e = u_map.gather(e, a);
        const auto b_e = up_map.gather(e, b);
        const auto theta_e = theta_map.gather(e, theta);
        for (int i = 0; i < 24; ++i) {
            EXPECT_DOUBLE_EQ(a_e[i], a[expected3[index][static_cast<std::size_t>(i)]]);
            EXPECT_DOUBLE_EQ(b_e[i], b[expected3[index][static_cast<std::size_t>(i)]]);
        }
        for (int i = 0; i < 16; ++i) {
            EXPECT_DOUBLE_EQ(theta_e[i], theta[expected2[index][static_cast<std::size_t>(i)]]);
        }
        EXPECT_TRUE(u_map.coefficient_jacobian(e).isIdentity());
    }

    const auto snapshot = u_map.gather(0, a);
    a[3] += 7.0;
    EXPECT_DOUBLE_EQ(snapshot[0], 103.0);
    EXPECT_DOUBLE_EQ(u_map.gather(0, a)[0], 110.0);
    EXPECT_DOUBLE_EQ(up_map.gather(0, b)[0], b[3]);
}

TEST(VertexElementDofMap, ThreeGatheredFieldsReproduceAffineValuesAndPhysicalGradients) {
    const auto topology = two_hexes();
    const auto X = rest_positions();
    const VertexElementDofMap u_map(topology, VertexDofLayout<3>(12));
    const VertexElementDofMap up_map(topology, VertexDofLayout<3>(12));
    const VertexElementDofMap theta_map(topology, VertexDofLayout<2>(12));

    Eigen::Matrix3d A, P;
    A << 0.2, -0.1, 0.3, 0.4, 0.2, -0.3, -0.2, 0.5, 0.1;
    P << 0.1, 0.02, -0.03, -0.04, -0.05, 0.07, 0.06, -0.08, 0.03;
    const Eigen::Vector3d a_offset(-0.3, 0.4, 0.7), b_offset(0.6, -0.2, 0.3);
    Eigen::Matrix<double, 2, 3> M;
    M << 0.4, -0.1, 0.2, -0.3, 0.5, 0.1;
    const Eigen::Vector2d theta_offset(2.0, 3.0);
    Vec a(36), b(36), theta(24);
    for (int v = 0; v < 12; ++v) {
        const Eigen::Vector3d position = X.row(v).transpose();
        a.segment<3>(3 * v) = A * position + a_offset;
        b.segment<3>(3 * v) = P * position + b_offset;
        theta.segment<2>(2 * v) = M * position + theta_offset;
    }

    for (int e = 0; e < 2; ++e) {
        Reference::GeometryCoefficients X_e;
        const auto& vertices = topology.cell_vertices(e);
        for (int i = 0; i < 8; ++i) {
            X_e.row(i) = X.row(vertices[static_cast<std::size_t>(i)]);
        }
        const Reference geometry(X_e);
        const VectorField u(u_map.gather(e, a));
        const VectorField up(up_map.gather(e, b));
        const ParameterField parameters(theta_map.gather(e, theta));
        for (const Eigen::Vector3d xi : {Eigen::Vector3d(0.2, 0.3, 0.4), Eigen::Vector3d(0.76, 0.11, 0.64)}) {
            const auto point = geometry.evaluate(xi);
            const Eigen::Vector3d position =
                Eigen::Vector3d(1.2, -0.7, 2.1) + reference_jacobian() * (xi + Eigen::Vector3d(e, 0, 0));
            const auto u_sample = u.evaluate(point);
            const auto up_sample = up.evaluate(point);
            const auto theta_sample = parameters.evaluate(point);
            EXPECT_TRUE(u_sample.value.isApprox(A * position + a_offset, 1e-13));
            EXPECT_TRUE(u_sample.gradient.isApprox(A, 1e-13));
            EXPECT_TRUE(up_sample.value.isApprox(P * position + b_offset, 1e-13));
            EXPECT_TRUE(up_sample.gradient.isApprox(P, 1e-13));
            EXPECT_TRUE(theta_sample.value.isApprox(M * position + theta_offset, 1e-13));
            EXPECT_TRUE(theta_sample.gradient.isApprox(M, 1e-13));
        }
    }
}

TEST(VertexElementDofMap, SupportsTetConnectivityAndEigenVectorViews) {
    const litepgo::TetTopology topology(5, {{4, 1, 3, 0}});
    const VertexElementDofMap map(topology, VertexDofLayout<2>(5));
    const Vec padded = Vec::LinSpaced(12, -1.0, 10.0);
    const auto local = map.gather(0, padded.segment(1, 10));
    const Eigen::Matrix<double, 8, 1> expected{8, 9, 2, 3, 6, 7, 0, 1};
    EXPECT_EQ(local, expected);
}

TEST(VertexElementDofMap, CopiesTheLayoutAndRejectsMismatchedVertexCounts) {
    const auto topology = two_hexes();
    VertexDofLayout<3> layout(12);
    const VertexElementDofMap map(topology, layout);
    layout = VertexDofLayout<3>(13);
    EXPECT_EQ(map.num_field_dofs(), 36);
    EXPECT_THROW((VertexElementDofMap(topology, layout)), std::invalid_argument);
}

TEST(VertexElementDofMap, ChecksFieldSizeAndElementIndices) {
    const auto topology = two_hexes();
    const VertexElementDofMap map(topology, VertexDofLayout<3>(12));
    const Vec values = Vec::Zero(36);
    for (int e : {-1, 2}) {
        EXPECT_THROW(map.dof_indices(e), std::out_of_range);
        EXPECT_THROW(map.gather(e, values), std::out_of_range);
        EXPECT_THROW(map.coefficient_jacobian(e), std::out_of_range);
    }
    EXPECT_THROW(map.gather(0, Vec::Zero(35)), std::invalid_argument);
    EXPECT_THROW(map.gather(0, Vec::Zero(37)), std::invalid_argument);
}

TEST(VertexElementDofMap, AllowsEmptyMeshes) {
    const HexTopology topology(0, {});
    const VertexElementDofMap map(topology, VertexDofLayout<3>(0));
    EXPECT_EQ(map.num_field_dofs(), 0);
    EXPECT_THROW(map.dof_indices(0), std::out_of_range);
    EXPECT_THROW(map.gather(0, Vec(0)), std::out_of_range);
}
