#include "litepgo/fem/field_interpolation.hpp"
#include "litepgo/fem/hex_q1_basis.hpp"
#include "litepgo/fem/vertex_element_dof_map.hpp"
#include "litepgo/fem/volume_reference_geometry.hpp"

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
using litepgo::VertexDofData;
using litepgo::VertexElementDofMap;

template <int N>
concept CanFormMap = requires { typename VertexElementDofMap<N>; };

template <int D>
concept CanGather = requires(const VertexElementDofMap<8>& map, const VertexDofData<D>& data) {
    map.gather(0, data);
    map.dof_indices(0, data);
};

static_assert(CanFormMap<8>);
static_assert(!CanFormMap<0>);
static_assert(!CanFormMap<-1>);
static_assert(CanGather<3>);
static_assert(CanGather<2>);
static_assert(!CanGather<std::numeric_limits<int>::max()>);
static_assert(std::same_as<decltype(VertexElementDofMap(std::declval<const HexTopology&>())), VertexElementDofMap<8>>);
static_assert(std::same_as<decltype(std::declval<const VertexElementDofMap<8>&>().gather(
                               0, std::declval<const VertexDofData<3>&>())),
                           Eigen::Matrix<double, 8, 3>>);
static_assert(std::same_as<decltype(std::declval<const VertexElementDofMap<8>&>().gather(
                               0, std::declval<const VertexDofData<2>&>())),
                           Eigen::Matrix<double, 8, 2>>);
static_assert(std::same_as<decltype(std::declval<const VertexElementDofMap<8>&>().gather(
                               0, std::declval<const VertexDofData<3>&>())),
                           Eigen::Matrix<double, 8, 3>>);
static_assert(std::same_as<decltype(std::declval<const VertexElementDofMap<8>&>().gather(
                               0, std::declval<const VertexDofData<2>&>())),
                           Eigen::Matrix<double, 8, 2>>);
static_assert(std::same_as<decltype(std::declval<const VertexElementDofMap<8>&>().gather(
                               0, std::declval<const VertexDofData<1>&>())),
                           Eigen::Matrix<double, 8, 1>>);
static_assert(VertexElementDofMap<8>::num_element_dofs<3> == 24);
static_assert(VertexElementDofMap<8>::num_element_dofs<2> == 16);
static_assert(!std::is_constructible_v<VertexElementDofMap<8>, HexTopology&&>);
static_assert(!std::is_constructible_v<VertexElementDofMap<8>, const HexTopology&&>);
static_assert(!std::is_convertible_v<const HexTopology&, VertexElementDofMap<8>>);

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
    const VertexElementDofMap map(topology);

    VertexDofData<3> a(12), b(12);
    VertexDofData<2> theta(12);
    for (int i = 0; i < 36; ++i) {
        a.values()[i] = 100.0 + i;
        b.values()[i] = -1000.0 - 2.0 * i;
    }
    for (int i = 0; i < 24; ++i) {
        theta.values()[i] = 2.0 + 0.1 * i;
    }

    const std::array<std::array<int, 24>, 2> expected3{{
        {3, 4, 5, 9, 10, 11, 27, 28, 29, 12, 13, 14, 21, 22, 23, 18, 19, 20, 0, 1, 2, 30, 31, 32},
        {9, 10, 11, 24, 25, 26, 15, 16, 17, 27, 28, 29, 18, 19, 20, 6, 7, 8, 33, 34, 35, 0, 1, 2},
    }};
    const std::array<std::array<int, 16>, 2> expected2{{
        {2, 3, 6, 7, 18, 19, 8, 9, 14, 15, 12, 13, 0, 1, 20, 21},
        {6, 7, 16, 17, 10, 11, 18, 19, 12, 13, 4, 5, 22, 23, 0, 1},
    }};
    for (int e = 0; e < 2; ++e) {
        const auto index = static_cast<std::size_t>(e);
        EXPECT_EQ(map.dof_indices(e, a), expected3[index]);
        EXPECT_EQ(map.dof_indices(e, b), expected3[index]);
        EXPECT_EQ(map.dof_indices(e, theta), expected2[index]);
        const auto a_e = map.gather(e, a);
        const auto b_e = map.gather(e, b);
        const auto theta_e = map.gather(e, theta);
        for (int i = 0; i < 8; ++i) {
            for (int c = 0; c < 3; ++c) {
                const int global = expected3[index][static_cast<std::size_t>(3 * i + c)];
                EXPECT_DOUBLE_EQ(a_e(i, c), a.values()[global]);
                EXPECT_DOUBLE_EQ(b_e(i, c), b.values()[global]);
            }
            for (int c = 0; c < 2; ++c) {
                const int global = expected2[index][static_cast<std::size_t>(2 * i + c)];
                EXPECT_DOUBLE_EQ(theta_e(i, c), theta.values()[global]);
            }
        }
    }

    const auto snapshot = map.gather(0, a);
    a.values()[3] += 7.0;
    EXPECT_DOUBLE_EQ(snapshot(0, 0), 103.0);
    EXPECT_DOUBLE_EQ(map.gather(0, a)(0, 0), 110.0);
    EXPECT_DOUBLE_EQ(map.gather(0, b)(0, 0), b.values()[3]);
    auto edited = map.gather(0, a);
    edited(0, 0) = -1.0;
    EXPECT_DOUBLE_EQ(a(1, 0), 110.0);
}

TEST(VertexElementDofMap, ThreeGatheredFieldsReproduceAffineValuesAndPhysicalGradients) {
    const auto topology = two_hexes();
    const auto X = rest_positions();
    const VertexElementDofMap map(topology);

    Eigen::Matrix3d A, P;
    A << 0.2, -0.1, 0.3, 0.4, 0.2, -0.3, -0.2, 0.5, 0.1;
    P << 0.1, 0.02, -0.03, -0.04, -0.05, 0.07, 0.06, -0.08, 0.03;
    const Eigen::Vector3d a_offset(-0.3, 0.4, 0.7), b_offset(0.6, -0.2, 0.3);
    Eigen::Matrix<double, 2, 3> M;
    M << 0.4, -0.1, 0.2, -0.3, 0.5, 0.1;
    const Eigen::Vector2d theta_offset(2.0, 3.0);
    VertexDofData<3> a(12), b(12);
    VertexDofData<2> theta(12);
    for (int v = 0; v < 12; ++v) {
        const Eigen::Vector3d position = X.row(v).transpose();
        a.values().segment<3>(3 * v) = A * position + a_offset;
        b.values().segment<3>(3 * v) = P * position + b_offset;
        theta.values().segment<2>(2 * v) = M * position + theta_offset;
    }

    for (int e = 0; e < 2; ++e) {
        Eigen::Matrix<double, 8, 3> X_e;
        const auto& vertices = topology.cell_vertices(e);
        for (int i = 0; i < 8; ++i) {
            X_e.row(i) = X.row(vertices[static_cast<std::size_t>(i)]);
        }
        const litepgo::VolumeReferenceGeometry<litepgo::HexQ1Basis> geometry(X_e);
        const litepgo::FieldInterpolation u(map.gather(e, a));
        const litepgo::FieldInterpolation up(map.gather(e, b));
        const litepgo::FieldInterpolation parameters(map.gather(e, theta));
        for (const Eigen::Vector3d xi : {Eigen::Vector3d(0.2, 0.3, 0.4), Eigen::Vector3d(0.76, 0.11, 0.64)}) {
            const auto point = geometry.evaluate(xi);
            const Eigen::Vector3d position =
                Eigen::Vector3d(1.2, -0.7, 2.1) + reference_jacobian() * (xi + Eigen::Vector3d(e, 0, 0));
            const auto u_sample_value = u.value(point);
            const auto u_sample_gradient = u.gradient(point);
            const auto up_sample_value = up.value(point);
            const auto up_sample_gradient = up.gradient(point);
            const auto theta_sample_value = parameters.value(point);
            const auto theta_sample_gradient = parameters.gradient(point);
            EXPECT_TRUE(u_sample_value.isApprox(A * position + a_offset, 1e-13));
            EXPECT_TRUE(u_sample_gradient.isApprox(A, 1e-13));
            EXPECT_TRUE(up_sample_value.isApprox(P * position + b_offset, 1e-13));
            EXPECT_TRUE(up_sample_gradient.isApprox(P, 1e-13));
            EXPECT_TRUE(theta_sample_value.isApprox(M * position + theta_offset, 1e-13));
            EXPECT_TRUE(theta_sample_gradient.isApprox(M, 1e-13));
        }
    }
}

TEST(VertexElementDofMap, SupportsTetConnectivityAndEigenVectorViews) {
    const litepgo::TetTopology topology(5, {{4, 1, 3, 0}});
    const VertexElementDofMap map(topology);
    const Vec padded = Vec::LinSpaced(12, -1.0, 10.0);
    const VertexDofData<2> data(padded.segment(1, 10));
    const auto local = map.gather(0, data);
    Eigen::Matrix<double, 4, 2> expected;
    expected << 8, 9, 2, 3, 6, 7, 0, 1;
    EXPECT_EQ(local, expected);
}

TEST(VertexElementDofMap, GathersScalarVertexDataAsAColumn) {
    const litepgo::TetTopology topology(5, {{4, 1, 3, 0}});
    const VertexElementDofMap map(topology);
    const VertexDofData<1> data(Vec::LinSpaced(5, 10.0, 14.0));
    const Eigen::Vector4d expected(14, 11, 13, 10);
    EXPECT_EQ(map.gather(0, data), expected);
}

TEST(VertexElementDofMap, RejectsMismatchedDataOnEveryCall) {
    const auto topology = two_hexes();
    const VertexElementDofMap map(topology);
    VertexDofData<3> data(12);
    EXPECT_TRUE(map.gather(0, data).isZero());
    for (int count : {11, 13}) {
        data = VertexDofData<3>(count);
        EXPECT_THROW(map.gather(0, data), std::invalid_argument);
        EXPECT_THROW(map.dof_indices(0, data), std::invalid_argument);
        const VertexDofData<2> parameters(count);
        EXPECT_THROW(map.gather(0, parameters), std::invalid_argument);
        EXPECT_THROW(map.dof_indices(0, parameters), std::invalid_argument);
    }
    data = VertexDofData<3>(12);
    EXPECT_TRUE(map.gather(0, data).isZero());
}

TEST(VertexElementDofMap, ChecksElementIndices) {
    const auto topology = two_hexes();
    const VertexElementDofMap map(topology);
    const VertexDofData<3> data(12);
    for (int e : {-1, 2}) {
        EXPECT_THROW(map.dof_indices(e, data), std::out_of_range);
        EXPECT_THROW(map.gather(e, data), std::out_of_range);
    }
}

TEST(VertexElementDofMap, AllowsEmptyMeshes) {
    const HexTopology topology(0, {});
    const VertexElementDofMap map(topology);
    const VertexDofData<3> data(0);
    EXPECT_THROW(map.dof_indices(0, data), std::out_of_range);
    EXPECT_THROW(map.gather(0, data), std::out_of_range);
}
