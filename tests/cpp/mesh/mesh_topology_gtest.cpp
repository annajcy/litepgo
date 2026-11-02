#include "litepgo/mesh/mesh_topology.hpp"

#include <gtest/gtest.h>

#include <concepts>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace litepgo::mesh::test {

namespace {

static_assert(std::same_as<TriangleTopology, MeshTopology<3>>);
static_assert(std::same_as<TetTopology, MeshTopology<4>>);
static_assert(std::same_as<HexTopology, MeshTopology<8>>);
static_assert(TriangleTopology::vertices_per_cell == 3);
static_assert(TetTopology::vertices_per_cell == 4);
static_assert(HexTopology::vertices_per_cell == 8);

template <int N>
concept CanFormTopology = requires { typename MeshTopology<N>; };

static_assert(CanFormTopology<1>);
static_assert(!CanFormTopology<0>);
static_assert(!CanFormTopology<-1>);

template <class Topology> std::vector<typename Topology::Cell> two_cells() {
    constexpr int n = Topology::vertices_per_cell;
    typename Topology::Cell first;
    typename Topology::Cell second;
    for (int i = 0; i < n; ++i) {
        first[static_cast<std::size_t>(i)] = n - 1 - i;
        second[static_cast<std::size_t>(i)] = 2 * n - 2 - i;
    }
    return {first, second};
}

template <class Topology> class MeshTopologyTest : public ::testing::Test {};

using TopologyTypes = ::testing::Types<TriangleTopology, TetTopology, HexTopology>;
TYPED_TEST_SUITE(MeshTopologyTest, TopologyTypes);

TYPED_TEST(MeshTopologyTest, PreservesCellAndLocalVertexOrdering) {
    constexpr int num_vertices = 2 * TypeParam::vertices_per_cell - 1;
    const auto cells = two_cells<TypeParam>();
    const TypeParam topology(num_vertices, cells);
    EXPECT_EQ(topology.num_vertices(), num_vertices);
    EXPECT_EQ(topology.num_cells(), 2);
    EXPECT_EQ(topology.cell_vertices(0), cells[0]);
    EXPECT_EQ(topology.cell_vertices(1), cells[1]);
}

TYPED_TEST(MeshTopologyTest, OwnsItsConnectivity) {
    constexpr int num_vertices = 2 * TypeParam::vertices_per_cell - 1;
    auto cells = two_cells<TypeParam>();
    const auto expected = cells[0];
    const TypeParam topology(num_vertices, cells);
    cells[0][0] = num_vertices - 1;
    cells.clear();
    EXPECT_EQ(topology.num_cells(), 2);
    EXPECT_EQ(topology.cell_vertices(0), expected);
}

TYPED_TEST(MeshTopologyTest, AllowsEmptyMeshesAndUnusedVertices) {
    const TypeParam empty(0, {});
    EXPECT_EQ(empty.num_vertices(), 0);
    EXPECT_EQ(empty.num_cells(), 0);

    const TypeParam vertices_only(3, {});
    EXPECT_EQ(vertices_only.num_vertices(), 3);
    EXPECT_EQ(vertices_only.num_cells(), 0);

    const auto cells = two_cells<TypeParam>();
    const TypeParam unused_vertex(TypeParam::vertices_per_cell + 1, {cells[0]});
    EXPECT_EQ(unused_vertex.num_vertices(), TypeParam::vertices_per_cell + 1);
    EXPECT_EQ(unused_vertex.num_cells(), 1);
}

TYPED_TEST(MeshTopologyTest, RejectsInvalidConnectivity) {
    constexpr int n = TypeParam::vertices_per_cell;
    constexpr int num_vertices = 2 * n - 1;
    const auto cells = two_cells<TypeParam>();
    EXPECT_THROW(TypeParam(-1, {}), std::invalid_argument);
    EXPECT_THROW(TypeParam(0, {cells[0]}), std::invalid_argument);
    EXPECT_THROW(TypeParam(n - 1, {cells[0]}), std::invalid_argument);

    auto negative = cells;
    negative[1][0] = -1;
    EXPECT_THROW(TypeParam(num_vertices, negative), std::invalid_argument);

    auto out_of_range = cells;
    out_of_range[1][0] = num_vertices;
    EXPECT_THROW(TypeParam(num_vertices, out_of_range), std::invalid_argument);

    auto repeated = cells;
    repeated[1][n - 1] = repeated[1][0];
    EXPECT_THROW(TypeParam(num_vertices, repeated), std::invalid_argument);
}

TYPED_TEST(MeshTopologyTest, ChecksCellIndices) {
    const TypeParam topology(2 * TypeParam::vertices_per_cell - 1, two_cells<TypeParam>());
    EXPECT_THROW(topology.cell_vertices(-1), std::out_of_range);
    EXPECT_THROW(topology.cell_vertices(2), std::out_of_range);

    const TypeParam empty(0, {});
    EXPECT_THROW(empty.cell_vertices(0), std::out_of_range);
}

TEST(MeshTopology, SupportsOtherPositiveCellSizes) {
    const MeshTopology<1> points(2, {{1}, {0}});
    EXPECT_EQ(points.num_cells(), 2);
    EXPECT_EQ(points.cell_vertices(0)[0], 1);

    const MeshTopology<5>::Cell cell = {4, 1, 3, 0, 2};
    const MeshTopology<5> topology(5, {cell});
    EXPECT_EQ(topology.cell_vertices(0), cell);
}

} // namespace

} // namespace litepgo::mesh::test
