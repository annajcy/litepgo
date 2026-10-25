#include "litepgo/dof/element_dof_map.hpp"
#include <gtest/gtest.h>
#include <limits>

namespace {
using namespace litepgo;

TEST(ElementDofMap, GathersIndependentBlocksWithColumnMajorIndices) {
    const HexTopology topology(8, {{7, 6, 5, 4, 3, 2, 1, 0}, {0, 1, 2, 3, 4, 5, 6, 7}});
    const ElementDofMap map(topology, 4);
    Vec data = Vec::Zero(8);
    data = Vec::LinSpaced(8, 1, 8);
    for (int e = 0; e < 2; ++e) {
        auto local = map.gather(e, data);
        const auto indices = map.dof_indices(e);
        for (int j = 0; j < 4; ++j) {
            EXPECT_EQ(indices[static_cast<std::size_t>(j)], 4 * e + j);
            EXPECT_EQ(local[j], data[indices[static_cast<std::size_t>(j)]]);
        }
        local.setZero();
        EXPECT_FALSE(data.segment<4>(4 * e).isZero());
    }
    const auto snapshot = map.gather(1, data);
    data.setZero();
    EXPECT_FALSE(snapshot.isZero());
    EXPECT_TRUE(map.gather(1, data).isZero());
    Vec parameters = Vec::Zero(6);
    parameters.tail<3>() << 2, 3, 4;
    EXPECT_EQ(ElementDofMap(topology, 3).gather(1, parameters), Eigen::Vector3d(2, 3, 4));
    EXPECT_EQ(ElementDofMap(topology, 3).dof_indices(1), (std::vector<int>{3, 4, 5}));
}

TEST(ElementDofMap, ValidatesElementCountsAndIndicesOnEveryCall) {
    const TetTopology topology(4, {{0, 1, 2, 3}, {3, 2, 1, 0}});
    const ElementDofMap map(topology, 2);
    const Vec data = Vec::Zero(4);
    for (int e : {-1, 2}) {
        EXPECT_THROW(map.gather(e, data), std::out_of_range);
        EXPECT_THROW(map.dof_indices(e), std::out_of_range);
    }
    for (int count : {0, 1, 3}) {
        const Vec wrong = Vec::Zero(2 * count);
        EXPECT_THROW(map.gather(0, wrong), std::invalid_argument);
    }
    const ElementDofMap empty(TriangleTopology(0, {}), 9);
    const Vec empty_data = Vec::Zero(0);
    EXPECT_THROW(empty.gather(0, empty_data), std::out_of_range);
    EXPECT_THROW(empty.dof_indices(0), std::out_of_range);
}

TEST(ElementDofMap, OwnsCountAfterTopologyDestruction) {
    const auto map = [] {
        const TriangleTopology topology(3, {{0, 1, 2}});
        return ElementDofMap(topology, 1);
    }();
    Vec data = Vec::Zero(1);
    data[0] = 7;
    EXPECT_EQ(map.gather(0, data)[0], 7);
    EXPECT_EQ(map.dof_indices(0), (std::vector<int>{0}));
}
} // namespace

TEST(ElementDofMap, OwnsRuntimeLayoutAndValidatesWidthAndLength) {
    const litepgo::TetTopology topology(4, {{0, 1, 2, 3}, {3, 2, 1, 0}});
    const litepgo::ElementDofMap map(topology, 5);
    EXPECT_EQ(map.num_dofs(), 10);
    EXPECT_EQ(map.dofs_per_element(), 5);
    EXPECT_EQ(map.num_elements(), 2);
    const litepgo::Vec values = litepgo::Vec::LinSpaced(12, 0, 11);
    EXPECT_EQ(map.gather(1, values.segment(1, 10)), values.segment(6, 5));
    for (int width : {0, -1, std::numeric_limits<int>::max()})
        EXPECT_THROW((litepgo::ElementDofMap(topology, width)), std::invalid_argument);
    for (int size : {9, 11})
        EXPECT_THROW(map.gather(0, litepgo::Vec::Zero(size)), std::invalid_argument);
}
