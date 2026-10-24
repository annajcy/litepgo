#include "litepgo/fem/dof/element_dof_map.hpp"
#include <gtest/gtest.h>

namespace {
using namespace litepgo;

TEST(ElementDofMap, GathersIndependentBlocksWithColumnMajorIndices) {
    const HexTopology topology(8, {{7, 6, 5, 4, 3, 2, 1, 0}, {0, 1, 2, 3, 4, 5, 6, 7}});
    const ElementDofMap map(topology);
    ElementDofData<4> data(2);
    data.values() = Vec::LinSpaced(8, 1, 8);
    for (int e = 0; e < 2; ++e) {
        auto local = map.gather(e, data);
        const auto indices = map.dof_indices(e, data);
        for (int j = 0; j < 4; ++j) {
            EXPECT_EQ(indices[static_cast<std::size_t>(j)], 4 * e + j);
            EXPECT_EQ(local[j], data.values()[indices[static_cast<std::size_t>(j)]]);
        }
        local.setZero();
        EXPECT_FALSE(data.values().segment<4>(4 * e).isZero());
    }
    const auto snapshot = map.gather(1, data);
    data.values().setZero();
    EXPECT_FALSE(snapshot.isZero());
    EXPECT_TRUE(map.gather(1, data).isZero());
    ElementDofData<3> parameters(2);
    parameters.values().tail<3>() << 2, 3, 4;
    EXPECT_EQ(map.gather(1, parameters), Eigen::Vector3d(2, 3, 4));
    EXPECT_EQ(map.dof_indices(1, parameters), (std::array<int, 3>{3, 4, 5}));
}

TEST(ElementDofMap, ValidatesElementCountsAndIndicesOnEveryCall) {
    const TetTopology topology(4, {{0, 1, 2, 3}, {3, 2, 1, 0}});
    const ElementDofMap map(topology);
    const ElementDofData<2> data(2);
    for (int e : {-1, 2}) {
        EXPECT_THROW(map.gather(e, data), std::out_of_range);
        EXPECT_THROW(map.dof_indices(e, data), std::out_of_range);
    }
    for (int count : {0, 1, 3}) {
        const ElementDofData<2> wrong(count);
        EXPECT_THROW(map.gather(0, wrong), std::invalid_argument);
        EXPECT_THROW(map.dof_indices(0, wrong), std::invalid_argument);
    }
    const ElementDofMap empty(TriangleTopology(0, {}));
    const ElementDofData<9> empty_data(0);
    EXPECT_THROW(empty.gather(0, empty_data), std::out_of_range);
    EXPECT_THROW(empty.dof_indices(0, empty_data), std::out_of_range);
}

TEST(ElementDofMap, OwnsCountAfterTopologyDestruction) {
    const auto map = [] {
        const TriangleTopology topology(3, {{0, 1, 2}});
        return ElementDofMap(topology);
    }();
    ElementDofData<1> data(1);
    data.values()[0] = 7;
    EXPECT_EQ(map.gather(0, data)[0], 7);
    EXPECT_EQ(map.dof_indices(0, data), (std::array<int, 1>{0}));
}
} // namespace
