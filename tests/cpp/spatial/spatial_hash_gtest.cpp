#include "litepgo/spatial/spatial_hash.hpp"
#include <gtest/gtest.h>
#include <limits>
#include <tbb/global_control.h>
using namespace litepgo::spatial;
TEST(SpatialIndex, QueriesAndPairsMatchBruteForce) {
    std::vector<AABB> boxes;
    for (int i = 0; i < 150; ++i) {
        Eigen::Vector3d lo((i % 7) - 3.2, (i % 5) - 2.3, (i % 3) - 1.1);
        boxes.push_back({lo, lo + Eigen::Vector3d(1.8, 1.2, 0.9)});
    }
    for (int threads : {1, 4, 8}) {
        const tbb::global_control control(tbb::global_control::max_allowed_parallelism, threads);
        for (double h : {0.5, 2.0}) {
            SpatialHash index(boxes, h);
            std::vector<BoxPair> expected;
            for (std::size_t i = 0; i < boxes.size(); ++i) {
                std::vector<std::size_t> hits;
                for (std::size_t j = 0; j < boxes.size(); ++j) {
                    bool overlap = !(boxes[i].lo.array() > boxes[j].hi.array()).any() &&
                                   !(boxes[j].lo.array() > boxes[i].hi.array()).any();
                    if (overlap)
                        hits.push_back(j);
                    if (overlap && i < j && (i + j) % 2 == 0)
                        expected.push_back({i, j});
                }
                EXPECT_EQ(index.query(boxes[i]), hits);
            }
            auto actual = index.overlapping_pairs([](auto i, auto j) { return (i + j) % 2 == 0; });
            std::sort(actual.begin(), actual.end());
            EXPECT_EQ(actual, expected);
            EXPECT_EQ(index.query({boxes[0].hi, boxes[0].hi}).front(), 0);
        }
    }
}
TEST(SpatialIndex, OwnsBoxesAndValidatesQueries) {
    std::vector<AABB> boxes{{Eigen::Vector3d::Zero(), Eigen::Vector3d::Ones()}};
    SpatialHash index(boxes);
    boxes.clear();
    EXPECT_EQ(index.query({Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()}), std::vector<std::size_t>{0});
    EXPECT_THROW(index.query({Eigen::Vector3d::Ones(), Eigen::Vector3d::Zero()}), std::invalid_argument);
    EXPECT_THROW(SpatialHash(boxes, 0), std::invalid_argument);
    const double inf = std::numeric_limits<double>::infinity();
    EXPECT_THROW(index.query({Eigen::Vector3d::Constant(-inf), Eigen::Vector3d::Constant(inf)}), std::overflow_error);
    EXPECT_THROW(index.query({Eigen::Vector3d::Zero(), Eigen::Vector3d::Constant(10000)}), std::length_error);
    boxes.push_back({Eigen::Vector3d::Ones(), Eigen::Vector3d::Constant(2)});
    EXPECT_THROW(SpatialHash(boxes, 1e-20), std::overflow_error);
    EXPECT_THROW(SpatialHash(boxes, 0.001), std::length_error);
}
