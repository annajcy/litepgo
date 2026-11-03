// Point, region and filtered pair queries; mirrors spatial_query.cpp.
// Build: cmake --build build/release --target spatial_query
// Run: build/release/examples/spatial/spatial_query

#include "litepgo/spatial/spatial_hash.hpp"
#include <iostream>
#include <stdexcept>
int main() {
    using namespace litepgo::spatial;
    std::vector<AABB> boxes{{{.2, .2, 0}, {1.8, .8, 1}}, {{1.7, .2, 0}, {1.9, .8, 1}}};
    SpatialHash index(boxes, 1);
    auto hits = index.query({{.9, .3, .2}, {1.2, .6, .8}});
    auto point = index.query({{1.75, .5, .5}, {1.75, .5, .5}});
    auto pairs = index.overlapping_pairs([](auto i, auto j) { return i != j; });
    std::sort(point.begin(), point.end());
    if (hits != std::vector<std::size_t>{0} || point != std::vector<std::size_t>{0, 1} ||
        pairs != std::vector<BoxPair>{{0, 1}})
        throw std::runtime_error("Unexpected spatial query result");
    std::cout << "region_hits=" << hits.size() << ", point_hits=" << point.size() << ", pairs=" << pairs.size()
              << std::endl;
}
