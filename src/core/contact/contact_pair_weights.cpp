#include "litepgo/contact/contact_pair_weights.hpp"
#include <algorithm>

namespace litepgo::contact {

std::vector<std::array<int, 2>> contact_edges(const mesh::TriangleTopology& topology) {
    std::vector<std::array<int, 2>> edges;
    edges.reserve(3 * static_cast<std::size_t>(topology.num_cells()));
    for (int f = 0; f < topology.num_cells(); ++f) {
        const auto& face = topology.cell_vertices(f);
        for (int j = 0; j < 3; ++j)
            edges.push_back({std::min(face[j], face[(j + 1) % 3]), std::max(face[j], face[(j + 1) % 3])});
    }
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
    return edges;
}

} // namespace litepgo::contact
