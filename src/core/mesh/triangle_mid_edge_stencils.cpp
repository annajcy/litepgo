#include "litepgo/mesh/triangle_mid_edge_stencils.hpp"

#include <algorithm>
#include <map>
#include <stdexcept>
#include <utility>

namespace litepgo::mesh {

TriangleMidEdgeStencils make_triangle_mid_edge_stencils(const TriangleTopology& topology) {
    TriangleMidEdgeStencils result;
    result.vertices.reserve(static_cast<std::size_t>(topology.num_cells()));
    result.opposite.resize(static_cast<std::size_t>(topology.num_cells()), {-1, -1, -1});
    struct Edge {
        int face;
        int opposite_vertex;
        bool paired = false;
    };
    std::map<std::pair<int, int>, Edge> edges;
    for (int f = 0; f < topology.num_cells(); ++f) {
        const auto& face = topology.cell_vertices(f);
        result.vertices.emplace_back(face.begin(), face.end());
        for (int k = 0; k < 3; ++k) {
            const int from = face[static_cast<std::size_t>((k + 1) % 3)];
            const int to = face[static_cast<std::size_t>((k + 2) % 3)];
            const auto [it, inserted] = edges.try_emplace(std::minmax(from, to), Edge{f, k});
            if (inserted)
                continue;
            auto& edge = it->second;
            if (edge.paired)
                throw std::invalid_argument("Triangle mid-edge stencils: edge has more than two incident faces");
            const auto& neighbor = topology.cell_vertices(edge.face);
            const int neighbor_opposite = neighbor[static_cast<std::size_t>(edge.opposite_vertex)];
            if (neighbor_opposite == face[static_cast<std::size_t>(k)])
                throw std::invalid_argument("Triangle mid-edge stencils: duplicate faces");
            if (neighbor[static_cast<std::size_t>((edge.opposite_vertex + 1) % 3)] != to)
                throw std::invalid_argument("Triangle mid-edge stencils: inconsistent face orientation");
            result.opposite[static_cast<std::size_t>(f)][static_cast<std::size_t>(k)] = neighbor_opposite;
            result.opposite[static_cast<std::size_t>(edge.face)][static_cast<std::size_t>(edge.opposite_vertex)] =
                face[static_cast<std::size_t>(k)];
            edge.paired = true;
        }
    }
    // Convert global opposite vertices into deduplicated local rows, in edge order.
    for (std::size_t f = 0; f < result.vertices.size(); ++f) {
        auto& vertices = result.vertices[f];
        for (int& opposite : result.opposite[f]) {
            if (opposite < 0)
                continue;
            const auto found = std::find(vertices.begin(), vertices.end(), opposite);
            const int local = static_cast<int>(found - vertices.begin());
            if (found == vertices.end())
                vertices.push_back(opposite);
            opposite = local;
        }
    }
    return result;
}

} // namespace litepgo::mesh
