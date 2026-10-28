#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace litepgo {

// Fixed-arity connectivity. Cell vertices must be ordered to match the chosen
// basis; the vertex count alone does not determine the local numbering.
template <int VerticesPerCell>
    requires(VerticesPerCell > 0)
class MeshTopology final {
  public:
    static constexpr int vertices_per_cell = VerticesPerCell;
    using Cell = std::array<int, vertices_per_cell>;

    // Owns cells and preserves their order. Each cell must contain distinct
    // vertex indices in [0, num_vertices). Empty meshes and unused vertices are allowed.
    MeshTopology(int num_vertices, std::vector<Cell> cells) : num_vertices_(num_vertices), cells_(std::move(cells)) {
        if (num_vertices_ < 0) {
            throw std::invalid_argument("MeshTopology requires a non-negative vertex count");
        }
        if (cells_.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::invalid_argument("MeshTopology cell count exceeds the int index range");
        }
        for (std::size_t element = 0; element < cells_.size(); ++element) {
            auto sorted = cells_[element];
            std::sort(sorted.begin(), sorted.end());
            if (sorted.front() < 0 || sorted.back() >= num_vertices_) {
                throw std::invalid_argument("MeshTopology cell " + std::to_string(element) +
                                            " has vertex indices outside [0, num_vertices)");
            }
            if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
                throw std::invalid_argument("MeshTopology cell " + std::to_string(element) +
                                            " must contain distinct vertices");
            }
        }
    }

    int num_vertices() const noexcept {
        return num_vertices_;
    }
    int num_cells() const noexcept {
        return static_cast<int>(cells_.size());
    }

    // Throws std::out_of_range for an invalid cell index.
    const Cell& cell_vertices(int cell) const {
        if (cell < 0 || cell >= num_cells()) {
            throw std::out_of_range("MeshTopology::cell_vertices: cell " + std::to_string(cell) + " is out of range");
        }
        return cells_[static_cast<std::size_t>(cell)];
    }

  private:
    int num_vertices_;
    std::vector<Cell> cells_;
};

using TriangleTopology = MeshTopology<3>;
using QuadTopology = MeshTopology<4>;
using TetTopology = MeshTopology<4>;
using HexTopology = MeshTopology<8>;

} // namespace litepgo
