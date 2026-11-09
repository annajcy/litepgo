#include "litepgo/fem/embedding/sample_boundary.hpp"
#include "litepgo/mesh/detail/hex_faces.hpp"
#include <Eigen/Geometry>
#include <cstdint>
#include <map>
#include <numeric>

namespace litepgo::fem {
template <>
SampledBoundary sample_boundary<HexQ1Basis>(const mesh::HexTopology& topology,
                                            const std::vector<std::shared_ptr<energy::Response>>& fields, int n) {
    if (n < 1 || fields.size() != static_cast<std::size_t>(topology.num_cells()))
        throw std::invalid_argument("sample_boundary: positive subdivisions and one X field per element required");
    using Face = std::array<int, 4>;
    struct BoundaryFace {
        int element;
        Face local, vertices;
    };
    std::map<Face, int> counts;
    std::vector<BoundaryFace> faces;
    std::map<int, Eigen::Vector3d> corners;
    for (int e = 0; e < topology.num_cells(); ++e) {
        if (!fields[e] || !dynamic_cast<InterpolatedResponse<HexQ1Basis>*>(fields[e].get()) ||
            fields[e]->num_outputs() != 3)
            throw std::invalid_argument("sample_boundary: X fields need three components");
        const auto& cell = topology.cell_vertices(e);
        for (int v = 0; v < 8; ++v) {
            Eigen::Vector3d p = fields[e]->evaluate(HexQ1Basis::coefficient_info(v).point)->value();
            auto [it, inserted] = corners.emplace(cell[v], p);
            if (!inserted && !(it->second.array() == p.array()).all())
                throw std::invalid_argument("sample_boundary: shared vertex has inconsistent rest positions");
        }
        for (const auto& local : mesh::detail::hex_faces) {
            Face vertices;
            for (int i = 0; i < 4; ++i)
                vertices[i] = cell[local[i]];
            Face key = vertices;
            std::sort(key.begin(), key.end());
            if (++counts[key] > 2)
                throw std::invalid_argument("sample_boundary: face shared by more than two elements");
            faces.push_back({e, local, vertices});
        }
    }
    std::erase_if(faces, [&](const auto& f) {
        auto key = f.vertices;
        std::sort(key.begin(), key.end());
        return counts.at(key) != 1;
    });
    const auto n64 = static_cast<std::uint64_t>(n);
    const auto per_face = 2 * n64 * n64;
    if (per_face > std::numeric_limits<int>::max() ||
        faces.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) / per_face)
        throw std::length_error("sample_boundary: triangle count exceeds int range");
    // Integer bilinear weights indexed by global corner IDs identify shared edges exactly.
    using Key = std::vector<std::pair<int, std::int64_t>>;
    struct Sample {
        int element;
        Eigen::Vector3d xi, position;
        Key key;
    };
    std::map<Key, int> ids;
    std::vector<Sample> samples;
    std::vector<mesh::TriangleTopology::Cell> triangles;
    for (const auto& f : faces) {
        std::vector<int> grid(static_cast<std::size_t>(n + 1) * (n + 1));
        const Eigen::Vector3d a = HexQ1Basis::coefficient_info(f.local[0]).point;
        const Eigen::Vector3d b = HexQ1Basis::coefficient_info(f.local[1]).point;
        const Eigen::Vector3d d = HexQ1Basis::coefficient_info(f.local[3]).point;
        for (int j = 0; j <= n; ++j)
            for (int i = 0; i <= n; ++i) {
                const std::int64_t u = i, v = j, m = n;
                const std::array<std::int64_t, 4> weights{(m - u) * (m - v), u * (m - v), u * v, (m - u) * v};
                Key key;
                for (int k = 0; k < 4; ++k)
                    if (weights[k])
                        key.emplace_back(f.vertices[k], weights[k]);
                std::sort(key.begin(), key.end());
                auto it = ids.find(key);
                if (it == ids.end()) {
                    if (samples.size() >= static_cast<std::size_t>(std::numeric_limits<int>::max() / 3))
                        throw std::length_error("sample_boundary: vertex count exceeds coordinate index range");
                    const int id = static_cast<int>(samples.size());
                    Eigen::Vector3d xi = a + (double(i) / n) * (b - a) + (double(j) / n) * (d - a);
                    Eigen::Vector3d position = fields[f.element]->evaluate(xi)->value();
                    samples.push_back({f.element, xi, position, key});
                    it = ids.emplace(std::move(key), id).first;
                }
                grid[static_cast<std::size_t>(j) * (n + 1) + i] = it->second;
            }
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                const auto k = static_cast<std::size_t>(j) * (n + 1) + i;
                triangles.push_back({grid[k], grid[k + 1], grid[k + n + 2]});
                triangles.push_back({grid[k], grid[k + n + 2], grid[k + n + 1]});
            }
    }
    std::vector<int> order(samples.size()), remap(samples.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int i, int j) {
        const auto& a = samples[i].key;
        const auto& b = samples[j].key;
        if ((a.size() == 1) != (b.size() == 1))
            return a.size() == 1;
        return a < b;
    });
    const int size = static_cast<int>(samples.size());
    Vec positions(3 * size);
    PointEmbedding embedding{Eigen::VectorXi(size), decltype(PointEmbedding::xi)(size, 3)};
    for (int i = 0; i < size; ++i) {
        const auto& sample = samples[order[i]];
        remap[order[i]] = i;
        positions.segment<3>(3 * i) = sample.position;
        embedding.element_ids[i] = sample.element;
        embedding.xi.row(i) = sample.xi.transpose();
    }
    for (auto& triangle : triangles)
        for (int& v : triangle)
            v = remap[v];
    return {{std::move(positions), mesh::TriangleTopology(size, std::move(triangles))}, std::move(embedding)};
}
} // namespace litepgo::fem
