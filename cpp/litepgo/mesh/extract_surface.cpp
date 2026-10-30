#include "litepgo/mesh/extract_surface.hpp"

#include <algorithm>
#include <array>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace litepgo::mesh {
namespace {
// Local faces wound outward for a positively oriented cell.
constexpr std::array<std::array<int, 3>, 4> tet_faces{{{0, 2, 1}, {0, 1, 3}, {0, 3, 2}, {1, 2, 3}}};
constexpr std::array<std::array<int, 4>, 6> hex_faces{
    {{0, 3, 2, 1}, {4, 5, 6, 7}, {0, 1, 5, 4}, {3, 7, 6, 2}, {0, 4, 7, 3}, {1, 2, 6, 5}}};

template <int K, std::size_t FaceSize, std::size_t NumFaces>
ExtractedSurface boundary(const MeshData<K>& mesh, const std::array<std::array<int, FaceSize>, NumFaces>& local_faces) {
    mesh.validate();
    using Face = std::array<int, FaceSize>;
    std::vector<Face> faces;
    faces.reserve(static_cast<std::size_t>(mesh.topology.num_cells()) * NumFaces);
    for (int e = 0; e < mesh.topology.num_cells(); ++e) {
        const auto& cell = mesh.topology.cell_vertices(e);
        for (const auto& local : local_faces) {
            Face face;
            for (std::size_t j = 0; j < FaceSize; ++j)
                face[j] = cell[static_cast<std::size_t>(local[j])];
            faces.push_back(face);
        }
    }

    // Group faces by vertex set; a boundary face is the only member of its group.
    std::vector<Face> keys = faces;
    for (auto& key : keys)
        std::sort(key.begin(), key.end());
    std::vector<std::size_t> order(faces.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return keys[a] < keys[b]; });
    std::vector<bool> on_boundary(faces.size(), false);
    for (std::size_t begin = 0; begin < order.size();) {
        std::size_t end = begin + 1;
        while (end < order.size() && keys[order[end]] == keys[order[begin]])
            ++end;
        if (end - begin > 2)
            throw std::invalid_argument("extract_surface: a face is shared by more than two cells");
        if (end - begin == 1)
            on_boundary[order[begin]] = true;
        begin = end;
    }

    std::vector<bool> used(static_cast<std::size_t>(mesh.topology.num_vertices()), false);
    for (std::size_t f = 0; f < faces.size(); ++f)
        if (on_boundary[f])
            for (const int v : faces[f])
                used[static_cast<std::size_t>(v)] = true;
    std::vector<int> new_index(used.size(), -1);
    std::vector<int> original;
    int num_vertices = 0;
    Vec positions(3 * std::count(used.begin(), used.end(), true));
    for (int v = 0; v < mesh.topology.num_vertices(); ++v)
        if (used[static_cast<std::size_t>(v)]) {
            positions.segment<3>(3 * Eigen::Index(num_vertices)) =
                mesh.positions.template segment<3>(3 * Eigen::Index(v));
            new_index[static_cast<std::size_t>(v)] = num_vertices++;
            original.push_back(v);
        }
    std::vector<TriangleTopology::Cell> triangles;
    for (std::size_t f = 0; f < faces.size(); ++f) {
        if (!on_boundary[f])
            continue;
        Face face = faces[f];
        for (auto& v : face)
            v = new_index[static_cast<std::size_t>(v)];
        // Fan triangulation; a quad splits along its 0-2 diagonal.
        for (std::size_t j = 1; j + 1 < FaceSize; ++j)
            triangles.push_back({face[0], face[j], face[j + 1]});
    }
    TriangleMesh surface{std::move(positions), TriangleTopology(num_vertices, std::move(triangles))};
    surface.validate();
    return {std::move(surface), std::move(original)};
}
} // namespace

TriangleMesh extract_surface(const TetMesh& mesh) {
    return boundary(mesh, tet_faces).mesh;
}

TriangleMesh extract_surface(const HexMesh& mesh) {
    return boundary(mesh, hex_faces).mesh;
}

TriangleMesh extract_surface(const VolumeMesh& mesh) {
    return std::visit([](const auto& data) { return extract_surface(data); }, mesh);
}

ExtractedSurface extract_surface_with_vertices(const TetMesh& mesh) {
    return boundary(mesh, tet_faces);
}
ExtractedSurface extract_surface_with_vertices(const HexMesh& mesh) {
    return boundary(mesh, hex_faces);
}
ExtractedSurface extract_surface_with_vertices(const VolumeMesh& mesh) {
    return std::visit([](const auto& data) { return extract_surface_with_vertices(data); }, mesh);
}
} // namespace litepgo::mesh
