#include "litepgo/io/obj.hpp"

#include <tiny_obj_loader.h>

#include <fstream>
#include <iomanip>
#include <limits>
#include <type_traits>

static_assert(std::is_same_v<tinyobj::real_t, double>);

namespace litepgo::io {
TriangleMesh read_obj(const std::filesystem::path& path) {
    tinyobj::ObjReader reader;
    tinyobj::ObjReaderConfig config;
    config.triangulate = true;
    if (!reader.ParseFromFile(path.string(), config))
        throw std::runtime_error("Cannot read OBJ " + path.string() + ": " + reader.Error());
    const auto& vertices = reader.GetAttrib().vertices;
    if (vertices.size() / 3 > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::runtime_error("OBJ vertex count exceeds int range");
    Vec positions(static_cast<Eigen::Index>(vertices.size()));
    std::copy(vertices.begin(), vertices.end(), positions.data());
    std::vector<TriangleTopology::Cell> faces;
    for (const auto& shape : reader.GetShapes()) {
        std::size_t offset = 0;
        for (auto width : shape.mesh.num_face_vertices) {
            if (width != 3)
                throw std::runtime_error("OBJ triangulation failed: " + path.string());
            faces.push_back({shape.mesh.indices.at(offset).vertex_index, shape.mesh.indices.at(offset + 1).vertex_index,
                             shape.mesh.indices.at(offset + 2).vertex_index});
            offset += 3;
        }
    }
    TriangleMesh mesh{std::move(positions), TriangleTopology(static_cast<int>(vertices.size() / 3), std::move(faces))};
    mesh.validate();
    return mesh;
}

void write_obj(const std::filesystem::path& path, const TriangleMesh& mesh) {
    mesh.validate();
    std::ofstream out(path);
    if (!out)
        throw std::runtime_error("Cannot write OBJ: " + path.string());
    out << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (int v = 0; v < mesh.topology.num_vertices(); ++v)
        out << "v " << mesh.positions[3 * Eigen::Index(v)] << ' ' << mesh.positions[3 * Eigen::Index(v) + 1] << ' '
            << mesh.positions[3 * Eigen::Index(v) + 2] << '\n';
    for (int e = 0; e < mesh.topology.num_cells(); ++e) {
        const auto& face = mesh.topology.cell_vertices(e);
        out << "f " << face[0] + 1 << ' ' << face[1] + 1 << ' ' << face[2] + 1 << '\n';
    }
    out.close();
    if (!out)
        throw std::runtime_error("Failed writing OBJ: " + path.string());
}
} // namespace litepgo::io
