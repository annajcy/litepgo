#include "litepgo/io/msh.hpp"
#include <gmsh.h>
#include <mutex>
#include <numeric>
#include <unordered_map>

namespace litepgo::io {
namespace {
std::mutex gmsh_mutex;
struct Session {
    Session() {
        if (gmsh::isInitialized())
            throw std::runtime_error("MSH I/O requires an independent Gmsh session");
        gmsh::initialize(0, nullptr, false);
    }
    ~Session() {
        try {
            gmsh::finalize();
        } catch (...) {
        }
    }
};
template <int K>
MeshData<K> extract(const std::vector<std::size_t>& tags, const std::vector<double>& xyz,
                    const std::vector<std::size_t>& nodes) {
    if (tags.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) || xyz.size() != tags.size() * 3 ||
        nodes.size() % K)
        throw std::runtime_error("Invalid or oversized MSH geometry");
    std::unordered_map<std::size_t, int> indices;
    for (std::size_t i = 0; i < tags.size(); ++i)
        if (!indices.emplace(tags[i], static_cast<int>(i)).second)
            throw std::runtime_error("Duplicate MSH node tag");
    Vec positions(static_cast<Eigen::Index>(xyz.size()));
    std::copy(xyz.begin(), xyz.end(), positions.data());
    std::vector<typename MeshTopology<K>::Cell> cells(nodes.size() / K);
    for (std::size_t e = 0; e < cells.size(); ++e)
        for (int j = 0; j < K; ++j) {
            auto it = indices.find(nodes[e * K + j]);
            if (it == indices.end())
                throw std::runtime_error("Missing MSH node");
            cells[e][j] = it->second;
        }
    MeshData<K> mesh{std::move(positions), MeshTopology<K>(static_cast<int>(tags.size()), std::move(cells))};
    mesh.validate();
    return mesh;
}
} // namespace
VolumeMesh read_msh(const std::filesystem::path& path) {
    std::lock_guard lock(gmsh_mutex);
    try {
        Session session;
        gmsh::open(path.string());
        std::vector<std::size_t> tags;
        std::vector<double> xyz, params;
        gmsh::model::mesh::getNodes(tags, xyz, params);
        std::vector<int> types;
        std::vector<std::vector<std::size_t>> elements, nodes;
        gmsh::model::mesh::getElements(types, elements, nodes, 3);
        if (types.size() != 1 || (types[0] != 4 && types[0] != 5))
            throw std::runtime_error("MSH requires homogeneous tet4 or hex8 volume elements");
        if (types[0] == 4)
            return extract<4>(tags, xyz, nodes[0]);
        return extract<8>(tags, xyz, nodes[0]);
    } catch (const std::string& error) {
        throw std::runtime_error(error);
    }
}
void write_msh(const std::filesystem::path& path, const VolumeMesh& mesh) {
    if (path.extension() != ".msh")
        throw std::invalid_argument("MSH output requires a .msh extension");
    std::visit(
        [](const auto& data) {
            data.validate();
            if (data.topology.num_cells() == 0)
                throw std::invalid_argument("MSH output requires volume cells to preserve the cell type");
        },
        mesh);
    std::lock_guard lock(gmsh_mutex);
    try {
        Session session;
        gmsh::model::add("litepgo");
        const int entity = gmsh::model::addDiscreteEntity(3);
        std::visit(
            [&](const auto& data) {
                constexpr int K = std::decay_t<decltype(data.topology)>::vertices_per_cell;
                std::vector<std::size_t> tags(data.topology.num_vertices());
                std::iota(tags.begin(), tags.end(), std::size_t{1});
                std::vector<double> xyz(data.positions.size());
                std::copy_n(data.positions.data(), xyz.size(), xyz.begin());
                gmsh::model::mesh::addNodes(3, entity, tags, xyz);
                std::vector<std::size_t> nodes;
                for (int e = 0; e < data.topology.num_cells(); ++e)
                    for (int v : data.topology.cell_vertices(e))
                        nodes.push_back(static_cast<std::size_t>(v) + 1);
                gmsh::model::mesh::addElementsByType(entity, K == 4 ? 4 : 5, {}, nodes);
            },
            mesh);
        gmsh::option::setNumber("Mesh.MshFileVersion", 4.1);
        gmsh::option::setNumber("Mesh.Binary", 0);
        gmsh::option::setNumber("Mesh.SaveAll", 1);
        gmsh::write(path.string());
    } catch (const std::string& error) {
        throw std::runtime_error(error);
    }
}
} // namespace litepgo::io
