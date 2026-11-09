#include "litepgo/mesh/io/alembic.hpp"
#include "litepgo/mesh/io/obj.hpp"

#include <Alembic/AbcCoreOgawa/All.h>
#include <Alembic/AbcGeom/All.h>

#include <cmath>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>

namespace litepgo::mesh {
void write_abc_from_objs(const std::filesystem::path& path, const std::vector<std::filesystem::path>& frame_paths,
                         double fps) {
    namespace abc = Alembic::Abc;
    namespace geom = Alembic::AbcGeom;
    if (frame_paths.empty())
        throw std::invalid_argument("Alembic export requires at least one OBJ frame");
    if (!std::isfinite(fps) || fps <= 0)
        throw std::invalid_argument("Alembic export requires a finite positive fps");
    const double interval = 1.0 / fps;
    if (interval >= abc::TimeSamplingType::AcyclicTimePerCycle() ||
        !std::isfinite(interval * static_cast<double>(frame_paths.size() - 1)))
        throw std::invalid_argument("Alembic export requires a finite positive fps and finite sample times");

    const auto read_frame = [&](std::size_t i) {
        try {
            auto frame = read_obj(frame_paths[i]);
            if (frame.topology.num_cells() == 0)
                throw std::invalid_argument("OBJ has no faces");
            if (frame.positions.cwiseAbs().maxCoeff() > std::numeric_limits<float>::max())
                throw std::invalid_argument("OBJ coordinates exceed the Alembic float range");
            return frame;
        } catch (const std::exception& e) {
            throw std::runtime_error("Alembic frame " + std::to_string(i) + " (" + frame_paths[i].string() +
                                     "): " + e.what());
        }
    };
    const auto first = read_frame(0);
    const auto& topology = first.topology;
    std::vector<abc::int32_t> counts(static_cast<std::size_t>(topology.num_cells()), 3);
    std::vector<abc::int32_t> indices;
    indices.reserve(3 * counts.size());
    for (int f = 0; f < topology.num_cells(); ++f) {
        const auto& face = topology.cell_vertices(f);
        // Alembic uses clockwise winding for front faces.
        indices.insert(indices.end(), {face[2], face[1], face[0]});
    }

    std::random_device random;
    const auto temporary =
        path.parent_path() / (".litepgo-abc-" + std::to_string(random()) + "-" + std::to_string(random()));
    if (!std::filesystem::create_directory(temporary))
        throw std::runtime_error("Cannot reserve temporary Alembic directory: " + temporary.string());
    const auto staging = temporary / "archive.abc";
    try {
        std::ofstream stream(staging, std::ios::binary);
        if (!stream)
            throw std::runtime_error("Cannot open temporary Alembic archive: " + staging.string());
        {
            abc::OArchive archive(Alembic::AbcCoreOgawa::WriteArchive()(&stream, {}));
            const auto sampling = archive.addTimeSampling(abc::TimeSampling(interval, 0.0));
            geom::OPolyMesh object(archive.getTop(), "mesh", sampling);
            std::vector<abc::V3f> positions(static_cast<std::size_t>(topology.num_vertices()));
            const auto sample_positions = [&](const TriangleMesh& frame) {
                for (std::size_t v = 0; v < positions.size(); ++v) {
                    const auto offset = 3 * static_cast<Eigen::Index>(v);
                    positions[v] = {static_cast<float>(frame.positions[offset]),
                                    static_cast<float>(frame.positions[offset + 1]),
                                    static_cast<float>(frame.positions[offset + 2])};
                }
                return abc::V3fArraySample(positions);
            };
            auto& schema = object.getSchema();
            schema.set({sample_positions(first), abc::Int32ArraySample(indices), abc::Int32ArraySample(counts)});
            for (std::size_t i = 1; i < frame_paths.size(); ++i) {
                const auto frame = read_frame(i);
                bool matches = frame.topology.num_vertices() == topology.num_vertices() &&
                               frame.topology.num_cells() == topology.num_cells();
                for (int f = 0; matches && f < topology.num_cells(); ++f)
                    matches = frame.topology.cell_vertices(f) == topology.cell_vertices(f);
                if (!matches)
                    throw std::runtime_error("Alembic frame " + std::to_string(i) + " (" + frame_paths[i].string() +
                                             "): topology differs from the first frame");
                schema.set(geom::OPolyMeshSchema::Sample(sample_positions(frame)));
            }
        }
        stream.close();
        if (!stream)
            throw std::runtime_error("Failed writing Alembic archive: " + path.string());
        std::filesystem::rename(staging, path);
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove_all(temporary, ignored);
        throw;
    }
    std::error_code ignored;
    std::filesystem::remove(temporary, ignored);
}
} // namespace litepgo::mesh
