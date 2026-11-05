// Export the complete active set on one mesh without changing the Hessian pattern.
#include "litepgo/contact/broad_phase/spatial_hash.hpp"
#include "litepgo/contact/ipc_contact_energy.hpp"
#include "litepgo/mesh/io/obj.hpp"

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>
#include <utility>

namespace {
litepgo::mesh::TriangleMesh contact_scene() {
    const std::array<std::array<double, 18>, 5> scenes{{
        {0, 0, 0, 2, 0, 0, 0, 2, 0, -.2, -.3, .4, -1, -2, 1, -2, -1, 1},
        {0, 0, 0, 2, 0, 0, 0, 2, 0, .7, -.2, .4, 2, -2, 1, -2, -1, 1},
        {0, 0, 0, 2, 0, 0, 0, 2, 0, .4, .3, .4, 2, 2, 1, -1, 2, 1},
        {-1, 0, 0, 1, 0, 0, 0, -4, -2, -.7, -.004, .2, .8, .008, .2, 0, 4, 2},
        {-1, 0, 0, 1, 0, 0, 0, -4, -2, -.7, 0, .2, .8, 0, .2, 0, 4, 2},
    }};
    litepgo::Vec positions(90);
    std::vector<litepgo::mesh::TriangleTopology::Cell> faces;
    for (int group = 0; group < 5; ++group) {
        for (int v = 0; v < 6; ++v)
            for (int k = 0; k < 3; ++k)
                positions[18 * group + 3 * v + k] = scenes[group][3 * v + k] + (k == 0 ? 10 * group : 0);
        const int base = 6 * group;
        faces.push_back({base, base + 1, base + 2});
        faces.push_back({base + 3, base + 4, base + 5});
    }
    return {std::move(positions), litepgo::mesh::TriangleTopology(30, std::move(faces))};
}
} // namespace

int main(int argc, char** argv) {
    using namespace litepgo;
    const std::filesystem::path output = argc > 1 ? argv[1] : LITEPGO_OUTPUT_DIR "/contact_snapshot";
    const auto surface = contact_scene();
    contact::IPCContactOptions options;
    options.dhat = .7;
    options.min_distance = .05;
    const contact::IPCContactEnergy barrier(surface, options, std::make_shared<contact::SpatialHash>());
    Vec x = surface.positions;
    for (int group = 0; group < 5; ++group)
        for (int v = 3; v < 6; ++v)
            x[18 * group + 3 * v + 2] -= .05;
    const auto snapshot = barrier.contact_snapshot(x);
    const auto count = snapshot.distances.size();
    if (count != barrier.num_active_pairs(x) || barrier.hessian_pattern().nonZeros() != 0 ||
        snapshot.distances.minCoeff() != barrier.min_active_distance(x) ||
        std::set<std::string>(snapshot.types.begin(), snapshot.types.end()) !=
            std::set<std::string>{"VV", "VE", "VF", "EE"} ||
        !(snapshot.mollifiers.array() == 0).any())
        throw std::runtime_error("expected the complete active set, all four kinds, and a retained zero-mollifier EE");
    std::filesystem::create_directories(output);
    mesh::write_obj(output / "mesh.obj", {snapshot.positions.reshaped<Eigen::RowMajor>(), surface.topology});
    std::ofstream stream(output / "contacts.csv");
    stream.exceptions(std::ios::failbit | std::ios::badbit);
    stream << "type,v0,v1,v2,v3,distance,weight,mollifier\n" << std::setprecision(17);
    for (Eigen::Index i = 0; i < count; ++i) {
        stream << snapshot.types[i];
        for (int k = 0; k < 4; ++k)
            stream << ',' << snapshot.vertex_ids(i, k);
        stream << ',' << snapshot.distances[i] << ',' << snapshot.weights[i] << ',' << snapshot.mollifiers[i] << '\n';
    }
    stream.close();
    std::cout << "records=" << count;
    for (const auto* kind : {"VV", "VE", "VF", "EE"})
        std::cout << ", " << kind << '=' << std::count(snapshot.types.begin(), snapshot.types.end(), kind);
    std::cout << ", min_distance=" << snapshot.distances.minCoeff()
              << ", zero_mollifiers=" << (snapshot.mollifiers.array() == 0).count() << '\n';
    std::cout << "Saved " << output << '\n';
}
