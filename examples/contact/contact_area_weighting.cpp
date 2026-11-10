// Compare area weighting on fixed parallel sheets under uniform refinement; no solve or coefficient conversion.
#include "litepgo/contact/broad_phase/spatial_hash.hpp"
#include "litepgo/contact/ipc_contact_energy.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
constexpr double side = 0.16, gap = 0.001, dhat = 0.002, kappa = 1e5;

litepgo::mesh::TriangleMesh parallel_sheets(int n) {
    const int sheet_vertices = (n + 1) * (n + 1);
    litepgo::Vec positions(6 * sheet_vertices);
    std::vector<litepgo::mesh::TriangleTopology::Cell> faces;
    for (int layer = 0; layer < 2; ++layer) {
        const int base = layer * sheet_vertices;
        for (int j = 0; j <= n; ++j)
            for (int i = 0; i <= n; ++i)
                positions.segment<3>(3 * (base + j * (n + 1) + i)) =
                    Eigen::Vector3d(side * i / n, side * j / n, layer * gap);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                const int a = base + j * (n + 1) + i;
                faces.push_back({a, a + 1, a + n + 2});
                faces.push_back({a, a + n + 2, a + n + 1});
            }
    }
    return {std::move(positions), litepgo::mesh::TriangleTopology(2 * sheet_vertices, std::move(faces))};
}
} // namespace

int main(int argc, char** argv) {
    using namespace litepgo;
    const std::filesystem::path output = argc > 1 ? argv[1] : LITEPGO_OUTPUT_DIR "/contact_area_weighting.csv";
    if (!output.parent_path().empty())
        std::filesystem::create_directories(output.parent_path());
    std::ofstream stream(output);
    stream.exceptions(std::ios::failbit | std::ios::badbit);
    stream << "subdivisions,edge_length,triangles,use_area_weighting,energy,normal_force\n" << std::setprecision(17);
    for (int n : {4, 8, 16}) {
        const auto surface = parallel_sheets(n);
        for (bool weighted : {false, true}) {
            contact::IPCContactOptions options;
            options.dhat = dhat;
            options.use_area_weighting = weighted;
            const contact::IPCContactEnergy barrier(surface, options, std::make_shared<contact::SpatialHash>());
            const double value = kappa * barrier.evaluate(surface.positions)->value();
            const Vec gradient = kappa * barrier.evaluate(surface.positions)->gradient();
            double force = 0, balance = 0;
            const int sheet_vertices = (n + 1) * (n + 1);
            for (int v = 0; v < 2 * sheet_vertices; ++v) {
                balance += gradient[3 * v + 2];
                if (v < sheet_vertices)
                    force += gradient[3 * v + 2];
            }
            force = std::abs(force);
            if (!std::isfinite(value) || !std::isfinite(force) || value <= 0 || force <= 0 ||
                std::abs(balance) > 1e-10 * force)
                throw std::runtime_error("expected finite positive energy and balanced normal forces");
            stream << n << ',' << side / n << ',' << surface.topology.num_cells() << ',' << weighted << ',' << value
                   << ',' << force << '\n';
            std::cout << "n=" << n << ", area_weighting=" << weighted << ", energy=" << value
                      << ", normal_force=" << force << '\n';
        }
    }
    stream.close();
    std::cout << "Saved " << output << '\n';
}
