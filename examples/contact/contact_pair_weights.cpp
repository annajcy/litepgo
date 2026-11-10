#include "litepgo/contact/broad_phase/spatial_hash.hpp"
#include "litepgo/contact/ipc_contact_energy.hpp"

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>

int main(int argc, char** argv) {
    using namespace litepgo;
    const std::array<std::array<int, 2>, 3> groups{{{0, 1}, {1, 2}, {0, 2}}};
    Vec pair(18);
    pair << 0, 0, 0, 10, 0, 0, 0, 10, 0, 2, 2, .5, 6, 2, 4.5, 2, 6, 4.5;
    Vec x(54);
    std::vector<mesh::TriangleTopology::Cell> faces;
    contact::ContactPairWeights weights;
    weights.weights.resize(3, 3);
    weights.weights << 1, .5, 0, .5, 1, 2, 0, 2, 1;
    for (int i = 0; i < 3; ++i) {
        x.segment<18>(18 * i) = pair;
        for (int v = 0; v < 6; ++v) {
            x[18 * i + 3 * v] += 30 * i;
            weights.vertex_groups.push_back(groups[i][v / 3]);
        }
        for (int t = 0; t < 2; ++t) {
            const int base = 6 * i + 3 * t;
            faces.push_back({base, base + 1, base + 2});
            weights.face_groups.push_back(groups[i][t]);
        }
    }
    const mesh::TriangleMesh surface{x, mesh::TriangleTopology(18, std::move(faces))};
    for (const auto edge : contact::contact_edges(surface.topology)) {
        if (weights.vertex_groups[edge[0]] != weights.vertex_groups[edge[1]])
            throw std::runtime_error("example edges must lie inside one uniformly tagged triangle");
        weights.edge_groups.push_back(weights.vertex_groups[edge[0]]);
    }
    contact::IPCContactOptions options;
    options.dhat = .7;
    options.min_distance = .05;
    const contact::IPCContactEnergy reference(surface, options, std::make_shared<contact::SpatialHash>());
    const contact::IPCContactEnergy grouped(surface, options, std::make_shared<contact::SpatialHash>(), weights);
    const std::array snapshots{reference.contact_snapshot(x), grouped.contact_snapshot(x)};
    const std::array gradients{reference.evaluate(x)->gradient(), grouped.evaluate(x)->gradient()};
    const std::filesystem::path output = argc > 1 ? argv[1] : LITEPGO_OUTPUT_DIR "/contact_pair_weights.csv";
    if (!output.parent_path().empty())
        std::filesystem::create_directories(output.parent_path());
    std::ofstream stream(output);
    stream.exceptions(std::ios::failbit | std::ios::badbit);
    stream << "pair,group_a,group_b,multiplier,reference_energy,weighted_energy,reference_force,weighted_force,active_"
              "records\n"
           << std::setprecision(17);
    std::array<double, 2> totals{};
    for (int i = 0; i < 3; ++i) {
        std::array<double, 2> values{}, forces{};
        std::array<int, 2> counts{};
        for (int mode = 0; mode < 2; ++mode) {
            const auto& s = snapshots[mode];
            for (Eigen::Index row = 0; row < s.distances.size(); ++row)
                if (s.vertex_ids(row, 0) / 6 == i) {
                    const double d = s.distances[row] * s.distances[row] - options.min_distance * options.min_distance;
                    const double h = options.dhat * (options.dhat + 2 * options.min_distance);
                    values[mode] -= s.weights[row] * s.mollifiers[row] * (d - h) * (d - h) * std::log(d / h);
                    ++counts[mode];
                }
            for (int v = 3; v < 6; ++v)
                forces[mode] += gradients[mode][18 * i + 3 * v + 2];
            forces[mode] = std::abs(forces[mode]);
            totals[mode] += values[mode];
        }
        const auto [a, b] = groups[i];
        const double scale = weights.weights(a, b);
        if (counts[0] != 1 || counts[1] != int(scale > 0) ||
            std::abs(values[1] - scale * values[0]) > 1e-12 * values[0] ||
            std::abs(forces[1] - scale * forces[0]) > 1e-12 * forces[0])
            throw std::runtime_error("expected each pair's energy and force to follow its multiplier");
        stream << i << ',' << a << ',' << b << ',' << scale << ',' << values[0] << ',' << values[1] << ',' << forces[0]
               << ',' << forces[1] << ',' << counts[1] << '\n';
        std::cout << "groups=(" << a << ',' << b << "), multiplier=" << scale << ", energy=" << values[1]
                  << ", force=" << forces[1] << ", records=" << counts[1] << '\n';
    }
    if (std::abs(totals[0] - reference.evaluate(x)->value()) > 1e-12 * totals[0] ||
        std::abs(totals[1] - grouped.evaluate(x)->value()) > 1e-12 * totals[0])
        throw std::runtime_error("snapshot contributions must sum to the actual energy");
    stream.close();
    std::cout << "Saved " << output << '\n';
}
