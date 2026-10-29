#include "litepgo/fem/material/shell/koiter_stvk.hpp"
#include "litepgo/fem/shell_fem_energy.hpp"
#include "litepgo/mesh/triangle_mid_edge_stencils.hpp"

#include <iostream>
#include <memory>
#include <utility>
#include <vector>

int main() {
    using namespace litepgo;
    // A flat patch with face areas 1, 2, 1/2, 2 and consistent winding.
    Eigen::Matrix<double, 6, 3> rest_nodes;
    rest_nodes << 0, 0, 0, 2, 0, 0, 0, 1, 0, 2, 2, 0, -1, 0, 0, 0, -2, 0;
    const Vec rest_positions = rest_nodes.reshaped<Eigen::RowMajor>();

    // Face 0 has three neighbors; boundary faces each have one neighbor.
    const auto topology = TriangleTopology(6, {{0, 1, 2}, {2, 1, 3}, {0, 2, 4}, {1, 0, 5}});
    const auto stencils = make_triangle_mid_edge_stencils(topology);
    const StencilDofMap map(topology.num_vertices(), stencils.vertices, 3);
    std::vector<TriangleMidEdgeRestData> rest;
    for (int f = 0; f < map.num_elements(); ++f)
        rest.emplace_back(map.gather(f, rest_positions), stencils.opposite[static_cast<std::size_t>(f)]);

    const double thickness = 0.1, Y = 100, nu = 0.3;
    // Plane-stress Lame parameters for a shell.
    const Eigen::Vector2d theta(Y * nu / (1 - nu * nu), Y / (2 * (1 + nu)));
    auto elastic = std::make_shared<ShellFemEnergy<KoiterStVK>>(
        map, std::move(rest), std::vector<double>(map.num_elements(), thickness),
        std::vector<Eigen::VectorXd>(map.num_elements(), theta));

    const Vec u = 0.1 * rest_positions;
    Vec gradient;
    SpMat hessian = elastic->hessian_pattern();
    const double value = elastic->eval(u, &gradient, &hessian);
    std::cout << "E=" << value << ", |g|=" << gradient.norm() << ", H nnz=" << hessian.nonZeros() << '\n';
}
