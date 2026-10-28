#pragma once

#include "litepgo/dof/stencil_dof_map.hpp"
#include "litepgo/energy/mesh_energy.hpp"
#include "litepgo/fem/material/shell/koiter_stvk.hpp"
#include "litepgo/fem/material/shell/shell_material_local_energy.hpp"
#include "litepgo/fem/quadrature/2d/triangle_centroid_quadrature.hpp"
#include "litepgo/mesh/triangle_mid_edge_stencils.hpp"

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace litepgo::example {

inline TriangleTopology shell_stvk_topology() {
    // Face 0 has three neighbors; boundary faces each have one neighbor.
    return TriangleTopology(6, {{0, 1, 2}, {2, 1, 3}, {0, 2, 4}, {1, 0, 5}});
}

inline Vec shell_stvk_rest_positions() {
    // A flat patch with face areas 1, 2, 1/2, 2 and consistent winding.
    Eigen::Matrix<double, 6, 3> X_nodes;
    X_nodes << 0, 0, 0, 2, 0, 0, 0, 1, 0, 2, 2, 0, -1, 0, 0, 0, -2, 0;
    return X_nodes.reshaped<Eigen::RowMajor>();
}

inline MeshEnergy make_shell_stvk_mesh_energy(const Vec& X_nodes) {
    const auto topology = shell_stvk_topology();
    const auto stencils = make_triangle_mid_edge_stencils(topology);
    const StencilDofMap map(topology.num_vertices(), stencils.vertices, 3);
    std::vector<TriangleMidEdgeRestData> rest;
    for (int f = 0; f < map.num_elements(); ++f)
        rest.emplace_back(map.gather(f, X_nodes), stencils.opposite[static_cast<std::size_t>(f)]);

    const auto reference_data = std::make_shared<const decltype(rest)>(std::move(rest));

    const double thickness = 0.1, Y = 100, nu = 0.3;
    // Plane-stress Lame parameters for a shell.
    const Eigen::Vector2d theta(Y * nu / (1 - nu * nu), Y / (2 * (1 + nu)));
    return MeshEnergy(
        map,
        [reference_data, thickness, theta](int f, const Eigen::MatrixXd& u_local, Vec* g, Eigen::MatrixXd* H) {
            const auto& r = (*reference_data)[static_cast<std::size_t>(f)];
            const ShellMaterialLocalEnergy2D<KoiterStVK> local_energy(r, u_local, thickness, theta);
            if (g)
                g->setZero(local_energy.num_dofs());
            if (H)
                H->setZero(local_energy.num_dofs(), local_energy.num_dofs());
            using Q = TriangleCentroidQuadrature;
            double E_face = 0;
            for (int q = 0; q < Q::num_points; ++q) {
                const Eigen::Vector2d xi = Q::points().row(q).transpose();
                const double weight = Q::weights()[q] * r.sqrt_det_a_bar;
                E_face += local_energy.accumulate(xi, weight, g, H);
            }
            return E_face;
        },
        [reference_data, thickness, theta](int f, const Eigen::MatrixXd& u_local, const Eigen::MatrixXd& du_local) {
            const auto& r = (*reference_data)[static_cast<std::size_t>(f)];
            // Validate the complete starting material state as well as the geometry.
            const ShellMaterialLocalEnergy2D<KoiterStVK> local_energy(r, u_local, thickness, theta);
            local_energy.eval_value(Eigen::Vector2d::Constant(1.0 / 3));
            return eval_triangle_mid_edge_feasible_step(r.X_local + u_local, du_local, r.opposite);
        });
}

} // namespace litepgo::example
