#pragma once

#include "litepgo/dof/vertex_dof_map.hpp"
#include "litepgo/energy/mesh_energy.hpp"
#include "litepgo/fem/basis/hex_q1_basis.hpp"
#include "litepgo/fem/field/deformation_gradient_field.hpp"
#include "litepgo/fem/field/identity_field.hpp"
#include "litepgo/fem/local/integrate_volume.hpp"
#include "litepgo/fem/local/material_local_energy.hpp"
#include "litepgo/fem/material/neo_hookean.hpp"
#include "litepgo/fem/quadrature/gauss_legendre_3d.hpp"

#include <memory>
#include <stdexcept>
#include <vector>

namespace litepgo::example {

inline Vec hex_q1_rest_positions() {
    // Two adjacent hexes of widths 1 and 2, sharing the face at X = 1.
    Eigen::Matrix<double, 12, 3> X_nodes;
    X_nodes << 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 3, 0, 0, 3, 1, 0, 3, 0, 1, 3, 1,
        1;
    return X_nodes.reshaped<Eigen::RowMajor>();
}

inline MeshEnergy make_hex_q1_mesh_energy(const Vec& X_nodes) {
    // Shared face vertices 1, 2, 5, 6 reuse the same displacement DOFs in both cells.
    const HexTopology topology(12, {{0, 1, 2, 3, 4, 5, 6, 7}, {1, 8, 9, 2, 5, 10, 11, 6}});
    // Three physical displacement components per vertex: global index 3*vertex + component.
    const VertexDofMap map(topology, 3);
    std::vector<Interpolation<HexQ1Basis>> rest_geometries;
    for (int e = 0; e < map.num_elements(); ++e)
        rest_geometries.emplace_back(map.gather(e, X_nodes));
    const auto Fp = std::make_shared<IdentityField<Eigen::Matrix3d>>(Eigen::Matrix3d::Identity());
    const auto theta = std::make_shared<IdentityField<Eigen::VectorXd>>(Eigen::Vector2d(2, 3));

    return MeshEnergy(
        map,
        [rest_geometries, Fp, theta](int e, const Eigen::MatrixXd& x_local, Vec* dE_dx_local,
                                     Eigen::MatrixXd* d2E_dx_local2) {
            // Geometry and gathered displacement both belong to element e.
            const auto& rest_geometry = rest_geometries[static_cast<std::size_t>(e)];
            const auto F = std::make_shared<DeformationGradientField<HexQ1Basis>>(x_local);
            const MaterialLocalEnergyF<NeoHookean> local_energy({F, Fp, theta});
            return integrate_volume<GaussLegendre3D<2>>(rest_geometry, local_energy, dE_dx_local, d2E_dx_local2);
        },
        [](int, const auto&, const auto&) -> double {
            // shortcut: evaluation only; supply feasible step bounds before using a solver.
            throw std::logic_error("Feasible step bounds are not implemented in this example");
        });
}

} // namespace litepgo::example
