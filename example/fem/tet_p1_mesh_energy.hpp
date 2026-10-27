#pragma once

#include "litepgo/dof/vertex_dof_map.hpp"
#include "litepgo/energy/mesh_energy.hpp"
#include "litepgo/fem/basis/tet_p1_basis.hpp"
#include "litepgo/fem/field/identity_field.hpp"
#include "litepgo/fem/field/interpolated_field.hpp"
#include "litepgo/fem/kinematics/deformation_gradient.hpp"
#include "litepgo/fem/local/integrate_volume.hpp"
#include "litepgo/fem/local/material_local_energy.hpp"
#include "litepgo/fem/material/neo_hookean.hpp"
#include "litepgo/fem/quadrature/tet_centroid_quadrature.hpp"

#include <stdexcept>
#include <vector>

namespace litepgo::example {

inline Vec tet_p1_rest_positions() {
    // Two positively oriented tets share a face; their volumes are 1/6 and 1/3.
    Eigen::Matrix<double, 5, 3> X_nodes;
    X_nodes << 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0.3, 0.2, -2;
    return X_nodes.reshaped<Eigen::RowMajor>();
}

inline MeshEnergy make_tet_p1_mesh_energy(const Vec& X_nodes) {
    // Shared face vertices 0, 1, 2 reuse the same displacement DOFs in both cells.
    const TetTopology topology(5, {{0, 1, 2, 3}, {0, 2, 1, 4}});
    // Three physical displacement components per vertex: global index 3*vertex + component.
    const VertexDofMap map(topology, 3);
    std::vector<InterpolatedField<TetP1Basis>> X_fields;
    for (int e = 0; e < map.num_elements(); ++e)
        X_fields.emplace_back(map.gather(e, X_nodes));
    const IdentityField<Eigen::Matrix3d> Fp_field(Eigen::Matrix3d::Identity());
    const IdentityField<Eigen::VectorXd> theta_field(Eigen::Vector2d(2, 3));

    return MeshEnergy(
        map,
        [X_fields, Fp_field, theta_field](int e, const Eigen::MatrixXd& x_local, Vec* dE_dx_local,
                                          Eigen::MatrixXd* d2E_dx_local2) {
            // Geometry and gathered displacement both belong to element e.
            const auto& X_field = X_fields[static_cast<std::size_t>(e)];
            const auto& u_local = x_local;
            const auto u_field = InterpolatedField<TetP1Basis>(u_local);
            const auto deformation = DeformationGradient(u_field, X_field);
            const auto local_energy =
                make_material_local_energy<NeoHookean, MaterialVariable::F>(deformation, Fp_field, theta_field);
            return integrate_volume<TetCentroidQuadrature>(local_energy, dE_dx_local, d2E_dx_local2);
        },
        [](int, const auto&, const auto&) -> double {
            // shortcut: evaluation only; supply feasible step bounds before using a solver.
            throw std::logic_error("Feasible step bounds are not implemented in this example");
        });
}

} // namespace litepgo::example
