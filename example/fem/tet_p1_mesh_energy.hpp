#pragma once

#include "litepgo/dof/vertex_dof_map.hpp"
#include "litepgo/energy/mesh_energy.hpp"
#include "litepgo/fem/basis/3d/tet_p1_basis.hpp"
#include "litepgo/fem/field/deformation_gradient_field.hpp"
#include "litepgo/fem/field/identity_field.hpp"
#include "litepgo/fem/field/interpolated_field.hpp"
#include "litepgo/fem/material/volumetric/volumetric_material_local_energy.hpp"
#include "litepgo/fem/material/volumetric/neo_hookean.hpp"
#include "litepgo/fem/quadrature/3d/tet_centroid_quadrature.hpp"

#include <Eigen/LU>
#include <cmath>
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
    std::vector<InterpolatedField3D<TetP1Basis>> X_fields;
    for (int e = 0; e < map.num_elements(); ++e)
        X_fields.emplace_back(map.gather(e, X_nodes));
    const IdentityField3D Fp_field(Eigen::Matrix3d::Identity().reshaped());
    const IdentityField3D theta_field(Eigen::Vector2d(2, 3));

    return MeshEnergy(
        map,
        [X_fields, Fp_field, theta_field](int e, const Eigen::MatrixXd& x_local, Vec* dE_dx_local,
                                          Eigen::MatrixXd* d2E_dx_local2) {
            // Geometry and gathered displacement both belong to element e.
            const auto& X_field = X_fields[static_cast<std::size_t>(e)];
            const auto& u_local = x_local;
            const auto u_field = InterpolatedField3D<TetP1Basis>(u_local);
            const auto deformation = DeformationGradientField3D(u_field, X_field);
            const VolumetricMaterialLocalEnergyF3D<NeoHookean> local_energy(deformation, Fp_field, theta_field);
            if (dE_dx_local)
                dE_dx_local->setZero(local_energy.num_dofs());
            if (d2E_dx_local2)
                d2E_dx_local2->setZero(local_energy.num_dofs(), local_energy.num_dofs());
            using Q = TetCentroidQuadrature;
            double E_element = 0;
            for (int q = 0; q < Q::num_points; ++q) {
                const Eigen::Vector3d xi = Q::points().row(q).transpose();
                const auto dX_dxi = X_field.eval_dvalue_dxi(xi);
                if (dX_dxi.rows() != 3 || dX_dxi.cols() != 3 || !dX_dxi.allFinite())
                    throw std::invalid_argument("tet_p1 requires a finite 3-by-3 coordinate derivative");
                const double determinant = Eigen::Matrix3d(dX_dxi).determinant();
                const double weight = Q::weights()[q] * determinant;
                if (!std::isfinite(determinant) || determinant <= 0 || !std::isfinite(weight))
                    throw std::invalid_argument(
                        "tet_p1 requires a finite positive coordinate determinant and finite volume weight");
                E_element += local_energy.accumulate(xi, weight, dE_dx_local, d2E_dx_local2);
            }
            return E_element;
        },
        [](int, const auto&, const auto&) -> double {
            // shortcut: evaluation only; supply feasible step bounds before using a solver.
            throw std::logic_error("Feasible step bounds are not implemented in this example");
        });
}

} // namespace litepgo::example
