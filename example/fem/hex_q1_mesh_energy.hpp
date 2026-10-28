#pragma once

#include "litepgo/dof/vertex_dof_map.hpp"
#include "litepgo/energy/mesh_energy.hpp"
#include "litepgo/fem/basis/3d/hex_q1_basis.hpp"
#include "litepgo/fem/field/deformation_gradient_field.hpp"
#include "litepgo/fem/field/identity_field.hpp"
#include "litepgo/fem/field/interpolated_field.hpp"
#include "litepgo/fem/kinematics/volumetric_element.hpp"
#include "litepgo/fem/material/volumetric/neo_hookean.hpp"
#include "litepgo/fem/material/volumetric/volumetric_material_local_energy.hpp"
#include "litepgo/fem/quadrature/3d/gauss_legendre_3d.hpp"

#include <memory>
#include <utility>
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
    const HexTopology topology(12, {{0, 1, 2, 3, 4, 5, 6, 7}, {1, 8, 9, 2, 5, 10, 11, 6}});
    // Three displacement components per vertex; shared vertices reuse the same DOFs.
    const VertexDofMap map(topology, 3);
    std::vector<InterpolatedField3D<HexQ1Basis>> reference_fields;
    std::vector<VolumetricElement3D<GaussLegendre3D<2>>> elements;
    for (int e = 0; e < map.num_elements(); ++e) {
        const InterpolatedField3D<HexQ1Basis> X_field(map.gather(e, X_nodes));
        elements.emplace_back(X_field);
        reference_fields.push_back(X_field);
    }
    const auto reference_data = std::make_shared<const decltype(reference_fields)>(std::move(reference_fields));
    const auto element_data = std::make_shared<const decltype(elements)>(std::move(elements));
    const IdentityField3D Fp_field(Eigen::Matrix3d::Identity().reshaped());
    const IdentityField3D theta_field(Eigen::Vector2d(2, 3));
    return MeshEnergy(
        map,
        [element_data, reference_data, Fp_field, theta_field](int e, const Eigen::MatrixXd& u_local, Vec* gradient,
                                                              Eigen::MatrixXd* hessian) {
            const auto& element = (*element_data)[static_cast<std::size_t>(e)];
            const InterpolatedField3D<HexQ1Basis> displacement(u_local);
            const DeformationGradientField3D deformation(displacement, (*reference_data)[static_cast<std::size_t>(e)]);
            const VolumetricMaterialLocalEnergyF3D<NeoHookean> local_energy(deformation, Fp_field, theta_field);
            return element.eval(local_energy, gradient, hessian);
        },
        [element_data, reference_data, Fp_field, theta_field](int e, const Eigen::MatrixXd& u_local,
                                                              const Eigen::MatrixXd& du_local) {
            const auto& element = (*element_data)[static_cast<std::size_t>(e)];
            const InterpolatedField3D<HexQ1Basis> displacement(u_local);
            const DeformationGradientField3D deformation(displacement, (*reference_data)[static_cast<std::size_t>(e)]);
            const VolumetricMaterialLocalEnergyF3D<NeoHookean> local_energy(deformation, Fp_field, theta_field);
            const Vec delta_coeffs = du_local.reshaped<Eigen::RowMajor>();
            return element.max_feasible_step(local_energy, delta_coeffs);
        });
}

} // namespace litepgo::example
