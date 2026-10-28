#pragma once

#include "litepgo/dof/vertex_dof_map.hpp"
#include "litepgo/energy/mesh_energy.hpp"
#include "litepgo/fem/basis/3d/tet_p1_basis.hpp"
#include "litepgo/fem/field/deformation_gradient_field.hpp"
#include "litepgo/fem/field/identity_field.hpp"
#include "litepgo/fem/field/interpolated_field.hpp"
#include "litepgo/fem/kinematics/volumetric_element.hpp"
#include "litepgo/fem/material/volumetric/neo_hookean.hpp"
#include "litepgo/fem/material/volumetric/volumetric_material_local_energy.hpp"
#include "litepgo/fem/quadrature/3d/tet_centroid_quadrature.hpp"

#include <memory>
#include <utility>
#include <vector>

namespace litepgo::example {

inline Vec tet_p1_rest_positions() {
    // Two positively oriented tets share a face; their volumes are 1/6 and 1/3.
    Eigen::Matrix<double, 5, 3> X_nodes;
    X_nodes << 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0.3, 0.2, -2;
    return X_nodes.reshaped<Eigen::RowMajor>();
}

inline MeshEnergy make_tet_p1_mesh_energy(const Vec& X_nodes) {
    const TetTopology topology(5, {{0, 1, 2, 3}, {0, 2, 1, 4}});
    // Three displacement components per vertex; shared vertices reuse the same DOFs.
    const VertexDofMap map(topology, 3);
    struct ReferenceData {
        InterpolatedField3D<TetP1Basis> X;
        VolumetricElement3D<TetCentroidQuadrature> integration;
        explicit ReferenceData(Eigen::MatrixXd X_local) : X(std::move(X_local)), integration(X) {}
    };
    std::vector<ReferenceData> reference;
    reference.reserve(map.num_elements());
    for (int e = 0; e < map.num_elements(); ++e)
        reference.emplace_back(map.gather(e, X_nodes));
    const auto reference_data = std::make_shared<const decltype(reference)>(std::move(reference));
    const IdentityField3D Fp_field(Eigen::Matrix3d::Identity().reshaped());
    const IdentityField3D theta_field(Eigen::Vector2d(2, 3));
    return MeshEnergy(
        map,
        [reference_data, Fp_field, theta_field](int e, const Eigen::MatrixXd& u_local, Vec* gradient,
                                                Eigen::MatrixXd* hessian) {
            const auto& r = (*reference_data)[static_cast<std::size_t>(e)];
            const InterpolatedField3D<TetP1Basis> displacement(u_local);
            const DeformationGradientField3D deformation(displacement, r.X);
            const VolumetricMaterialLocalEnergyF3D<NeoHookean> local_energy(deformation, Fp_field, theta_field);
            return r.integration.eval(local_energy, gradient, hessian);
        },
        [reference_data, Fp_field, theta_field](int e, const Eigen::MatrixXd& u_local,
                                                const Eigen::MatrixXd& du_local) {
            const auto& r = (*reference_data)[static_cast<std::size_t>(e)];
            const InterpolatedField3D<TetP1Basis> displacement(u_local);
            const DeformationGradientField3D deformation(displacement, r.X);
            const VolumetricMaterialLocalEnergyF3D<NeoHookean> local_energy(deformation, Fp_field, theta_field);
            const Vec delta_coeffs = du_local.reshaped<Eigen::RowMajor>();
            return r.integration.max_feasible_step(local_energy, delta_coeffs);
        });
}

} // namespace litepgo::example
