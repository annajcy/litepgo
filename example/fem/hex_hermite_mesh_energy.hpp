#pragma once

#include "litepgo/dof/vertex_dof_map.hpp"
#include "litepgo/energy/mesh_energy.hpp"
#include "litepgo/fem/basis/3d/hex_hermite_basis.hpp"
#include "litepgo/fem/basis/3d/hex_q1_basis.hpp"
#include "litepgo/fem/field/deformation_gradient_field.hpp"
#include "litepgo/fem/field/identity_field.hpp"
#include "litepgo/fem/field/interpolated_field.hpp"
#include "litepgo/fem/kinematics/hex_hermite_dof_transform.hpp"
#include "litepgo/fem/kinematics/volumetric_element.hpp"
#include "litepgo/fem/material/volumetric/neo_hookean.hpp"
#include "litepgo/fem/material/volumetric/volumetric_material_local_energy.hpp"
#include "litepgo/fem/quadrature/3d/gauss_legendre_3d.hpp"

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace litepgo::example {

inline HexTopology hex_hermite_topology() {
    // Shared face at X = 1: vertices 1, 2, 5, 6 reuse all 24 physical DOFs per vertex.
    return HexTopology(12, {{0, 1, 2, 3, 4, 5, 6, 7}, {1, 8, 9, 2, 5, 10, 11, 6}});
}

inline Vec hex_hermite_rest_positions() {
    // Two boxes of sizes (1,2,3) and (2,2,3) share the face at X = 1.
    Eigen::Matrix<double, 12, 3> X_nodes;
    X_nodes << 0, 0, 0, 1, 0, 0, 1, 2, 0, 0, 2, 0, 0, 0, 3, 1, 0, 3, 1, 2, 3, 0, 2, 3, 3, 0, 0, 3, 2, 0, 3, 0, 3, 3, 2,
        3;
    return X_nodes.reshaped<Eigen::RowMajor>();
}

inline Eigen::MatrixXd hex_hermite_reference_coefficients(const Eigen::MatrixXd& u_physical_local, const Vec& scales) {
    if (u_physical_local.rows() != 8 || u_physical_local.cols() != 24 || scales.size() != 192 ||
        !u_physical_local.allFinite() || !scales.allFinite())
        throw std::invalid_argument("Hex Hermite example requires finite 8x24 physical DOFs and 192 scales");
    Eigen::MatrixXd u_local = u_physical_local.reshaped<Eigen::RowMajor>(64, 3);
    for (int row = 0; row < 64; ++row)
        u_local.row(row) *= scales[3 * row];
    if (!u_local.allFinite())
        throw std::domain_error("Hex Hermite example: non-finite reference coefficients");
    return u_local;
}

inline MeshEnergy make_hex_hermite_mesh_energy(const Vec& X_nodes) {
    const auto topology = hex_hermite_topology();
    // Each vertex stores [u, u_X, u_Y, u_Z, u_XY, u_XZ, u_YZ, u_XYZ], each a 3-vector.
    const VertexDofMap geometry_map(topology, 3), displacement_map(topology, 24);
    struct ReferenceData {
        InterpolatedField3D<HexQ1Basis> X;
        VolumetricElement3D<GaussLegendre3D<4>> integration;
        Eigen::DiagonalMatrix<double, Eigen::Dynamic> transform;
        explicit ReferenceData(Eigen::MatrixXd X_local)
            : X(X_local), integration(X), transform(eval_hex_hermite_dof_transform(X_local)) {}
    };
    std::vector<ReferenceData> reference;
    reference.reserve(geometry_map.num_elements());
    for (int e = 0; e < geometry_map.num_elements(); ++e)
        reference.emplace_back(geometry_map.gather(e, X_nodes));
    const auto reference_data = std::make_shared<const decltype(reference)>(std::move(reference));
    const IdentityField3D Fp_field(Eigen::Matrix3d::Identity().reshaped());
    const IdentityField3D theta_field(Eigen::Vector2d(2, 3));
    return MeshEnergy(
        displacement_map,
        [reference_data, Fp_field, theta_field](int e, const Eigen::MatrixXd& u_physical_local, Vec* gradient,
                                                Eigen::MatrixXd* hessian) {
            const auto& r = (*reference_data)[static_cast<std::size_t>(e)];
            const auto& scales = r.transform.diagonal();
            const auto u_local = hex_hermite_reference_coefficients(u_physical_local, scales);
            const InterpolatedField3D<HexHermiteBasis> displacement(u_local);
            const DeformationGradientField3D deformation(displacement, r.X);
            const VolumetricMaterialLocalEnergyF3D<NeoHookean> local_energy(deformation, Fp_field, theta_field);
            const double E = r.integration.eval(local_energy, gradient, hessian);
            // Pull reference-coefficient derivatives back to physical Hermite DOFs.
            if (gradient)
                gradient->array() *= scales.array();
            if (hessian) {
                hessian->array().colwise() *= scales.array();
                hessian->array().rowwise() *= scales.transpose().array();
            }
            if ((gradient && !gradient->allFinite()) || (hessian && !hessian->allFinite()))
                throw std::domain_error("Hex Hermite example: non-finite physical derivatives");
            return E;
        },
        [reference_data, Fp_field, theta_field](int e, const Eigen::MatrixXd& u_physical_local,
                                                const Eigen::MatrixXd& du_physical_local) {
            const auto& r = (*reference_data)[static_cast<std::size_t>(e)];
            const auto& scales = r.transform.diagonal();
            const auto u_local = hex_hermite_reference_coefficients(u_physical_local, scales);
            const auto du_local = hex_hermite_reference_coefficients(du_physical_local, scales);
            const InterpolatedField3D<HexHermiteBasis> displacement(u_local);
            const DeformationGradientField3D deformation(displacement, r.X);
            const VolumetricMaterialLocalEnergyF3D<NeoHookean> local_energy(deformation, Fp_field, theta_field);
            const Vec delta_coeffs = du_local.reshaped<Eigen::RowMajor>();
            return r.integration.max_feasible_step(local_energy, delta_coeffs);
        });
}

} // namespace litepgo::example
