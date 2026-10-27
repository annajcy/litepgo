#pragma once

#include "litepgo/dof/vertex_dof_map.hpp"
#include "litepgo/energy/mesh_energy.hpp"
#include "litepgo/fem/basis/3d/hex_hermite_basis.hpp"
#include "litepgo/fem/basis/3d/hex_q1_basis.hpp"
#include "litepgo/fem/field/hex_hermite_dof_transform.hpp"
#include "litepgo/fem/field/identity_field.hpp"
#include "litepgo/fem/field/interpolated_field.hpp"
#include "litepgo/fem/kinematics/deformation_gradient.hpp"
#include "litepgo/fem/local/integrate_volume.hpp"
#include "litepgo/fem/local/material_local_energy.hpp"
#include "litepgo/fem/material/neo_hookean.hpp"
#include "litepgo/fem/quadrature/3d/gauss_legendre_3d.hpp"

#include <stdexcept>
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
    if (u_physical_local.rows() != 8 || u_physical_local.cols() != 24 || scales.size() != 192)
        throw std::invalid_argument("Hex Hermite example requires 8x24 physical DOFs and 192 scales");
    Eigen::MatrixXd u_local = u_physical_local.reshaped<Eigen::RowMajor>(64, 3);
    for (int row = 0; row < 64; ++row)
        u_local.row(row) *= scales[3 * row];
    return u_local;
}

inline MeshEnergy make_hex_hermite_mesh_energy(const Vec& X_nodes) {
    const auto topology = hex_hermite_topology();
    // Each vertex stores [u, u_X, u_Y, u_Z, u_XY, u_XZ, u_YZ, u_XYZ], each a 3-vector.
    // Global index: 24*vertex + 3*slot + component; shared vertices use the same indices.
    const VertexDofMap geometry_map(topology, 3), displacement_map(topology, 24);
    std::vector<InterpolatedField<HexQ1Basis>> X_fields;
    std::vector<Eigen::DiagonalMatrix<double, Eigen::Dynamic>> transforms;
    for (int e = 0; e < geometry_map.num_elements(); ++e) {
        X_fields.emplace_back(geometry_map.gather(e, X_nodes));
        transforms.push_back(hex_hermite_dof_transform(X_fields.back()));
    }
    const IdentityField<Eigen::Matrix3d> Fp_field(Eigen::Matrix3d::Identity());
    const IdentityField<Eigen::VectorXd> theta_field(Eigen::Vector2d(2, 3));
    return MeshEnergy(
        displacement_map,
        [X_fields, transforms, Fp_field, theta_field](int e, const Eigen::MatrixXd& x_local, Vec* dE_dx_local,
                                                      Eigen::MatrixXd* d2E_dx_local2) {
            const auto index = static_cast<std::size_t>(e);
            const auto& X_field = X_fields[index];
            const auto& S = transforms[index];
            const auto& dof_scales = S.diagonal();
            const auto& u_physical_local = x_local;
            const auto u_local = hex_hermite_reference_coefficients(u_physical_local, dof_scales);
            const auto u_field = InterpolatedField<HexHermiteBasis>(u_local);
            const auto deformation = DeformationGradient(u_field, X_field);
            const auto local_energy =
                make_material_local_energy<NeoHookean, MaterialVariable::F>(deformation, Fp_field, theta_field);
            Vec dE_du_local;
            Eigen::MatrixXd d2E_du_local2;
            const double E_element = integrate_volume<GaussLegendre3D<4>>(
                local_energy, dE_dx_local ? &dE_du_local : nullptr, d2E_dx_local2 ? &d2E_du_local2 : nullptr);
            // Pull reference-coefficient derivatives back to gathered physical Hermite DOFs.
            if (dE_dx_local)
                *dE_dx_local = dof_scales.array() * dE_du_local.array();
            if (d2E_dx_local2) {
                d2E_du_local2.array().colwise() *= dof_scales.array();
                d2E_du_local2.array().rowwise() *= dof_scales.transpose().array();
                *d2E_dx_local2 = std::move(d2E_du_local2);
            }
            return E_element;
        },
        [](int, const auto&, const auto&) -> double {
            // shortcut: evaluation only; supply feasible step bounds before using a solver.
            throw std::logic_error("Feasible step bounds are not implemented in this example");
        });
}

} // namespace litepgo::example
