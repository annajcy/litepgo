#pragma once

#include "litepgo/dof/vertex_dof_map.hpp"
#include "litepgo/energy/mesh_energy.hpp"
#include "litepgo/fem/basis/hex_hermite_basis.hpp"
#include "litepgo/fem/basis/hex_q1_basis.hpp"
#include "litepgo/fem/field/deformation_gradient_field.hpp"
#include "litepgo/fem/field/identity_field.hpp"
#include "litepgo/fem/interpolation/hex_hermite_dof_transform.hpp"
#include "litepgo/fem/local/integrate_volume.hpp"
#include "litepgo/fem/local/material_local_energy.hpp"
#include "litepgo/fem/material/neo_hookean.hpp"
#include "litepgo/fem/quadrature/gauss_legendre_3d.hpp"

#include <memory>
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

inline Eigen::MatrixXd hex_hermite_reference_coefficients(const Eigen::MatrixXd& x_local, const Vec& scales) {
    if (x_local.rows() != 8 || x_local.cols() != 24 || scales.size() != 192)
        throw std::invalid_argument("Hex Hermite example requires 8x24 physical DOFs and 192 scales");
    Eigen::MatrixXd reference_coefficients = x_local.reshaped<Eigen::RowMajor>(64, 3);
    for (int row = 0; row < 64; ++row)
        reference_coefficients.row(row) *= scales[3 * row];
    return reference_coefficients;
}

inline MeshEnergy make_hex_hermite_mesh_energy(const Vec& X_nodes) {
    const auto topology = hex_hermite_topology();
    // Each vertex stores [u, u_X, u_Y, u_Z, u_XY, u_XZ, u_YZ, u_XYZ], each a 3-vector.
    // Global index: 24*vertex + 3*slot + component; shared vertices use the same indices.
    const VertexDofMap geometry_map(topology, 3), displacement_map(topology, 24);
    std::vector<Interpolation<HexQ1Basis>> rest_geometries;
    std::vector<Eigen::DiagonalMatrix<double, Eigen::Dynamic>> transforms;
    for (int e = 0; e < geometry_map.num_elements(); ++e) {
        rest_geometries.emplace_back(geometry_map.gather(e, X_nodes));
        transforms.push_back(hex_hermite_dof_transform(rest_geometries.back()));
    }
    const auto Fp = std::make_shared<IdentityField<Eigen::Matrix3d>>(Eigen::Matrix3d::Identity());
    const auto theta = std::make_shared<IdentityField<Eigen::VectorXd>>(Eigen::Vector2d(2, 3));
    return MeshEnergy(
        displacement_map,
        [rest_geometries, transforms, Fp, theta](int e, const Eigen::MatrixXd& x_local, Vec* dE_dx_local,
                                                 Eigen::MatrixXd* d2E_dx_local2) {
            const auto index = static_cast<std::size_t>(e);
            const auto& rest_geometry = rest_geometries[index];
            const auto& S = transforms[index];
            const auto& dof_scales = S.diagonal();
            const auto reference_coefficients = hex_hermite_reference_coefficients(x_local, dof_scales);
            const auto F = std::make_shared<DeformationGradientField<HexHermiteBasis>>(reference_coefficients);
            const MaterialLocalEnergyF<NeoHookean> local_energy({F, Fp, theta});
            // Reuse outputs for derivatives with respect to reference coefficients c until the pullback below.
            const double E_element =
                integrate_volume<GaussLegendre3D<4>>(rest_geometry, local_energy, dE_dx_local, d2E_dx_local2);
            // c = S*flatten(x_local): fixed rest geometry makes S constant, so g_local = S^T*g_c, H_local = S^T*H_c*S.
            if (dE_dx_local)
                dE_dx_local->array() *= dof_scales.array();
            if (d2E_dx_local2) {
                d2E_dx_local2->array().colwise() *= dof_scales.array();
                d2E_dx_local2->array().rowwise() *= dof_scales.transpose().array();
            }
            return E_element;
        },
        [](int, const auto&, const auto&) -> double {
            // shortcut: evaluation only; supply feasible step bounds before using a solver.
            throw std::logic_error("Feasible step bounds are not implemented in this example");
        });
}

} // namespace litepgo::example
