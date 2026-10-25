#pragma once

#include "litepgo/dof/vertex_dof_map.hpp"
#include "litepgo/energy/mesh_energy.hpp"
#include "litepgo/fem/basis/tet_p1_basis.hpp"
#include "litepgo/fem/field/deformation_gradient_field.hpp"
#include "litepgo/fem/field/identity_field.hpp"
#include "litepgo/fem/local/integrate_volume.hpp"
#include "litepgo/fem/local/material_local_energy.hpp"
#include "litepgo/fem/material/neo_hookean.hpp"
#include "litepgo/fem/quadrature/tet_centroid_quadrature.hpp"

#include <memory>
#include <stdexcept>
#include <vector>

namespace litepgo::example {

inline Vec tet_p1_rest_positions() {
    // Two positively oriented tets share a face; their volumes are 1/6 and 1/3.
    Eigen::Matrix<double, 5, 3> rest;
    rest << 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0.3, 0.2, -2;
    return rest.reshaped<Eigen::RowMajor>();
}

inline MeshEnergy make_tet_p1_mesh_energy(const Vec& X) {
    const VertexDofMap map(TetTopology(5, {{0, 1, 2, 3}, {0, 2, 1, 4}}), 3);
    std::vector<Interpolation<TetP1Basis>> geometries;
    for (int e = 0; e < map.num_elements(); ++e)
        geometries.emplace_back(map.gather(e, X));
    const auto Fp = std::make_shared<IdentityField<Eigen::Matrix3d>>(Eigen::Matrix3d::Identity());
    const auto theta = std::make_shared<IdentityField<Eigen::VectorXd>>(Eigen::Vector2d(2, 3));

    return MeshEnergy(
        map,
        [geometries, Fp, theta](int e, const Eigen::MatrixXd& local_u, Vec* g, Eigen::MatrixXd* h) {
            // The geometry and local displacement both belong to element e.
            const MaterialLocalEnergyF<NeoHookean> local(
                {std::make_shared<DeformationGradientField<TetP1Basis>>(local_u), Fp, theta});
            return integrate_volume<TetCentroidQuadrature>(geometries[static_cast<std::size_t>(e)], local, g, h);
        },
        [](int, const auto&, const auto&) -> double {
            // shortcut: evaluation only; supply feasible step bounds before using a solver.
            throw std::logic_error("Feasible step bounds are not implemented in this example");
        });
}

} // namespace litepgo::example
