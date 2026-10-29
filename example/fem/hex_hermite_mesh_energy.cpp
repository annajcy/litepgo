#include "litepgo/fem/field/identity_field.hpp"
#include "litepgo/fem/material/volumetric/neo_hookean.hpp"
#include "litepgo/fem/quadrature/3d/gauss_legendre_3d.hpp"
#include "litepgo/fem/volumetric_fem_energy.hpp"

#include <iostream>
#include <memory>
#include <utility>
#include <vector>

int main() {
    using namespace litepgo;
    // Two boxes of sizes (1,2,3) and (2,2,3) share the face at X = 1.
    Eigen::Matrix<double, 12, 3> rest_nodes;
    rest_nodes << 0, 0, 0, 1, 0, 0, 1, 2, 0, 0, 2, 0, 0, 0, 3, 1, 0, 3, 1, 2, 3, 0, 2, 3, 3, 0, 0, 3, 2, 0, 3, 0, 3, 3,
        2, 3;
    const Vec rest_positions = rest_nodes.reshaped<Eigen::RowMajor>();

    // Shared face at X = 1: vertices 1, 2, 5, 6 reuse all 24 physical DOFs per vertex.
    const auto topology = HexTopology(12, {{0, 1, 2, 3, 4, 5, 6, 7}, {1, 8, 9, 2, 5, 10, 11, 6}});
    const VertexDofMap rest_position_map(topology, 3), displacement_map(topology, 24);
    using XField = InterpolatedField3D<HexQ1Basis>;
    std::vector<XField> rest_position_fields;
    rest_position_fields.reserve(topology.num_cells());
    for (int e = 0; e < topology.num_cells(); ++e)
        rest_position_fields.emplace_back(rest_position_map.gather(e, rest_positions));
    const IdentityField3D Fp(Eigen::Matrix3d::Identity().reshaped());
    const IdentityField3D theta(Eigen::Vector2d(2, 3));
    auto elastic = std::make_shared<
        VolumetricFemEnergy<NeoHookean, HexHermiteBasis, GaussLegendre3D<4>, XField, IdentityField3D, IdentityField3D>>(
        displacement_map, std::move(rest_position_fields), std::vector<IdentityField3D>(topology.num_cells(), Fp),
        std::vector<IdentityField3D>(topology.num_cells(), theta));

    Vec u = Vec::Zero(elastic->num_dofs());
    for (int v = 0; v < topology.num_vertices(); ++v) {
        u.segment<3>(24 * v) = 0.1 * rest_positions.segment<3>(3 * v);
        for (int d = 0; d < 3; ++d)
            u.segment<3>(24 * v + 3 * (d + 1)) = 0.1 * Eigen::Vector3d::Unit(d);
    }
    Vec gradient;
    SpMat hessian = elastic->hessian_pattern();
    const double value = elastic->eval(u, &gradient, &hessian);
    std::cout << "E=" << value << ", |g|=" << gradient.norm() << ", H nnz=" << hessian.nonZeros() << '\n';
}
