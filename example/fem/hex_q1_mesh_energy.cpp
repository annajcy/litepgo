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
    // Two adjacent hexes of widths 1 and 2, sharing the face at X = 1.
    Eigen::Matrix<double, 12, 3> rest_nodes;
    rest_nodes << 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 3, 0, 0, 3, 1, 0, 3, 0, 1, 3,
        1, 1;
    const Vec rest_positions = rest_nodes.reshaped<Eigen::RowMajor>();

    const HexTopology topology(12, {{0, 1, 2, 3, 4, 5, 6, 7}, {1, 8, 9, 2, 5, 10, 11, 6}});
    const VertexDofMap rest_position_map(topology, 3);
    using XField = InterpolatedField3D<HexQ1Basis>;
    std::vector<XField> rest_position_fields;
    rest_position_fields.reserve(topology.num_cells());
    for (int e = 0; e < topology.num_cells(); ++e)
        rest_position_fields.emplace_back(rest_position_map.gather(e, rest_positions));
    const IdentityField3D Fp_field(Eigen::Matrix3d::Identity().reshaped());
    const IdentityField3D theta_field(Eigen::Vector2d(2, 3));
    auto elastic = std::make_shared<
        VolumetricFemEnergy<NeoHookean, HexQ1Basis, GaussLegendre3D<2>, XField, IdentityField3D, IdentityField3D>>(
        rest_position_map, std::move(rest_position_fields),
        std::vector<IdentityField3D>(topology.num_cells(), Fp_field),
        std::vector<IdentityField3D>(topology.num_cells(), theta_field));

    const Vec u = 0.1 * rest_positions;
    Vec gradient;
    SpMat hessian = elastic->hessian_pattern();
    const double value = elastic->eval(u, &gradient, &hessian);
    std::cout << "E=" << value << ", |g|=" << gradient.norm() << ", H nnz=" << hessian.nonZeros() << '\n';
}
