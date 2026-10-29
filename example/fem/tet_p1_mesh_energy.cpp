#include "litepgo/fem/field/identity_field.hpp"
#include "litepgo/fem/material/volumetric/neo_hookean.hpp"
#include "litepgo/fem/quadrature/3d/tet_centroid_quadrature.hpp"
#include "litepgo/fem/volumetric_fem_energy.hpp"

#include <iostream>
#include <memory>
#include <utility>
#include <vector>

int main() {
    using namespace litepgo;
    // Two positively oriented tets share a face; their volumes are 1/6 and 1/3.
    Eigen::Matrix<double, 5, 3> rest_nodes;
    rest_nodes << 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0.3, 0.2, -2;
    const Vec rest_positions = rest_nodes.reshaped<Eigen::RowMajor>();

    const TetTopology topology(5, {{0, 1, 2, 3}, {0, 2, 1, 4}});
    const VertexDofMap rest_position_map(topology, 3);
    using XField = InterpolatedField3D<TetP1Basis>;
    std::vector<XField> rest_position_fields;
    rest_position_fields.reserve(topology.num_cells());
    for (int e = 0; e < topology.num_cells(); ++e)
        rest_position_fields.emplace_back(rest_position_map.gather(e, rest_positions));
    const IdentityField3D Fp_field(Eigen::Matrix3d::Identity().reshaped());
    const IdentityField3D theta_field(Eigen::Vector2d(2, 3));
    auto elastic = std::make_shared<
        VolumetricFemEnergy<NeoHookean, TetP1Basis, TetCentroidQuadrature, XField, IdentityField3D, IdentityField3D>>(
        rest_position_map, std::move(rest_position_fields),
        std::vector<IdentityField3D>(topology.num_cells(), Fp_field),
        std::vector<IdentityField3D>(topology.num_cells(), theta_field));

    const Vec u = 0.1 * rest_positions;
    Vec gradient;
    SpMat hessian = elastic->hessian_pattern();
    const double value = elastic->eval(u, &gradient, &hessian);
    std::cout << "E=" << value << ", |g|=" << gradient.norm() << ", H nnz=" << hessian.nonZeros() << '\n';
}
