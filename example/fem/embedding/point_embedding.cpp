#include "litepgo/dof/surface_kinematics.hpp"
#include "litepgo/fem/point_displacement_map.hpp"
#include <iostream>
#include <stdexcept>
using namespace litepgo;
template <class B, class G> void run(const char* name) {
    constexpr int nodes = G::num_basis_functions;
    constexpr int width = fem::BasisDofTransform<B>::dofs_per_vertex;
    Eigen::MatrixXd X(nodes, 3);
    typename mesh::MeshTopology<nodes>::Cell cell;
    for (int i = 0; i < nodes; ++i) {
        X.row(i) = G::reference_node(i).cwiseProduct(Eigen::Vector3d(2, 3, 4)).transpose();
        cell[i] = i;
    }
    std::vector<fem::InterpolatedField3D<G>> fields{fem::InterpolatedField3D<G>(X)};
    Eigen::MatrixXd coordinates(2, 3);
    coordinates << .1, .2, .3, .2, .1, .2;
    Eigen::MatrixXd points(2, 3);
    for (int i = 0; i < 2; ++i)
        points.row(i) = fields[0].eval_value(coordinates.row(i).transpose()).transpose();
    auto embedding = fem::ElementLocator<G>(fields).locate(points);
    fem::PointEmbedding known{Eigen::VectorXi::Zero(2), coordinates};
    dof::VertexDofMap map(mesh::MeshTopology<nodes>(nodes, {cell}), width);
    const auto W = fem::build_point_displacement_map<B>(embedding, map, fields);
    const auto exact = fem::build_point_displacement_map<B>(known, map, fields);
    const Vec u = Vec::LinSpaced(map.num_dofs(), -.2, .3);
    fem::BasisDofTransform<B> transform(fields[0]);
    fem::InterpolatedField3D<B> field(transform.to_coefficients(map.gather(0, u)));
    dof::SurfaceKinematics kinematics(points.reshaped<Eigen::RowMajor>(), W);
    const Vec displacement = kinematics.displacements(u);
    double error = 0;
    for (int i = 0; i < 2; ++i)
        error =
            std::max(error, (displacement.segment<3>(3 * i) - field.eval_value(coordinates.row(i).transpose())).norm());
    if (error > 1e-10 || !(Vec(W * u)).isApprox(exact * u, 1e-10))
        throw std::runtime_error("Embedding interpolation mismatch");
    std::cout << name << ": points=2, verified=1\n";
}
int main() {
    run<fem::TetP1Basis, fem::TetP1Basis>("TetP1");
    run<fem::HexQ1Basis, fem::HexQ1Basis>("HexQ1");
    run<fem::HexHermiteBasis, fem::HexQ1Basis>("HexHermite");
}
