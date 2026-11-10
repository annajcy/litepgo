#include "litepgo/fem/response/sampling_response.hpp"
#include "litepgo/fem/assembly/volumetric_fem_energy.hpp"
#include "litepgo/fem/basis/3d/hex_q1_basis.hpp"
#include "litepgo/fem/material/volumetric/neo_hookean.hpp"
#include "litepgo/fem/material/volumetric/pointwise_volumetric_adapter.hpp"
#include "litepgo/fem/quadrature/3d/gauss_legendre_3d.hpp"
#include "litepgo/fem/response/deformation_gradient_sampling_response.hpp"
#include <iostream>
#include <memory>
#include <stdexcept>

int main() {
    using namespace litepgo;
    using B = fem::HexQ1Basis;
    Eigen::MatrixXd points(2, 3);
    points << .2, .3, .4, .6, .7, .8;
    Eigen::MatrixXd C(8, 2);
    for (int a = 0; a < 8; ++a) {
        const auto p = B::coefficient_info(a).point;
        C(a, 0) = p.x() * p.y();
        C(a, 1) = p.y() * p.z();
    }
    const Vec c = C.reshaped<Eigen::RowMajor>();
    auto values = fem::value_sampling_response<B>(points, 2);
    auto gradients = fem::gradient_sampling_response<B>(points, 2);
    auto hessians = fem::hessian_sampling_response<B>(points, 2);
    Vec expected_g(12), expected_h(24);
    expected_g << .3, 0, .2, .4, 0, .3, .7, 0, .6, .8, 0, .7;
    expected_h << 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0;
    if (!values.evaluate(c)->value().isApprox(Eigen::Vector4d(.06, .12, .42, .56)) ||
        !gradients.evaluate(c)->value().isApprox(expected_g) || !hessians.evaluate(c)->value().isApprox(expected_h))
        throw std::runtime_error("sampling mismatch");
    const auto A = gradients.evaluate(c)->jacobian();
    const Vec direction = Vec::Ones(c.size());
    const Vec weights = Vec::LinSpaced(gradients.num_outputs(), 0, gradients.num_outputs() - 1);
    if (!gradients.evaluate(c)->jvp(direction).isApprox(Vec(A * direction)) ||
        !gradients.evaluate(c)->vjp(weights).isApprox(Vec(A.transpose() * weights)) ||
        hessians.evaluate(c)->hessian(Vec::Ones(hessians.num_outputs())).nonZeros() != 0)
        throw std::runtime_error("coefficient derivative mismatch");
    energy::AffineResponse geometry(Eigen::Vector3d(2, 3, 4).asDiagonal().toDenseMatrix().sparseView());
    auto deformation = fem::deformation_gradient_sampling_response<B>(geometry, points);
    Eigen::MatrixXd U(8, 3);
    for (int a = 0; a < 8; ++a)
        U.row(a) = .1 * B::coefficient_info(a).point.transpose();
    const Vec u = U.reshaped<Eigen::RowMajor>();
    const Vec F = deformation.evaluate(u)->value();
    const Eigen::Matrix3d expected = Eigen::Vector3d(1.05, 1 + .1 / 3, 1.025).asDiagonal();
    for (int q = 0; q < points.rows(); ++q)
        if (!F.segment<9>(9 * q).isApprox(expected.reshaped()))
            throw std::runtime_error("physical deformation gradient mismatch");
    // The same sampling factory constructs the maps used by native FEM.
    using Q = fem::GaussLegendre3D<2>;
    using M = fem::NeoHookean;
    const fem::VolumetricElement3D<Q> element(geometry);
    auto sample = fem::deformation_gradient_sampling_response<B>(geometry, Q::points());
    const Eigen::MatrixXd J = sample.evaluate(u)->jacobian();
    const Vec samples = sample.evaluate(u)->value();
    Vec g = Vec::Zero(24);
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(24, 24);
    double E = 0;
    for (int q = 0; q < Q::num_points; ++q) {
        const fem::VolumetricMaterialInputs inputs{samples.segment<9>(9 * q).reshaped(3, 3),
                                                   Eigen::Matrix3d::Identity(), Eigen::Vector2d(2, 3)};
        const auto Bq = J.middleRows(9 * q, 9);
        const double weight = element.point(q).w_dV;
        const Eigen::Matrix3d P = M::eval_dW_dF(inputs);
        E += weight * M::eval_W(inputs);
        g.noalias() += weight * Bq.transpose() * P.reshaped();
        H.noalias() += weight * Bq.transpose() * M::eval_d2W_dF2(inputs) * Bq;
    }
    const mesh::HexTopology topology(8, {{0, 1, 2, 3, 4, 5, 6, 7}});
    const dof::VertexDofMap map(topology, 3);
    auto fixed_geometry =
        std::make_shared<energy::AffineResponse>(Eigen::Vector3d(2, 3, 4).asDiagonal().toDenseMatrix().sparseView());
    auto fixed_Fp =
        std::make_shared<energy::AffineResponse>(SpMat(9, 3), Eigen::Matrix3d::Identity().eval().reshaped());
    auto fixed_theta = std::make_shared<energy::AffineResponse>(SpMat(2, 3), Eigen::Vector2d(2, 3));
    fem::VolumetricFemEnergy<B, Q> native(std::make_shared<litepgo::fem::PointwiseVolumetricAdapter<M>>(), map,
                                          {fixed_geometry}, {fixed_Fp}, {fixed_theta});
    if (std::abs(E - native.evaluate(u)->value()) > 1e-13 || !g.isApprox(native.evaluate(u)->gradient(), 1e-12) ||
        !H.isApprox(Eigen::MatrixXd(native.evaluate(u)->hessian()), 1e-12))
        throw std::runtime_error("explicit integration differs from native FEM");
    std::cout << "coefficient value, gradient and Hessian sampling: verified\n";
}
