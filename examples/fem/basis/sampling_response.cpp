#include "litepgo/fem/response/sampling_response.hpp"
#include "litepgo/fem/basis/3d/hex_q1_basis.hpp"
#include <iostream>
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
    if (!values.eval_value(c).isApprox(Eigen::Vector4d(.06, .12, .42, .56)) ||
        !gradients.eval_value(c).isApprox(expected_g) || !hessians.eval_value(c).isApprox(expected_h))
        throw std::runtime_error("sampling mismatch");
    const auto A = gradients.eval_jacobian(c);
    const Vec direction = Vec::Ones(c.size());
    const Vec weights = Vec::LinSpaced(gradients.num_outputs(), 0, gradients.num_outputs() - 1);
    if (!gradients.eval_jvp(c, direction).isApprox(Vec(A * direction)) ||
        !gradients.eval_vjp(c, weights).isApprox(Vec(A.transpose() * weights)) ||
        hessians.eval_hessian(c, Vec::Ones(hessians.num_outputs())).nonZeros() != 0)
        throw std::runtime_error("coefficient derivative mismatch");
    std::cout << "coefficient value, gradient and Hessian sampling: verified\n";
}
