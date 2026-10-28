#include "litepgo/fem/material/shell/koiter_stvk.hpp"
#include "litepgo/energy/energy.hpp"

#include <Eigen/Cholesky>
#include <array>
#include <cmath>
#include <stdexcept>

namespace litepgo {
namespace {

Eigen::Matrix<double, 6, 1> eval_delta_ab(const ShellMaterialInputs& inputs) {
    const Eigen::Matrix2d da = inputs.a - inputs.a_bar, db = inputs.b - inputs.b_bar;
    Eigen::Matrix<double, 6, 1> delta_ab;
    delta_ab << da(0, 0), da(0, 1), da(1, 1), db(0, 0), db(0, 1), db(1, 1);
    if (!delta_ab.allFinite())
        throw InfeasibleState("KoiterStVK: non-finite form differences");
    return delta_ab;
}

Eigen::Matrix<double, 6, 6> eval_stiffness(const ShellMaterialInputs& inputs) {
    for (const auto* form : {&inputs.a, &inputs.b, &inputs.a_bar, &inputs.b_bar})
        if (!form->allFinite() || !form->isApprox(form->transpose(), 1e-12))
            throw std::invalid_argument("KoiterStVK requires finite symmetric fundamental forms");
    if (!std::isfinite(inputs.thickness) || inputs.thickness <= 0 || inputs.theta.size() != 2 ||
        !inputs.theta.allFinite() || inputs.theta[1] <= 0 || inputs.theta.sum() <= 0)
        throw std::invalid_argument("KoiterStVK requires t > 0, mu > 0, and lambda_ps+mu > 0");
    const Eigen::LLT<Eigen::Matrix2d> current_metric(inputs.a);
    const Eigen::LLT<Eigen::Matrix2d> reference_metric(inputs.a_bar);
    if (current_metric.info() != Eigen::Success || reference_metric.info() != Eigen::Success)
        throw InfeasibleState("KoiterStVK requires positive-definite current and reference metrics");
    const Eigen::Matrix2d inv_a_bar = reference_metric.solve(Eigen::Matrix2d::Identity());
    if (!inv_a_bar.allFinite())
        throw InfeasibleState("KoiterStVK: non-finite inverse reference metric");

    std::array<Eigen::Matrix2d, 3> E;
    E[0] << 1, 0, 0, 0;
    E[1] << 0, 1, 1, 0;
    E[2] << 0, 0, 0, 1;
    std::array<Eigen::Matrix2d, 3> G;
    for (std::size_t i = 0; i < E.size(); ++i)
        G[i] = inv_a_bar * E[i];
    Eigen::Matrix3d C;
    const double lambda_ps = inputs.theta[0], mu = inputs.theta[1];
    for (int i = 0; i < 3; ++i)
        for (int j = i; j < 3; ++j) {
            const auto& A = G[static_cast<std::size_t>(i)];
            const auto& B = G[static_cast<std::size_t>(j)];
            C(i, j) = C(j, i) = lambda_ps * A.trace() * B.trace() + 2 * mu * (A * B).trace();
        }
    Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
    H.topLeftCorner<3, 3>() = (inputs.thickness / 4) * C;
    H.bottomRightCorner<3, 3>() = ((inputs.thickness / 12) * inputs.thickness * inputs.thickness) * C;
    if (!H.allFinite())
        throw InfeasibleState("KoiterStVK: non-finite material stiffness");
    return H;
}

} // namespace

double KoiterStVK::eval_W(const ShellMaterialInputs& inputs) {
    const auto H = eval_stiffness(inputs);
    const auto delta_ab = eval_delta_ab(inputs);
    const double W = 0.5 * delta_ab.dot(H * delta_ab);
    if (!std::isfinite(W))
        throw InfeasibleState("KoiterStVK: non-finite density");
    return W;
}

Eigen::Matrix<double, 6, 1> KoiterStVK::eval_dW_dab(const ShellMaterialInputs& inputs) {
    const auto H = eval_stiffness(inputs);
    const Eigen::Matrix<double, 6, 1> gradient = H * eval_delta_ab(inputs);
    if (!gradient.allFinite())
        throw InfeasibleState("KoiterStVK: non-finite material gradient");
    return gradient;
}

Eigen::Matrix<double, 6, 6> KoiterStVK::eval_d2W_dab2(const ShellMaterialInputs& inputs) {
    const auto H = eval_stiffness(inputs);
    eval_delta_ab(inputs);
    return H;
}

} // namespace litepgo
