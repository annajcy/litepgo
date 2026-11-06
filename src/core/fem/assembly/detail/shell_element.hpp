#pragma once

#include "litepgo/common/types.hpp"
#include "litepgo/fem/kinematics/triangle_mid_edge_kinematics.hpp"
#include "litepgo/fem/material/shell/shell_material_inputs.hpp"
#include <array>
#include <cmath>
#include <stdexcept>

namespace litepgo::fem::detail {

// Constant density on the reference triangle; apply its physical area exactly once.
// Outputs are pre-sized views and are overwritten, not accumulated.
template <class Material>
double eval_shell_element(const TriangleMidEdgeRestData& rest, const Eigen::Ref<const Eigen::MatrixXd>& u_local,
                          double thickness, const Eigen::Ref<const Eigen::VectorXd>& theta,
                          StridedVecRef* gradient = nullptr, StridedMatRef* hessian = nullptr) {
    const auto n = u_local.size();
    if (rest.X_local.rows() < 3 || rest.X_local.rows() > 6 || rest.X_local.cols() != 3 ||
        u_local.rows() != rest.X_local.rows() || u_local.cols() != 3 || !u_local.allFinite() ||
        !rest.X_local.allFinite() || !std::isfinite(thickness) || thickness <= 0 ||
        theta.size() != Material::num_parameters || !theta.allFinite() || !std::isfinite(rest.sqrt_det_a_bar) ||
        rest.sqrt_det_a_bar <= 0)
        throw std::invalid_argument("eval_shell_element: invalid stencil, area, thickness, or parameters");
    if ((gradient && gradient->size() != n) || (hessian && (hessian->rows() != n || hessian->cols() != n)))
        throw std::invalid_argument("eval_shell_element: invalid output dimensions");
    const Eigen::MatrixXd x_local = rest.X_local + u_local;
    if (!x_local.allFinite())
        throw std::domain_error("eval_shell_element: non-finite current positions");
    if (gradient)
        gradient->setZero();
    if (hessian)
        hessian->setZero();
    const double weight = 0.5 * rest.sqrt_det_a_bar;
    Eigen::MatrixXd dab_dx;
    std::array<Eigen::MatrixXd, 6> d2ab_dx2;
    const auto current = eval_triangle_mid_edge_kinematics(
        x_local, rest.opposite, (gradient || hessian) ? &dab_dx : nullptr, hessian ? &d2ab_dx2 : nullptr);
    const ShellMaterialInputs inputs{current.a, current.b, rest.a_bar, rest.b_bar, thickness, theta};
    const double W = weight * Material::eval_W(inputs);
    if (!std::isfinite(W))
        throw std::domain_error("eval_shell_element: non-finite density");
    if (gradient || hessian) {
        const Eigen::VectorXd dW_dab = Material::eval_dW_dab(inputs);
        if (dW_dab.size() != 6)
            throw std::logic_error("eval_shell_element: invalid material gradient");
        if (!dW_dab.allFinite())
            throw std::domain_error("eval_shell_element: non-finite material gradient");
        if (gradient) {
            gradient->noalias() += weight * (dab_dx.transpose() * dW_dab);
            if (!gradient->allFinite())
                throw std::domain_error("eval_shell_element: non-finite gradient");
        }
        if (hessian) {
            const Eigen::MatrixXd d2W_dab2 = Material::eval_d2W_dab2(inputs);
            if (d2W_dab2.rows() != 6 || d2W_dab2.cols() != 6)
                throw std::logic_error("eval_shell_element: invalid material Hessian");
            if (!d2W_dab2.allFinite())
                throw std::domain_error("eval_shell_element: non-finite material Hessian");
            hessian->noalias() += weight * (dab_dx.transpose() * d2W_dab2 * dab_dx);
            for (std::size_t i = 0; i < d2ab_dx2.size(); ++i)
                *hessian += weight * dW_dab[static_cast<Eigen::Index>(i)] * d2ab_dx2[i];
            if (!hessian->allFinite())
                throw std::domain_error("eval_shell_element: non-finite Hessian");
        }
    }
    return W;
}

} // namespace litepgo::fem::detail
