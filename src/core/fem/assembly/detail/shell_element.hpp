#pragma once

#include "litepgo/common/types.hpp"
#include "litepgo/fem/kinematics/triangle_mid_edge_kinematics.hpp"
#include "litepgo/fem/material/shell/shell_material_inputs.hpp"
#include <array>
#include <cmath>
#include <stdexcept>

namespace litepgo::fem::detail {

// Current forms of one face and their displacement derivatives, before the material is evaluated.
struct ShellElementState {
    ShellMaterialInputs inputs;
    Eigen::MatrixXd dab_dx;
    std::array<Eigen::MatrixXd, 6> d2ab_dx2;
    double weight = 0; // Original reference area.
};

inline ShellElementState shell_element_state(const TriangleMidEdgeRestData& rest,
                                             const Eigen::Ref<const Eigen::MatrixXd>& u_local, double thickness,
                                             const Eigen::Ref<const Eigen::VectorXd>& theta, int num_parameters,
                                             bool need_gradient, bool need_hessian) {
    if (rest.X_local.rows() < 3 || rest.X_local.rows() > 6 || rest.X_local.cols() != 3 ||
        u_local.rows() != rest.X_local.rows() || u_local.cols() != 3 || !u_local.allFinite() ||
        !rest.X_local.allFinite() || !std::isfinite(thickness) || thickness <= 0 || theta.size() != num_parameters ||
        !theta.allFinite() || !std::isfinite(rest.sqrt_det_a_bar) || rest.sqrt_det_a_bar <= 0)
        throw std::invalid_argument("shell_element_state: invalid stencil, area, thickness, or parameters");
    const Eigen::MatrixXd x_local = rest.X_local + u_local;
    if (!x_local.allFinite())
        throw std::domain_error("shell_element_state: non-finite current positions");
    ShellElementState state;
    state.weight = 0.5 * rest.sqrt_det_a_bar;
    const auto current = eval_triangle_mid_edge_kinematics(x_local, rest.opposite,
                                                           (need_gradient || need_hessian) ? &state.dab_dx : nullptr,
                                                           need_hessian ? &state.d2ab_dx2 : nullptr);
    state.inputs = {current.a, current.b, rest.a_bar, rest.b_bar, thickness, theta};
    return state;
}

// Chains material values at the current forms to the face; returns the weighted energy.
// Outputs are pre-sized views and are overwritten, not accumulated. dW_dab is read for any
// derivative output and d2W_dab2 only for the Hessian.
inline double finish_shell_element(const ShellElementState& state, double density,
                                   const Eigen::Ref<const Eigen::VectorXd>& dW_dab,
                                   const Eigen::Ref<const Eigen::MatrixXd>& d2W_dab2, StridedVecRef* gradient,
                                   StridedMatRef* hessian) {
    const auto n = state.dab_dx.cols();
    if ((gradient && gradient->size() != n) || (hessian && (hessian->rows() != n || hessian->cols() != n)))
        throw std::invalid_argument("finish_shell_element: invalid output dimensions");
    const double weight = state.weight;
    const double W = weight * density;
    if (!std::isfinite(W))
        throw std::domain_error("finish_shell_element: non-finite density");
    if (gradient)
        gradient->setZero();
    if (hessian)
        hessian->setZero();
    if (gradient || hessian) {
        if (dW_dab.size() != 6)
            throw std::logic_error("finish_shell_element: invalid material gradient");
        if (!dW_dab.allFinite())
            throw std::domain_error("finish_shell_element: non-finite material gradient");
        if (gradient) {
            gradient->noalias() += weight * (state.dab_dx.transpose() * dW_dab);
            if (!gradient->allFinite())
                throw std::domain_error("finish_shell_element: non-finite gradient");
        }
        if (hessian) {
            if (d2W_dab2.rows() != 6 || d2W_dab2.cols() != 6)
                throw std::logic_error("finish_shell_element: invalid material Hessian");
            if (!d2W_dab2.allFinite())
                throw std::domain_error("finish_shell_element: non-finite material Hessian");
            hessian->noalias() += weight * (state.dab_dx.transpose() * d2W_dab2 * state.dab_dx);
            for (std::size_t i = 0; i < state.d2ab_dx2.size(); ++i)
                *hessian += weight * dW_dab[static_cast<Eigen::Index>(i)] * state.d2ab_dx2[i];
            if (!hessian->allFinite())
                throw std::domain_error("finish_shell_element: non-finite Hessian");
        }
    }
    return W;
}

} // namespace litepgo::fem::detail
