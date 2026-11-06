#pragma once

#include "litepgo/common/types.hpp"
#include "litepgo/fem/detail/shell_force_data.hpp"
#include "litepgo/fem/kinematics/triangle_mid_edge_kinematics.hpp"
#include "litepgo/fem/material/shell/shell_material_inputs.hpp"
#include <cmath>
#include <stdexcept>

namespace litepgo::fem {
namespace detail {
// The material parameter vector is the active input.
template <class Material>
void accumulate_shell_force(const ShellForceData& data, const Eigen::VectorXd& theta, double weight,
                            StridedVecRef value, StridedMatRef* J) {
    constexpr int n = Material::num_parameters;
    if (!std::isfinite(weight) || theta.size() != Material::num_parameters || !theta.allFinite() ||
        value.size() != data.dab_du.cols() || (J && (J->rows() != value.size() || J->cols() != n)))
        throw std::invalid_argument("Shell force: invalid parameters, derivative, weight or output dimensions");
    auto inputs = data.inputs;
    inputs.theta = theta;
    const Eigen::VectorXd stress = Material::eval_dW_dab(inputs);
    if (stress.size() != 6)
        throw std::logic_error("Shell force: invalid material gradient dimensions");
    if (!stress.allFinite())
        throw std::domain_error("Shell force: non-finite material gradient");
    value.noalias() += weight * (data.dab_du.transpose() * stress);
    if (J) {
        const Eigen::MatrixXd mixed = Material::eval_d2W_dab_dtheta(inputs);
        if (mixed.rows() != 6 || mixed.cols() != Material::num_parameters)
            throw std::logic_error("Shell force: invalid material mixed derivative dimensions");
        if (!mixed.allFinite())
            throw std::domain_error("Shell force: non-finite material mixed derivative");
        J->noalias() += weight * (data.dab_du.transpose() * mixed);
    }
    if (!value.allFinite() || (J && !J->allFinite()))
        throw std::domain_error("Shell force: non-finite accumulation");
}
} // namespace detail
} // namespace litepgo::fem
