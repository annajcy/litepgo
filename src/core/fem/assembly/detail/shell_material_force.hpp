#pragma once

#include "litepgo/common/types.hpp"
#include "litepgo/fem/assembly/detail/shell_force_data.hpp"
#include "litepgo/fem/kinematics/triangle_mid_edge_kinematics.hpp"
#include "litepgo/fem/material/shell/shell_material_inputs.hpp"
#include <cmath>
#include <stdexcept>

namespace litepgo::fem {
namespace detail {
// Overwrites the theta Jacobian from d vec(dW/dab) / d theta (6 x k).
inline void shell_material_force_jacobian(const ShellForceData& data, const Eigen::Ref<const Eigen::MatrixXd>& mixed,
                                          int num_parameters, StridedMatRef& J) {
    if (mixed.rows() != 6 || mixed.cols() != num_parameters)
        throw std::logic_error("Shell force: invalid material mixed derivative dimensions");
    if (!mixed.allFinite())
        throw std::domain_error("Shell force: non-finite material mixed derivative");
    J.noalias() = data.area * (data.dab_du.transpose() * mixed);
    if (!J.allFinite())
        throw std::domain_error("Shell force: non-finite Jacobian");
}

} // namespace detail
} // namespace litepgo::fem
