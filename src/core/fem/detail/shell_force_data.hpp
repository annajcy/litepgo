#pragma once
#include "litepgo/common/types.hpp"
#include "litepgo/fem/kinematics/triangle_mid_edge_kinematics.hpp"
#include "litepgo/fem/material/shell/shell_material_inputs.hpp"
#include <cmath>
#include <stdexcept>

namespace litepgo::fem::detail {
// Fixed pose and original area for one face; shared by both parameter-force paths.
struct ShellForceData {
    ShellMaterialInputs inputs;
    Eigen::MatrixXd dab_du;
    double area;
    ShellForceData(const TriangleMidEdgeRestData& rest, const Eigen::MatrixXd& u, double thickness) {
        if (rest.X_local.rows() < 3 || rest.X_local.rows() > 6 || rest.X_local.cols() != 3 ||
            !rest.X_local.allFinite() || u.rows() != rest.X_local.rows() || u.cols() != 3 || !u.allFinite() ||
            !std::isfinite(thickness) || thickness <= 0 || !std::isfinite(rest.sqrt_det_a_bar) ||
            rest.sqrt_det_a_bar <= 0)
            throw std::invalid_argument("ShellForceData: invalid displacement, thickness or reference area");
        area = 0.5 * rest.sqrt_det_a_bar;
        const Eigen::MatrixXd x = rest.X_local + u;
        if (!x.allFinite())
            throw std::domain_error("ShellForceData: non-finite current positions");
        const auto current = eval_triangle_mid_edge_kinematics(x, rest.opposite, &dab_du);
        inputs = {current.a, current.b, rest.a_bar, rest.b_bar, thickness, {}};
    }
};

// Overwrite the positive displacement-gradient force, including original area once.
template <class Material>
void eval_shell_force_value(const ShellForceData& data, const ShellMaterialInputs& inputs, StridedVecRef value) {
    if (value.size() != data.dab_du.cols())
        throw std::invalid_argument("Shell force: invalid output dimensions");
    const Eigen::VectorXd stress = Material::eval_dW_dab(inputs);
    if (stress.size() != 6)
        throw std::logic_error("Shell force: invalid material gradient dimensions");
    if (!stress.allFinite())
        throw std::domain_error("Shell force: non-finite material gradient");
    value.noalias() = data.area * (data.dab_du.transpose() * stress);
    if (!value.allFinite())
        throw std::domain_error("Shell force: non-finite force");
}

} // namespace litepgo::fem::detail
