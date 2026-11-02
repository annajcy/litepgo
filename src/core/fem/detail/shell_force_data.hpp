#pragma once
#include "litepgo/fem/kinematics/triangle_mid_edge_kinematics.hpp"
#include "litepgo/fem/material/shell/shell_material_inputs.hpp"
#include <cmath>
#include <stdexcept>

namespace litepgo::fem::detail {
// Fixed numerical geometry shared by local evaluation and batched mesh response.
struct ShellForceData {
    ShellMaterialInputs inputs;
    Eigen::MatrixXd dab_du;
    ShellForceData(const TriangleMidEdgeRestData& rest, const Eigen::MatrixXd& u, double thickness) {
        if (rest.X_local.rows() < 3 || rest.X_local.rows() > 6 || rest.X_local.cols() != 3 ||
            !rest.X_local.allFinite() || u.rows() != rest.X_local.rows() || u.cols() != 3 || !u.allFinite() ||
            !std::isfinite(thickness) || thickness <= 0)
            throw std::invalid_argument("ShellForceData: invalid displacement or thickness");
        const Eigen::MatrixXd x = rest.X_local + u;
        if (!x.allFinite())
            throw std::domain_error("ShellForceData: non-finite current positions");
        const auto current = eval_triangle_mid_edge_kinematics(x, rest.opposite, &dab_du);
        inputs = {current.a, current.b, rest.a_bar, rest.b_bar, thickness, {}};
    }
};

} // namespace litepgo::fem::detail
