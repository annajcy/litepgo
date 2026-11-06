#pragma once
#include "litepgo/common/types.hpp"
#include "litepgo/fem/assembly/detail/shell_force_data.hpp"
#include "litepgo/fem/assembly/detail/shell_plastic_geometry.hpp"
#include <utility>

namespace litepgo::fem {
namespace detail {
struct ShellFpForceData {
    ShellForceData current;
    Eigen::MatrixXd reference, dy_dFp;
    std::array<int, 3> opposite;
    ShellFpForceData(const TriangleMidEdgeRestData& rest, const Eigen::MatrixXd& u, double thickness,
                     Eigen::VectorXd theta)
        : current(rest, u, thickness), opposite(rest.opposite) {
        if (!theta.allFinite())
            throw std::invalid_argument("Shell Fp force: non-finite material parameters");
        current.inputs.theta = std::move(theta);
        reference = rest.X_local.rowwise() - rest.X_local.row(0);
        if (!reference.allFinite())
            throw std::domain_error("Shell Fp force: non-finite reference offsets");
        dy_dFp = Eigen::MatrixXd::Zero(3 * reference.rows(), 9);
        for (Eigen::Index i = 0; i < reference.rows(); ++i)
            for (int k = 0; k < 3; ++k)
                for (int l = 0; l < 3; ++l)
                    dy_dFp(3 * i + k, k + 3 * l) = reference(i, l);
    }
};

// Natural forms for one active Fp and, for Jacobians, their derivative with respect to vec(Fp).
struct ShellFpNatural {
    ShellMaterialInputs inputs;
    Eigen::MatrixXd natural_derivative; // d(packed natural forms) / d(natural stencil)
};
inline ShellFpNatural shell_fp_natural(const ShellFpForceData& data, const Eigen::VectorXd& packed, bool jacobian) {
    const Eigen::Matrix3d Fp = packed.reshaped(3, 3);
    Eigen::MatrixXd natural_derivative;
    const auto natural =
        shell_plastic_geometry(data.reference, data.opposite, Fp, jacobian ? &natural_derivative : nullptr);
    ShellFpNatural result{data.current.inputs, {}};
    result.inputs.a_bar = natural.a;
    result.inputs.b_bar = natural.b;
    result.natural_derivative = std::move(natural_derivative);
    return result;
}

// Overwrites the vec(Fp) Jacobian from d vec(dW/dab) / d(packed natural forms) (6 x 6).
inline void shell_fp_force_jacobian(const ShellFpForceData& data, const ShellFpNatural& natural,
                                    const Eigen::Ref<const Eigen::MatrixXd>& mixed, StridedMatRef& J) {
    if (mixed.rows() != 6 || mixed.cols() != 6)
        throw std::logic_error("Shell Fp force: invalid natural-form mixed derivative dimensions");
    if (!mixed.allFinite())
        throw std::domain_error("Shell Fp force: non-finite natural-form mixed derivative");
    const Eigen::MatrixXd full = data.current.dab_du.transpose() * mixed * natural.natural_derivative * data.dy_dFp;
    J.noalias() = data.current.area * full;
    if (!J.allFinite())
        throw std::domain_error("Shell Fp force: non-finite Jacobian");
}

inline void check_shell_fp_input(const ShellFpForceData& data, const Eigen::VectorXd& packed, int num_parameters,
                                 Eigen::Index value_size, const StridedMatRef* J) {
    if (packed.size() != 9 || !packed.allFinite() || data.current.inputs.theta.size() != num_parameters ||
        value_size != data.current.dab_du.cols() || (J && (J->rows() != value_size || J->cols() != 9)))
        throw std::invalid_argument("Shell Fp force: invalid Fp, derivative, parameters or output dimensions");
}

// The nine column-major Fp entries are the active input.
template <class Material>
void eval_shell_fp_force(const ShellFpForceData& data, const Eigen::VectorXd& packed, StridedVecRef value,
                         StridedMatRef* J) {
    check_shell_fp_input(data, packed, Material::num_parameters, value.size(), J);
    const auto natural = shell_fp_natural(data, packed, J != nullptr);
    eval_shell_force_value<Material>(data.current, natural.inputs, value);
    if (J)
        shell_fp_force_jacobian(data, natural, Material::eval_d2W_dab_dabarbbar(natural.inputs), *J);
}
} // namespace detail
} // namespace litepgo::fem
