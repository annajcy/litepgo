#pragma once

#include "litepgo/fem/detail/shell_force_data.hpp"
#include "litepgo/fem/field/field.hpp"
#include "litepgo/fem/kinematics/triangle_mid_edge_kinematics.hpp"
#include "litepgo/fem/local/local_response.hpp"
#include "litepgo/fem/material/shell/shell_material_inputs.hpp"
#include <cmath>
#include <stdexcept>

namespace litepgo::fem {
namespace detail {
// A null parameter derivative means theta itself is the active coefficient block.
template <class Material>
void accumulate_shell_force(const ShellForceData& data, const Eigen::VectorXd& theta, const Eigen::MatrixXd* dtheta,
                            double weight, StridedVecRef value, StridedMatRef* J) {
    const int n = dtheta ? static_cast<int>(dtheta->cols()) : Material::num_parameters;
    if (!std::isfinite(weight) || theta.size() != Material::num_parameters || !theta.allFinite() ||
        value.size() != data.dab_du.cols() || (J && (J->rows() != value.size() || J->cols() != n)) ||
        (dtheta && (dtheta->rows() != Material::num_parameters || n <= 0 || !dtheta->allFinite())))
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
        if (dtheta)
            J->noalias() += weight * (data.dab_du.transpose() * mixed * *dtheta);
        else
            J->noalias() += weight * (data.dab_du.transpose() * mixed);
    }
    if (!value.allFinite() || (J && !J->allFinite()))
        throw std::domain_error("Shell force: non-finite accumulation");
}
} // namespace detail

// Positive displacement-gradient density and its derivative wrt theta coefficients.
// Owns fixed geometry/displacement/thickness; theta is borrowed and must outlive this object.
template <class Material> class ShellMaterialForceLocalResponse2D final : public LocalResponse2D {
  public:
    ShellMaterialForceLocalResponse2D(const TriangleMidEdgeRestData& rest, const Eigen::MatrixXd& u_local,
                                      double thickness, const Field2D& theta)
        : data_(rest, u_local, thickness), theta_(theta) {
        if (theta_.num_components() != Material::num_parameters || theta_.num_dofs() <= 0)
            throw std::invalid_argument("ShellMaterialForceLocalResponse2D: incompatible parameter field");
    }
    ShellMaterialForceLocalResponse2D(const TriangleMidEdgeRestData&, const Eigen::MatrixXd&, double,
                                      const Field2D&&) = delete;
    int num_inputs() const override {
        return theta_.num_dofs();
    }
    int num_outputs() const override {
        return static_cast<int>(data_.dab_du.cols());
    }
    bool has_jacobian() const noexcept override {
        return true;
    }
    Eigen::VectorXd eval_value(const Eigen::Vector2d& xi) const override {
        Eigen::VectorXd value = Eigen::VectorXd::Zero(num_outputs());
        accumulate_into(xi, 1, StridedVecRef(value));
        return value;
    }
    Eigen::MatrixXd eval_jacobian(const Eigen::Vector2d& xi) const override {
        Eigen::VectorXd value = Eigen::VectorXd::Zero(num_outputs());
        Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(num_outputs(), num_inputs());
        StridedMatRef J(jacobian);
        accumulate_into(xi, 1, StridedVecRef(value), &J);
        return jacobian;
    }
    void accumulate_into(const Eigen::Vector2d& xi, double weight, StridedVecRef value,
                         StridedMatRef* J = nullptr) const override {
        if (!xi.allFinite() || !std::isfinite(weight) || value.size() != num_outputs() ||
            (J && (J->rows() != num_outputs() || J->cols() != num_inputs())))
            throw std::invalid_argument("ShellMaterialForceLocalResponse2D: invalid coordinates, weight or outputs");
        const Eigen::VectorXd theta = theta_.eval_value(xi);
        Eigen::MatrixXd derivative;
        if (J) {
            derivative = theta_.eval_dvalue_dcoeffs(xi);
            if (derivative.cols() != num_inputs())
                throw std::invalid_argument("ShellMaterialForceLocalResponse2D: invalid parameter derivative");
        }
        detail::accumulate_shell_force<Material>(data_, theta, J ? &derivative : nullptr, weight, value, J);
    }

  private:
    detail::ShellForceData data_;
    const Field2D& theta_;
};
} // namespace litepgo::fem
