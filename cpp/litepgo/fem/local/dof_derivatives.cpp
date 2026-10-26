#include "litepgo/fem/local/dof_derivatives.hpp"

#include <cmath>
#include <stdexcept>

namespace litepgo {

Eigen::VectorXd gradient_wrt_dofs(const Eigen::Ref<const Eigen::MatrixXd>& dvalue_dx_local,
                                  const Eigen::Ref<const Eigen::MatrixXd>& gradient) {
    if (dvalue_dx_local.rows() != gradient.size() || dvalue_dx_local.rows() == 0 || dvalue_dx_local.cols() == 0)
        throw std::invalid_argument("gradient_wrt_dofs: incompatible Jacobian and gradient");
    const Eigen::MatrixXd packed = gradient;
    return dvalue_dx_local.transpose() * Eigen::Map<const Eigen::VectorXd>(packed.data(), packed.size());
}

Eigen::MatrixXd hessian_wrt_dofs(const Eigen::Ref<const Eigen::MatrixXd>& dvalue_dx_local,
                                 const Eigen::Ref<const Eigen::MatrixXd>& hessian) {
    if (dvalue_dx_local.rows() != hessian.rows() || dvalue_dx_local.rows() != hessian.cols() ||
        dvalue_dx_local.rows() == 0 || dvalue_dx_local.cols() == 0)
        throw std::invalid_argument("hessian_wrt_dofs: incompatible Jacobian and Hessian");
    return dvalue_dx_local.transpose() * hessian * dvalue_dx_local;
}

void add_gradient_wrt_dofs(const Eigen::Ref<const Eigen::MatrixXd>& dvalue_dx_local,
                           const Eigen::Ref<const Eigen::MatrixXd>& gradient, double weight,
                           Eigen::Ref<Eigen::VectorXd> output) {
    if (dvalue_dx_local.rows() != gradient.size() || dvalue_dx_local.rows() == 0 || dvalue_dx_local.cols() == 0 ||
        output.size() != dvalue_dx_local.cols() || !std::isfinite(weight))
        throw std::invalid_argument("add_gradient_wrt_dofs: incompatible dimensions or non-finite weight");
    output.noalias() += weight * (dvalue_dx_local.transpose() * gradient.reshaped());
}

void add_hessian_wrt_dofs(const Eigen::Ref<const Eigen::MatrixXd>& dvalue_dx_local,
                          const Eigen::Ref<const Eigen::MatrixXd>& hessian, double weight,
                          Eigen::Ref<Eigen::MatrixXd> output) {
    if (dvalue_dx_local.rows() != hessian.rows() || dvalue_dx_local.rows() != hessian.cols() ||
        dvalue_dx_local.rows() == 0 || dvalue_dx_local.cols() == 0 || output.rows() != dvalue_dx_local.cols() ||
        output.cols() != dvalue_dx_local.cols() || !std::isfinite(weight))
        throw std::invalid_argument("add_hessian_wrt_dofs: incompatible dimensions or non-finite weight");
    output.noalias() += weight * (dvalue_dx_local.transpose() * hessian * dvalue_dx_local);
}

} // namespace litepgo
