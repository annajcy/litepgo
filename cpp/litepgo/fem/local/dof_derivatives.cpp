#include "litepgo/fem/local/dof_derivatives.hpp"

#include <cmath>
#include <stdexcept>

namespace litepgo {

Eigen::VectorXd gradient_wrt_dofs(const Eigen::Ref<const Eigen::MatrixXd>& J,
                                  const Eigen::Ref<const Eigen::MatrixXd>& gradient) {
    if (J.rows() != gradient.size() || J.rows() == 0 || J.cols() == 0)
        throw std::invalid_argument("gradient_wrt_dofs: incompatible Jacobian and gradient");
    const Eigen::MatrixXd packed = gradient;
    return J.transpose() * Eigen::Map<const Eigen::VectorXd>(packed.data(), packed.size());
}

Eigen::MatrixXd hessian_wrt_dofs(const Eigen::Ref<const Eigen::MatrixXd>& J,
                                 const Eigen::Ref<const Eigen::MatrixXd>& hessian) {
    if (J.rows() != hessian.rows() || J.rows() != hessian.cols() || J.rows() == 0 || J.cols() == 0)
        throw std::invalid_argument("hessian_wrt_dofs: incompatible Jacobian and Hessian");
    return J.transpose() * hessian * J;
}

void add_gradient_wrt_dofs(const Eigen::Ref<const Eigen::MatrixXd>& J,
                           const Eigen::Ref<const Eigen::MatrixXd>& gradient, double weight,
                           Eigen::Ref<Eigen::VectorXd> output) {
    if (J.rows() != gradient.size() || J.rows() == 0 || J.cols() == 0 || output.size() != J.cols() ||
        !std::isfinite(weight))
        throw std::invalid_argument("add_gradient_wrt_dofs: incompatible dimensions or non-finite weight");
    output.noalias() += weight * (J.transpose() * gradient.reshaped());
}

void add_hessian_wrt_dofs(const Eigen::Ref<const Eigen::MatrixXd>& J, const Eigen::Ref<const Eigen::MatrixXd>& hessian,
                          double weight, Eigen::Ref<Eigen::MatrixXd> output) {
    if (J.rows() != hessian.rows() || J.rows() != hessian.cols() || J.rows() == 0 || J.cols() == 0 ||
        output.rows() != J.cols() || output.cols() != J.cols() || !std::isfinite(weight))
        throw std::invalid_argument("add_hessian_wrt_dofs: incompatible dimensions or non-finite weight");
    output.noalias() += weight * (J.transpose() * hessian * J);
}

} // namespace litepgo
