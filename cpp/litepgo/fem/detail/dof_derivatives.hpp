#pragma once
#include <Eigen/Core>
#include <stdexcept>

namespace litepgo::detail {

// Input matrices flatten by column, independently of their storage order.
inline Eigen::VectorXd gradient_wrt_dofs(const Eigen::Ref<const Eigen::MatrixXd>& J,
                                         const Eigen::Ref<const Eigen::MatrixXd>& gradient) {
    if (J.rows() != gradient.size() || J.rows() == 0 || J.cols() == 0)
        throw std::invalid_argument("gradient_wrt_dofs: incompatible Jacobian and gradient");
    const Eigen::MatrixXd packed = gradient;
    return J.transpose() * Eigen::Map<const Eigen::VectorXd>(packed.data(), packed.size());
}

// Exact for fields affine in their local DOFs at fixed reference geometry.
inline Eigen::MatrixXd hessian_wrt_dofs(const Eigen::Ref<const Eigen::MatrixXd>& J,
                                        const Eigen::Ref<const Eigen::MatrixXd>& hessian) {
    if (J.rows() != hessian.rows() || J.rows() != hessian.cols() || J.rows() == 0 || J.cols() == 0)
        throw std::invalid_argument("hessian_wrt_dofs: incompatible Jacobian and Hessian");
    return J.transpose() * hessian * J;
}
} // namespace litepgo::detail
