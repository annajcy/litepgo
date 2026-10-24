#pragma once

#include <Eigen/Core>
#include <concepts>

namespace litepgo::detail {

// J = d vec(input) / d dofs. Input matrices are vectorized by column,
// independently of their storage order. Results own their data.
template <class Jacobian, class Gradient>
    requires std::same_as<typename Jacobian::Scalar, double> && std::same_as<typename Gradient::Scalar, double> &&
             (Jacobian::RowsAtCompileTime > 0 && Jacobian::ColsAtCompileTime > 0 && Gradient::RowsAtCompileTime > 0 &&
              Gradient::ColsAtCompileTime > 0 && Jacobian::RowsAtCompileTime == Gradient::SizeAtCompileTime)
Eigen::Matrix<double, Jacobian::ColsAtCompileTime, 1> gradient_wrt_dofs(const Eigen::MatrixBase<Jacobian>& J,
                                                                        const Eigen::MatrixBase<Gradient>& gradient) {
    const Eigen::Matrix<double, Gradient::RowsAtCompileTime, Gradient::ColsAtCompileTime> packed = gradient;
    return J.transpose() * Eigen::Map<const Eigen::Matrix<double, Gradient::SizeAtCompileTime, 1>>(packed.data());
}

// Exact only when the input is affine in the DOFs; no projection is applied.
template <class Jacobian, class Hessian>
    requires std::same_as<typename Jacobian::Scalar, double> && std::same_as<typename Hessian::Scalar, double> &&
             (Jacobian::RowsAtCompileTime > 0 && Jacobian::ColsAtCompileTime > 0 &&
              Hessian::RowsAtCompileTime == Jacobian::RowsAtCompileTime &&
              Hessian::ColsAtCompileTime == Jacobian::RowsAtCompileTime)
Eigen::Matrix<double, Jacobian::ColsAtCompileTime, Jacobian::ColsAtCompileTime> hessian_wrt_dofs(
    const Eigen::MatrixBase<Jacobian>& J, const Eigen::MatrixBase<Hessian>& hessian) {
    return J.transpose() * hessian * J;
}

} // namespace litepgo::detail
