#pragma once
#include <Eigen/Core>

namespace litepgo {

// Input matrices flatten by column, independently of their storage order.
Eigen::VectorXd gradient_wrt_dofs(const Eigen::Ref<const Eigen::MatrixXd>& J,
                                  const Eigen::Ref<const Eigen::MatrixXd>& gradient);

// Exact for fields affine in their local DOFs at fixed reference geometry.
Eigen::MatrixXd hessian_wrt_dofs(const Eigen::Ref<const Eigen::MatrixXd>& J,
                                 const Eigen::Ref<const Eigen::MatrixXd>& hessian);
} // namespace litepgo
