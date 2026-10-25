#pragma once
#include <Eigen/Core>

namespace litepgo {

// Input matrices flatten by column, independently of their storage order.
Eigen::VectorXd gradient_wrt_dofs(const Eigen::Ref<const Eigen::MatrixXd>& J,
                                  const Eigen::Ref<const Eigen::MatrixXd>& gradient);

// Exact for fields affine in their local DOFs at fixed reference geometry.
Eigen::MatrixXd hessian_wrt_dofs(const Eigen::Ref<const Eigen::MatrixXd>& J,
                                 const Eigen::Ref<const Eigen::MatrixXd>& hessian);

// Add weighted contractions into initialized outputs, which must not alias
// inputs. In particular, the Hessian product accumulates without an n-by-n temporary.
void add_gradient_wrt_dofs(const Eigen::Ref<const Eigen::MatrixXd>& J,
                           const Eigen::Ref<const Eigen::MatrixXd>& gradient, double weight,
                           Eigen::Ref<Eigen::VectorXd> output);
void add_hessian_wrt_dofs(const Eigen::Ref<const Eigen::MatrixXd>& J, const Eigen::Ref<const Eigen::MatrixXd>& hessian,
                          double weight, Eigen::Ref<Eigen::MatrixXd> output);
} // namespace litepgo
