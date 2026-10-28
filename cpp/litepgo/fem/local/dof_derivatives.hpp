#pragma once
#include <Eigen/Core>

namespace litepgo {

// dvalue_dcoeffs has output components in rows and local scalar DOFs in columns.
// Input matrices flatten by column, independently of their storage order.
// Non-finite numerical results throw std::domain_error; incompatible sizes throw invalid_argument.
Eigen::VectorXd eval_gradient_wrt_dofs(const Eigen::Ref<const Eigen::MatrixXd>& dvalue_dcoeffs,
                                       const Eigen::Ref<const Eigen::MatrixXd>& gradient);

// Exact for fields affine in their local DOFs at fixed reference geometry.
Eigen::MatrixXd eval_hessian_wrt_dofs(const Eigen::Ref<const Eigen::MatrixXd>& dvalue_dcoeffs,
                                      const Eigen::Ref<const Eigen::MatrixXd>& hessian);

// Add weighted contractions into initialized outputs, which must not alias
// inputs. In particular, the Hessian product accumulates without an n-by-n temporary.
void add_gradient_wrt_dofs(const Eigen::Ref<const Eigen::MatrixXd>& dvalue_dcoeffs,
                           const Eigen::Ref<const Eigen::MatrixXd>& gradient, double weight,
                           Eigen::Ref<Eigen::VectorXd> output);
void add_hessian_wrt_dofs(const Eigen::Ref<const Eigen::MatrixXd>& dvalue_dcoeffs,
                          const Eigen::Ref<const Eigen::MatrixXd>& hessian, double weight,
                          Eigen::Ref<Eigen::MatrixXd> output);
} // namespace litepgo
