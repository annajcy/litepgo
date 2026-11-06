#pragma once
#include "litepgo/common/types.hpp"

namespace litepgo::fem {

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
                           const Eigen::Ref<const Eigen::MatrixXd>& gradient, double weight, StridedVecRef output,
                           bool check_finite = true);
// With check_finite=false, the caller must validate the completed accumulation.
void add_hessian_wrt_dofs(const Eigen::Ref<const Eigen::MatrixXd>& dvalue_dcoeffs,
                          const Eigen::Ref<const Eigen::MatrixXd>& hessian, double weight, StridedMatRef output,
                          bool check_finite = true);
} // namespace litepgo::fem
