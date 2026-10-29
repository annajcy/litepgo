#include "litepgo/fem/local/dof_derivatives.hpp"

#include <cmath>
#include <stdexcept>

namespace litepgo::fem {

Eigen::VectorXd eval_gradient_wrt_dofs(const Eigen::Ref<const Eigen::MatrixXd>& dvalue_dcoeffs,
                                       const Eigen::Ref<const Eigen::MatrixXd>& gradient) {
    if (dvalue_dcoeffs.rows() != gradient.size() || dvalue_dcoeffs.rows() == 0 || dvalue_dcoeffs.cols() == 0)
        throw std::invalid_argument("eval_gradient_wrt_dofs: incompatible Jacobian and gradient");
    const Eigen::MatrixXd packed = gradient;
    Eigen::VectorXd result =
        dvalue_dcoeffs.transpose() * Eigen::Map<const Eigen::VectorXd>(packed.data(), packed.size());
    if (!result.allFinite())
        throw std::domain_error("eval_gradient_wrt_dofs: non-finite pullback");
    return result;
}

Eigen::MatrixXd eval_hessian_wrt_dofs(const Eigen::Ref<const Eigen::MatrixXd>& dvalue_dcoeffs,
                                      const Eigen::Ref<const Eigen::MatrixXd>& hessian) {
    if (dvalue_dcoeffs.rows() != hessian.rows() || dvalue_dcoeffs.rows() != hessian.cols() ||
        dvalue_dcoeffs.rows() == 0 || dvalue_dcoeffs.cols() == 0)
        throw std::invalid_argument("eval_hessian_wrt_dofs: incompatible Jacobian and Hessian");
    Eigen::MatrixXd result = dvalue_dcoeffs.transpose() * hessian * dvalue_dcoeffs;
    if (!result.allFinite())
        throw std::domain_error("eval_hessian_wrt_dofs: non-finite pullback");
    return result;
}

void add_gradient_wrt_dofs(const Eigen::Ref<const Eigen::MatrixXd>& dvalue_dcoeffs,
                           const Eigen::Ref<const Eigen::MatrixXd>& gradient, double weight,
                           Eigen::Ref<Eigen::VectorXd> output, bool check_finite) {
    if (dvalue_dcoeffs.rows() != gradient.size() || dvalue_dcoeffs.rows() == 0 || dvalue_dcoeffs.cols() == 0 ||
        output.size() != dvalue_dcoeffs.cols() || !std::isfinite(weight))
        throw std::invalid_argument("add_gradient_wrt_dofs: incompatible dimensions or non-finite weight");
    output.noalias() += weight * (dvalue_dcoeffs.transpose() * gradient.reshaped());
    if (check_finite && !output.allFinite())
        throw std::domain_error("add_gradient_wrt_dofs: non-finite accumulation");
}

void add_hessian_wrt_dofs(const Eigen::Ref<const Eigen::MatrixXd>& dvalue_dcoeffs,
                          const Eigen::Ref<const Eigen::MatrixXd>& hessian, double weight,
                          Eigen::Ref<Eigen::MatrixXd> output, bool check_finite) {
    if (dvalue_dcoeffs.rows() != hessian.rows() || dvalue_dcoeffs.rows() != hessian.cols() ||
        dvalue_dcoeffs.rows() == 0 || dvalue_dcoeffs.cols() == 0 || output.rows() != dvalue_dcoeffs.cols() ||
        output.cols() != dvalue_dcoeffs.cols() || !std::isfinite(weight))
        throw std::invalid_argument("add_hessian_wrt_dofs: incompatible dimensions or non-finite weight");
    output.noalias() += weight * (dvalue_dcoeffs.transpose() * hessian * dvalue_dcoeffs);
    if (check_finite && !output.allFinite())
        throw std::domain_error("add_hessian_wrt_dofs: non-finite accumulation");
}

} // namespace litepgo::fem
