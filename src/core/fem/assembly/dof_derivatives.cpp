#include "litepgo/fem/assembly/dof_derivatives.hpp"

#include <cmath>
#include <stdexcept>

namespace litepgo::fem {

Eigen::VectorXd eval_gradient_wrt_dofs(const Eigen::Ref<const Eigen::MatrixXd>& dvalue_dcoeffs,
                                       const Eigen::Ref<const Eigen::MatrixXd>& gradient) {
    Eigen::VectorXd result = Eigen::VectorXd::Zero(dvalue_dcoeffs.cols());
    add_gradient_wrt_dofs(dvalue_dcoeffs, gradient, 1, result);
    return result;
}

Eigen::MatrixXd eval_hessian_wrt_dofs(const Eigen::Ref<const Eigen::MatrixXd>& dvalue_dcoeffs,
                                      const Eigen::Ref<const Eigen::MatrixXd>& hessian) {
    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(dvalue_dcoeffs.cols(), dvalue_dcoeffs.cols());
    add_hessian_wrt_dofs(dvalue_dcoeffs, hessian, 1, result);
    return result;
}

void add_gradient_wrt_dofs(const Eigen::Ref<const Eigen::MatrixXd>& dvalue_dcoeffs,
                           const Eigen::Ref<const Eigen::MatrixXd>& gradient, double weight, StridedVecRef output,
                           bool check_finite) {
    if (dvalue_dcoeffs.rows() != gradient.size() || dvalue_dcoeffs.rows() == 0 || dvalue_dcoeffs.cols() == 0 ||
        output.size() != dvalue_dcoeffs.cols() || !std::isfinite(weight))
        throw std::invalid_argument("add_gradient_wrt_dofs: incompatible dimensions or non-finite weight");
    output.noalias() += weight * (dvalue_dcoeffs.transpose() * gradient.reshaped());
    if (check_finite && !output.allFinite())
        throw std::domain_error("add_gradient_wrt_dofs: non-finite accumulation");
}

void add_hessian_wrt_dofs(const Eigen::Ref<const Eigen::MatrixXd>& dvalue_dcoeffs,
                          const Eigen::Ref<const Eigen::MatrixXd>& hessian, double weight, StridedMatRef output,
                          bool check_finite) {
    if (dvalue_dcoeffs.rows() != hessian.rows() || dvalue_dcoeffs.rows() != hessian.cols() ||
        dvalue_dcoeffs.rows() == 0 || dvalue_dcoeffs.cols() == 0 || output.rows() != dvalue_dcoeffs.cols() ||
        output.cols() != dvalue_dcoeffs.cols() || !std::isfinite(weight))
        throw std::invalid_argument("add_hessian_wrt_dofs: incompatible dimensions or non-finite weight");
    // Expose contiguous rows so Eigen can use its vectorized matrix-product kernel.
    if (output.colStride() == 1) {
        using RowMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        Eigen::Map<RowMatrix, 0, Eigen::OuterStride<>> rows(output.data(), output.rows(), output.cols(),
                                                            Eigen::OuterStride<>(output.rowStride()));
        rows.noalias() += weight * (dvalue_dcoeffs.transpose() * hessian * dvalue_dcoeffs);
    } else {
        output.noalias() += weight * (dvalue_dcoeffs.transpose() * hessian * dvalue_dcoeffs);
    }
    if (check_finite && !output.allFinite())
        throw std::domain_error("add_hessian_wrt_dofs: non-finite accumulation");
}

} // namespace litepgo::fem
