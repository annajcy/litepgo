#pragma once
#include "litepgo/fem/assembly/detail/shell_element.hpp"

namespace litepgo::fem::test {
// Single-face reference composition for derivative and global assembly comparisons.
template <class Material>
double evaluate_shell_element(const TriangleMidEdgeRestData& rest, const Eigen::Ref<const Eigen::MatrixXd>& u,
                              double thickness, const Eigen::Ref<const Eigen::VectorXd>& theta,
                              StridedVecRef* gradient = nullptr, StridedMatRef* hessian = nullptr) {
    const auto state = detail::shell_element_state(rest, u, thickness, theta, Material::num_parameters,
                                                   gradient != nullptr, hessian != nullptr);
    const double density = Material::eval_W(state.inputs);
    const Vec stress = (gradient || hessian) ? Vec(Material::eval_dW_dab(state.inputs)) : Vec();
    const Eigen::MatrixXd tangent =
        hessian ? Eigen::MatrixXd(Material::eval_d2W_dab2(state.inputs)) : Eigen::MatrixXd();
    return detail::finish_shell_element(state, density, stress, tangent, gradient, hessian);
}
} // namespace litepgo::fem::test
