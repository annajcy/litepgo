#pragma once
#include "litepgo/contact/detail/distance.hpp"
namespace litepgo::contact::detail {
using LocalGradient = Eigen::Matrix<double, 12, 1>;
using LocalHessian = Eigen::Matrix<double, 12, 12>;
// Selected-feature derivatives scattered into the four query vertices' order.
// A null Hessian requests first derivatives only.
void distance_derivatives(const Points& p, const Feature& feature, LocalGradient& gradient, LocalHessian* hessian);
void cross_squarednorm_derivatives(const Points& p, LocalGradient& gradient, LocalHessian* hessian);
} // namespace litepgo::contact::detail
