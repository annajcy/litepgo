#pragma once

#include <Eigen/Core>

namespace litepgo {

using Mat3 = Eigen::Matrix3d;
using Mat9 = Eigen::Matrix<double, 9, 9>;

template <int NumParameters>
    requires(NumParameters >= 0)
struct MaterialTypes {
    static constexpr int num_parameters = NumParameters;
    using Parameters = Eigen::Matrix<double, NumParameters, 1>;
    using ParameterHessian = Eigen::Matrix<double, NumParameters, NumParameters>;
};

} // namespace litepgo
