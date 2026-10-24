#pragma once

#include "litepgo/fem/basis.hpp"

#include <Eigen/Core>
#include <limits>

namespace litepgo {

// Coefficient rows follow basis order, columns are field components.
// The same fixed reference geometry and basis are used for all three fields.
template <Basis3D Basis, int NumParameters>
    requires(NumParameters >= 0 && Basis::num_basis_functions <= std::numeric_limits<int>::max() / 3 &&
             (NumParameters == 0 || Basis::num_basis_functions <= std::numeric_limits<int>::max() / NumParameters))
struct VolumeLocalFields {
    static constexpr int num_basis_functions = Basis::num_basis_functions;
    using VectorCoefficients = Eigen::Matrix<double, num_basis_functions, 3>;
    using ParameterCoefficients = Eigen::Matrix<double, num_basis_functions, NumParameters>;

    VectorCoefficients u;
    VectorCoefficients up;
    ParameterCoefficients theta;
};

} // namespace litepgo
