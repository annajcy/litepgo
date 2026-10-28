#pragma once

#include <Eigen/Core>

namespace litepgo {

template <int NumBasisFunctions, int ParamDim> struct BasisSample {
    static_assert(NumBasisFunctions > 0 && ParamDim > 0);

    Eigen::Matrix<double, NumBasisFunctions, 1> N;
    Eigen::Matrix<double, NumBasisFunctions, ParamDim> dN_dxi;
    // Upper triangle: (11) in 1D, (11,12,22) in 2D, (11,12,13,22,23,33) in 3D.
    // Mixed derivatives are stored once, without scaling.
    Eigen::Matrix<double, NumBasisFunctions, ParamDim*(ParamDim + 1) / 2> d2N_dxi2;
};

} // namespace litepgo
