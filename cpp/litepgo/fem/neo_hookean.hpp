#pragma once

#include "litepgo/fem/material_types.hpp"

namespace litepgo {

// W = det(Fp) [mu/2 (||Fe||^2 - 3) - mu log(Je) + lambda/2 log(Je)^2],
// Fe = F Fp^-1, Je = det(Fe). Density is per original reference volume.
class NeoHookean final {
  public:
    static constexpr int num_parameters = 2;
    using Parameters = MaterialTypes<num_parameters>::Parameters;

    // theta = [lambda, mu], finite lambda >= 0 and mu > 0 (invalid_argument
    // otherwise). Requires positive det(Fp) and det(Fe) (InfeasibleState).
    // Non-finite F/Fp are invalid_argument. Other inputs are fixed when
    // differentiating F; Hessian uses column-major vec and is not projected.
    double value(const Mat3& F, const Mat3& Fp, const Parameters& theta) const;
    Mat3 gradient_F(const Mat3& F, const Mat3& Fp, const Parameters& theta) const;
    Mat9 hessian_F(const Mat3& F, const Mat3& Fp, const Parameters& theta) const;
};

} // namespace litepgo
