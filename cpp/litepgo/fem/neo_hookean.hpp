#pragma once

#include "litepgo/fem/material_inputs.hpp"

namespace litepgo {

// W = det(Fp) [mu/2 (||Fe||^2 - 3) - mu log(Je) + lambda/2 log(Je)^2],
// Fe = F Fp^-1, Je = det(Fe). Density is per original reference volume.
// All objects below require finite theta = [lambda, mu], lambda >= 0, mu > 0,
// and finite F/Fp (invalid_argument otherwise). Non-positive det(Fp)/det(Fe)
// or unrepresentable kinematics throw InfeasibleState.
class NeoHookean final {
  public:
    static constexpr int num_parameters = 2;
    using Point = MaterialInputs<num_parameters>;

    double value(const Point& point) const;
};

// All three objects evaluate the same W. Derivatives hold the other arguments
// fixed, retain the argument shape, and use column-major vec for exact Hessians.
class NeoHookeanEnergyF final {
  public:
    using Point = MaterialInputs<2>;
    using Gradient = Eigen::Matrix3d;
    using Hessian = Eigen::Matrix<double, 9, 9>;

    double value(const Point& point) const {
        return NeoHookean{}.value(point);
    }
    Gradient gradient(const Point& point) const;
    Hessian hessian(const Point& point) const;
};

class NeoHookeanEnergyFp final {
  public:
    using Point = MaterialInputs<2>;
    using Gradient = Eigen::Matrix3d;
    using Hessian = Eigen::Matrix<double, 9, 9>;

    double value(const Point& point) const {
        return NeoHookean{}.value(point);
    }
    Gradient gradient(const Point& point) const;
    Hessian hessian(const Point& point) const;
};

class NeoHookeanEnergyTheta final {
  public:
    using Point = MaterialInputs<2>;
    using Gradient = Eigen::Vector2d;
    using Hessian = Eigen::Matrix2d;

    double value(const Point& point) const {
        return NeoHookean{}.value(point);
    }
    Gradient gradient(const Point& point) const;
    Hessian hessian(const Point& point) const;
};

} // namespace litepgo
