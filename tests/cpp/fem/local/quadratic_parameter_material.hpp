#pragma once
#include "litepgo/fem/material/material_inputs.hpp"
#include <stdexcept>

// Nonzero parameter Hessian complements NeoHookean's linear theta dependence.
template <int Parameters> struct QuadraticParameterMaterial {
    QuadraticParameterMaterial() = delete; // The adapter must not construct a material.
    static constexpr int num_parameters = Parameters;
    static double value(const litepgo::MaterialInputs& inputs) {
        check(inputs);
        return 0.5 * inputs.theta.squaredNorm();
    }
    static Eigen::VectorXd gradient_theta(const litepgo::MaterialInputs& inputs) {
        check(inputs);
        return inputs.theta;
    }
    static Eigen::MatrixXd hessian_theta(const litepgo::MaterialInputs& inputs) {
        check(inputs);
        return Eigen::MatrixXd::Identity(num_parameters, num_parameters);
    }

  private:
    static void check(const litepgo::MaterialInputs& inputs) {
        if (inputs.theta.size() != num_parameters)
            throw std::invalid_argument("QuadraticParameterMaterial: invalid parameter count");
    }
};
