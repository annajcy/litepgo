#pragma once

// Central-difference checks of an energy's gradient and exact Hessian.

#include "litepgo/energy/energy.h"

#include <Eigen/Dense>
#include <gtest/gtest.h>

namespace litepgo::testing {

inline Vec finite_difference_gradient(const Energy& energy, const Vec& x, double step = 1e-6) {
    Vec gradient(x.size());
    for (Eigen::Index i = 0; i < x.size(); ++i) {
        Vec forward = x;
        Vec backward = x;
        forward[i] += step;
        backward[i] -= step;
        gradient[i] = (energy.value(forward) - energy.value(backward)) / (2.0 * step);
    }
    return gradient;
}

inline Eigen::MatrixXd finite_difference_hessian(const Energy& energy, const Vec& x, double step = 1e-6) {
    Eigen::MatrixXd hessian(x.size(), x.size());
    for (Eigen::Index j = 0; j < x.size(); ++j) {
        Vec forward = x;
        Vec backward = x;
        forward[j] += step;
        backward[j] -= step;
        hessian.col(j) = (energy.gradient(forward) - energy.gradient(backward)) / (2.0 * step);
    }
    return hessian;
}

// Expects the gradient and exact Hessian at x to match central differences, to
// a tolerance relative to the size of the derivatives.
inline void expect_derivatives_match(const Energy& energy, const Vec& x, double tolerance = 1e-6) {
    const Vec gradient = energy.gradient(x);
    const Vec expected_gradient = finite_difference_gradient(energy, x);
    EXPECT_LE((gradient - expected_gradient).cwiseAbs().maxCoeff(),
              tolerance * (1.0 + expected_gradient.cwiseAbs().maxCoeff()));

    const Eigen::MatrixXd hessian = Eigen::MatrixXd(energy.hessian(x, HessianKind::Exact));
    const Eigen::MatrixXd expected_hessian = finite_difference_hessian(energy, x);
    EXPECT_LE((hessian - expected_hessian).cwiseAbs().maxCoeff(),
              tolerance * (1.0 + expected_hessian.cwiseAbs().maxCoeff()));
}

} // namespace litepgo::testing
