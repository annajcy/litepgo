// Derived from IPC Toolkit 1.6.1. Copyright (c) 2020–2025 Zachary Ferguson.
// SPDX-License-Identifier: MIT — see LICENSE and README.md in this directory.
#include "derivatives.hpp"
#include <cassert>
namespace litepgo::contact::detail::ipc_derivatives {
Vector6 point_point_distance_gradient(
    const Eigen::Vector3d& p0, const Eigen::Vector3d& p1)
{
    int dim = static_cast<int>(p0.size());
    assert(p1.size() == dim);
    (void)p1;

    Vector6 grad(2 * dim);

    grad.head(dim) = 2.0 * (p0 - p1);
    grad.tail(dim) = -grad.head(dim);

    return grad;
}

Matrix6 point_point_distance_hessian(
    const Eigen::Vector3d& p0, const Eigen::Vector3d& p1)
{
    int dim = static_cast<int>(p0.size());
    assert(p1.size() == dim);
    (void)p1;

    Matrix6 hess(2 * dim, 2 * dim);

    hess.setZero();
    hess.diagonal().setConstant(2.0);
    for (int i = 0; i < dim; i++) {
        hess(i, i + dim) = hess(i + dim, i) = -2;
    }

    return hess;
}

}
