#pragma once

#include "litepgo/energy/response/affine_response.hpp"
#include "litepgo/fem/basis/basis.hpp"
#include <Eigen/LU>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unsupported/Eigen/KroneckerProduct>
#include <utility>
#include <vector>

namespace litepgo::fem {
// Fixed reference geometry and points; input is row-major displacement coefficients, not physical DOFs.
template <Basis3D B>
energy::AffineResponse deformation_gradient_sampling_response(energy::Response& geometry,
                                                              const Eigen::Ref<const Eigen::MatrixXd>& points) {
    if (geometry.num_inputs() != 3 || geometry.num_outputs() != 3 || points.cols() != 3 || !points.allFinite() ||
        points.rows() > std::numeric_limits<int>::max() / 9 ||
        B::num_basis_functions > std::numeric_limits<int>::max() / 3)
        throw std::invalid_argument("Deformation gradient sampling: require R3 geometry and finite (Q,3) points");
    std::vector<Triplet> entries;
    Vec offset = Vec::Zero(9 * points.rows());
    for (Eigen::Index q = 0; q < points.rows(); ++q) {
        const Eigen::Matrix3d J = geometry.evaluate(points.row(q).transpose())->jacobian();
        const double determinant = J.determinant();
        if (!std::isfinite(determinant) || determinant <= 0)
            throw std::invalid_argument("Deformation gradient sampling: positive geometry determinant required");
        const Eigen::MatrixXd gradient = B::eval_dN_dxi(points.row(q).transpose()) * J.inverse();
        const Eigen::MatrixXd Bq = Eigen::kroneckerProduct(gradient.transpose(), Eigen::Matrix3d::Identity());
        if (!Bq.allFinite())
            throw std::invalid_argument("Deformation gradient sampling: non-finite physical gradient");
        for (int c = 0; c < Bq.cols(); ++c)
            for (int r = 0; r < 9; ++r)
                if (Bq(r, c) != 0) {
                    if (entries.size() >= static_cast<std::size_t>(std::numeric_limits<int>::max()))
                        throw std::invalid_argument("Deformation gradient sampling: too many sparse entries");
                    entries.emplace_back(9 * q + r, c, Bq(r, c));
                }
        offset[9 * q] = offset[9 * q + 4] = offset[9 * q + 8] = 1;
    }
    SpMat A(offset.size(), 3 * B::num_basis_functions);
    A.setFromTriplets(entries.begin(), entries.end());
    return energy::AffineResponse(std::move(A), std::move(offset));
}
} // namespace litepgo::fem
