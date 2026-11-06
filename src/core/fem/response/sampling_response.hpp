#pragma once

#include "litepgo/energy/response/affine_response.hpp"
#include "litepgo/fem/basis/basis.hpp"
#include <Eigen/LU>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <unsupported/Eigen/KroneckerProduct>
#include <utility>
#include <vector>

namespace litepgo::fem {
namespace detail {
// Rows: point, derivative slot, component. Columns: basis coefficient, component.
template <class B, class Evaluate>
SpMat sampling_matrix(const Eigen::Ref<const Eigen::MatrixXd>& points, int components, Evaluate evaluate) {
    constexpr int dimension = std::tuple_size_v<decltype(B::coefficient_info(0).derivative_order)>;
    static_assert(Basis<B, dimension>);
    using Coordinate = std::conditional_t<dimension == 1, double, Eigen::Matrix<double, dimension, 1>>;
    using Derivatives = decltype(evaluate(std::declval<Coordinate>()));
    constexpr int slots = Derivatives::ColsAtCompileTime;
    static_assert(slots > 0);
    constexpr auto limit = std::numeric_limits<int>::max();
    if (points.cols() != dimension || !points.allFinite() || components <= 0)
        throw std::invalid_argument("Sampling response: require finite (Q, dimension) points and positive components");
    if (components > limit / B::num_basis_functions || components > limit / slots ||
        points.rows() > limit / (slots * components))
        throw std::invalid_argument("Sampling response: dimensions exceed int-sized sparse indices");
    const int rows = static_cast<int>(points.rows()) * slots * components;
    std::vector<Triplet> entries;
    for (Eigen::Index q = 0; q < points.rows(); ++q) {
        const Coordinate xi = [&]() -> Coordinate {
            if constexpr (dimension == 1)
                return points(q, 0);
            else
                return points.row(q).transpose();
        }();
        const auto values = evaluate(xi);
        if (!values.allFinite())
            throw std::domain_error("Sampling response: non-finite basis evaluation");
        for (int a = 0; a < B::num_basis_functions; ++a)
            for (int j = 0; j < slots; ++j) {
                if (values(a, j) == 0)
                    continue;
                if (entries.size() > static_cast<std::size_t>(limit - components))
                    throw std::invalid_argument("Sampling response: too many sparse entries");
                for (int c = 0; c < components; ++c)
                    entries.emplace_back((static_cast<int>(q) * slots + j) * components + c, a * components + c,
                                         values(a, j));
            }
    }
    SpMat A(rows, B::num_basis_functions * components);
    A.setFromTriplets(entries.begin(), entries.end());
    return A;
}
} // namespace detail

// Fixed points, variable row-major coefficients; no integration weights or geometry transform.
template <class B>
energy::AffineResponse value_sampling_response(const Eigen::Ref<const Eigen::MatrixXd>& points, int components) {
    return energy::AffineResponse(
        detail::sampling_matrix<B>(points, components, [](const auto& xi) { return B::eval_N(xi); }));
}

// Samples reference gradients, with each component-by-coordinate matrix flattened by column.
template <class B>
energy::AffineResponse gradient_sampling_response(const Eigen::Ref<const Eigen::MatrixXd>& points, int components) {
    return energy::AffineResponse(
        detail::sampling_matrix<B>(points, components, [](const auto& xi) { return B::eval_dN_dxi(xi); }));
}

// Samples independent reference Hessian entries in Basis order, then component; mixed entries are unscaled.
// The returned Response's own coefficient Hessian is zero.
template <class B>
energy::AffineResponse hessian_sampling_response(const Eigen::Ref<const Eigen::MatrixXd>& points, int components) {
    return energy::AffineResponse(
        detail::sampling_matrix<B>(points, components, [](const auto& xi) { return B::eval_d2N_dxi2(xi); }));
}
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
        const Eigen::Matrix3d J = geometry.eval_jacobian(points.row(q).transpose());
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
