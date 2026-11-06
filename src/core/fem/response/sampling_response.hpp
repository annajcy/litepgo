#pragma once

#include "litepgo/energy/response/affine_response.hpp"
#include "litepgo/fem/basis/basis.hpp"
#include <limits>
#include <stdexcept>
#include <tuple>
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
} // namespace litepgo::fem
