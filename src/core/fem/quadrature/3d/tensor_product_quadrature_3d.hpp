#pragma once

#include "litepgo/fem/quadrature/quadrature.hpp"

#include <Eigen/Core>
#include <unsupported/Eigen/KroneckerProduct>

namespace litepgo::fem {

// Three copies of a 1D rule with fixed-size Eigen column vectors returned by
// points() and weights(). Coordinates form a Cartesian product; weights multiply.
template <Quadrature1D Rule1D> class TensorProductQuadrature3D final {
  public:
    static constexpr int n = Rule1D::num_points;
    static constexpr int num_points = n * n * n;

    // Row q contains (xi, eta, zeta). The last parameter varies fastest.

    static const Eigen::Matrix<double, num_points, 3>& points() {
        static const Eigen::Matrix<double, num_points, 3> result = compute_points();
        return result;
    }

    static const Eigen::Matrix<double, num_points, 1>& weights() {
        const auto& w = Rule1D::weights();
        static const Eigen::Matrix<double, num_points, 1> result =
            Eigen::kroneckerProduct(w, Eigen::kroneckerProduct(w, w));
        return result;
    }

  private:
    static Eigen::Matrix<double, num_points, 3> compute_points() {
        const auto& t = Rule1D::points();
        Eigen::Matrix<double, num_points, 3> coordinates;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                for (int k = 0; k < n; ++k) {
                    const int q = (i * n + j) * n + k;
                    coordinates.row(q) << t[i], t[j], t[k];
                }
            }
        }
        return coordinates;
    }
};

} // namespace litepgo::fem
