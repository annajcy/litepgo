#pragma once

#include "litepgo/fem/quadrature/quadrature.hpp"

#include <unsupported/Eigen/KroneckerProduct>

namespace litepgo {

// Two copies of a 1D rule on a Cartesian product; weights multiply.
template <Quadrature1D Rule1D> class TensorProductQuadrature2D final {
  public:
    static constexpr int n = Rule1D::num_points;
    static constexpr int num_points = n * n;

    // Row q contains (xi,eta). The last parameter varies fastest.
    static const Eigen::Matrix<double, num_points, 2>& points() {
        static const Eigen::Matrix<double, num_points, 2> result = compute_points();
        return result;
    }
    static const Eigen::Matrix<double, num_points, 1>& weights() {
        const auto& w = Rule1D::weights();
        static const Eigen::Matrix<double, num_points, 1> result = Eigen::kroneckerProduct(w, w);
        return result;
    }

  private:
    static Eigen::Matrix<double, num_points, 2> compute_points() {
        const auto& t = Rule1D::points();
        Eigen::Matrix<double, num_points, 2> coordinates;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                coordinates.row(i * n + j) << t[i], t[j];
        return coordinates;
    }
};

} // namespace litepgo
