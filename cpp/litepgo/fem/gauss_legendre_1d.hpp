#pragma once

#include <Eigen/Core>

#include <cmath>

namespace litepgo {

template <int NumPoints> class GaussLegendre1D;

// Two-point Gauss-Legendre rule on [0,1], exact through degree 3.
template <> class GaussLegendre1D<2> final {
  public:
    static constexpr int num_points = 2;

    static const Eigen::Matrix<double, num_points, 1>& points() {
        static const Eigen::Matrix<double, num_points, 1> result{0.5 - 0.5 / std::sqrt(3.0),
                                                                 0.5 + 0.5 / std::sqrt(3.0)};
        return result;
    }

    static const Eigen::Matrix<double, num_points, 1>& weights() {
        static const Eigen::Matrix<double, num_points, 1> result = Eigen::Matrix<double, num_points, 1>::Constant(0.5);
        return result;
    }
};

} // namespace litepgo
