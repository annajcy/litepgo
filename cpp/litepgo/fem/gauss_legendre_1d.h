#pragma once

#include <Eigen/Core>

#include <cmath>

namespace litepgo {

template <int NumPoints> class GaussLegendre1D;

// Two-point Gauss-Legendre rule on [0,1], exact through degree 3.
template <> class GaussLegendre1D<2> final {
  public:
    static constexpr int num_points = 2;
    using Points = Eigen::Matrix<double, num_points, 1>;
    using Weights = Eigen::Matrix<double, num_points, 1>;

    static const Points& points() {
        static const Points result{0.5 - 0.5 / std::sqrt(3.0), 0.5 + 0.5 / std::sqrt(3.0)};
        return result;
    }

    static const Weights& weights() {
        static const Weights result = Weights::Constant(0.5);
        return result;
    }
};

} // namespace litepgo
