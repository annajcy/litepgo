#pragma once

#include <Eigen/Core>

namespace litepgo::fem {

// Centroid rule on xi,eta >= 0, xi+eta <= 1, exact through degree 1.
// The weight includes the reference triangle's area of 1/2.
class TriangleCentroidQuadrature final {
  public:
    static constexpr int num_points = 1;

    static const Eigen::Matrix<double, num_points, 2>& points() {
        static const Eigen::Matrix<double, num_points, 2> result =
            Eigen::Matrix<double, num_points, 2>::Constant(1.0 / 3);
        return result;
    }

    static const Eigen::Matrix<double, num_points, 1>& weights() {
        static const Eigen::Matrix<double, num_points, 1> result = Eigen::Matrix<double, num_points, 1>::Constant(0.5);
        return result;
    }
};

} // namespace litepgo::fem
