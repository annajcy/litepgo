#pragma once

#include <Eigen/Core>

namespace litepgo::fem {

// Centroid rule on r,s,t >= 0, r+s+t <= 1, exact through degree 1.
// The weight includes the reference tetrahedron's volume of 1/6.
class TetCentroidQuadrature final {
  public:
    static constexpr int num_points = 1;

    static const Eigen::Matrix<double, num_points, 3>& points() {
        static const Eigen::Matrix<double, num_points, 3> result = Eigen::Matrix<double, num_points, 3>::Constant(0.25);
        return result;
    }

    static const Eigen::Matrix<double, num_points, 1>& weights() {
        static const Eigen::Matrix<double, num_points, 1> result =
            Eigen::Matrix<double, num_points, 1>::Constant(1.0 / 6);
        return result;
    }
};

} // namespace litepgo::fem
