#pragma once

#include "litepgo/fem/quadrature/1d/gauss_legendre_1d.hpp"
#include "litepgo/fem/quadrature/quadrature.hpp"

namespace litepgo::fem {

// Gauss-Legendre on [0,1]^3; the last axis varies fastest.
template <int PointsPerAxis>
    requires Quadrature1D<GaussLegendre1D<PointsPerAxis>>
class GaussLegendre3D final {
  public:
    static constexpr int n = PointsPerAxis;
    static constexpr int num_points = n * n * n;

    static const Eigen::Matrix<double, num_points, 3>& points() {
        static const Eigen::Matrix<double, num_points, 3> result = [] {
            const auto& t = GaussLegendre1D<n>::points();
            Eigen::Matrix<double, num_points, 3> values;
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                    for (int k = 0; k < n; ++k) {
                        const int q = (i * n + j) * n + k;
                        values.row(q) << t[i], t[j], t[k];
                    }
            return values;
        }();
        return result;
    }

    static const Eigen::Matrix<double, num_points, 1>& weights() {
        static const Eigen::Matrix<double, num_points, 1> result = [] {
            const auto& w = GaussLegendre1D<n>::weights();
            Eigen::Matrix<double, num_points, 1> values;
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                    for (int k = 0; k < n; ++k) {
                        const int q = (i * n + j) * n + k;
                        values[q] = w[i] * w[j] * w[k];
                    }
            return values;
        }();
        return result;
    }
};

} // namespace litepgo::fem
