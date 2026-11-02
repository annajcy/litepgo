#pragma once

#include <Eigen/Core>

#include <cmath>

namespace litepgo::fem {

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

// Four-point Gauss-Legendre rule on [0,1], exact through degree 7.
template <> class GaussLegendre1D<4> final {
  public:
    static constexpr int num_points = 4;

    static const Eigen::Vector4d& points() {
        static const Eigen::Vector4d result = [] {
            const double outer = std::sqrt((3 + 2 * std::sqrt(6.0 / 5)) / 7);
            const double inner = std::sqrt((3 - 2 * std::sqrt(6.0 / 5)) / 7);
            return Eigen::Vector4d(0.5 * (1 - outer), 0.5 * (1 - inner), 0.5 * (1 + inner), 0.5 * (1 + outer));
        }();
        return result;
    }

    static const Eigen::Vector4d& weights() {
        static const Eigen::Vector4d result = [] {
            const double outer = (18 - std::sqrt(30.0)) / 72;
            const double inner = (18 + std::sqrt(30.0)) / 72;
            return Eigen::Vector4d(outer, inner, inner, outer);
        }();
        return result;
    }
};

} // namespace litepgo::fem
