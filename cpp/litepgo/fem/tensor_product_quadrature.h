#pragma once

#include <Eigen/Core>
#include <unsupported/Eigen/KroneckerProduct>

namespace litepgo {

// Three copies of a 1D rule with fixed-size Eigen column vectors returned by
// points() and weights(). Coordinates form a Cartesian product; weights multiply.
template <class Rule1D> class TensorProductQuadrature3D final {
  public:
    static constexpr int n = Rule1D::num_points;
    static_assert(n > 0);
    static constexpr int num_points = n * n * n;

    // Row q contains (xi, eta, zeta). The last parameter varies fastest.
    using Points = Eigen::Matrix<double, num_points, 3>;
    using Weights = Eigen::Matrix<double, num_points, 1>;

    static const Points& points() {
        static const Points result = compute_points();
        return result;
    }

    static const Weights& weights() {
        const auto& w = Rule1D::weights();
        static const Weights result = Eigen::kroneckerProduct(w, Eigen::kroneckerProduct(w, w));
        return result;
    }

  private:
    static Points compute_points() {
        const auto& t = Rule1D::points();
        Points coordinates;
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

} // namespace litepgo
