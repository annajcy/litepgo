#include "litepgo/fem/kinematics/positive_det_feasible_step.hpp"

#include <Eigen/Geometry>
#include <Eigen/LU>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace litepgo::fem {

double eval_positive_det_feasible_step(const Eigen::Matrix3d& A, const Eigen::Matrix3d& delta_A) {
    if (!A.allFinite() || !delta_A.allFinite())
        throw std::invalid_argument("Positive determinant step requires finite matrices");
    const double det_A = A.determinant();
    if (!std::isfinite(det_A) || det_A <= 0)
        throw std::domain_error("Positive determinant step requires finite det(A)>0");

    // Positive column scaling preserves roots and bounds the cubic coefficients.
    Eigen::Matrix3d a, d;
    for (int j = 0; j < 3; ++j) {
        const double scale = std::max(A.col(j).cwiseAbs().maxCoeff(), delta_A.col(j).cwiseAbs().maxCoeff());
        a.col(j) = A.col(j) / scale;
        d.col(j) = delta_A.col(j) / scale;
    }
    const double c0 = a.determinant();
    const double c1 = d.col(0).dot(a.col(1).cross(a.col(2))) + a.col(0).dot(d.col(1).cross(a.col(2))) +
                      a.col(0).dot(a.col(1).cross(d.col(2)));
    const double c2 = d.col(0).dot(d.col(1).cross(a.col(2))) + d.col(0).dot(a.col(1).cross(d.col(2))) +
                      a.col(0).dot(d.col(1).cross(d.col(2)));
    const double c3 = d.determinant();
    if (!(c0 > 0))
        return 0;
    const auto value = [&](double t) { return std::fma(std::fma(std::fma(c3, t, c2), t, c1), t, c0); };
    const auto tolerance = [&](double t) {
        return 64 * std::numeric_limits<double>::epsilon() *
               (((std::abs(c3) * t + std::abs(c2)) * t + std::abs(c1)) * t + c0);
    };

    std::array<double, 4> knots{0, 1};
    int count = 2;
    const auto add_knot = [&](double t) {
        if (t > 0 && t < 1)
            knots[static_cast<std::size_t>(count++)] = t;
    };
    const double qa = 3 * c3, qb = 2 * c2, qc = c1;
    if (qa == 0) {
        if (qb != 0)
            add_knot(-qc / qb);
    } else {
        double discriminant = std::fma(qb, qb, -4 * qa * qc);
        const double error = 8 * std::numeric_limits<double>::epsilon() * (qb * qb + std::abs(4 * qa * qc));
        if (discriminant >= -error) {
            discriminant = std::max(0.0, discriminant);
            const double q = -0.5 * (qb + std::copysign(std::sqrt(discriminant), qb));
            if (q == 0)
                add_knot(-qb / (2 * qa));
            else {
                add_knot(q / qa);
                add_knot(qc / q);
            }
        }
    }
    std::sort(knots.begin(), knots.begin() + count);
    // Between stationary points the cubic is monotone; checking the stationary
    // values also catches a root that touches zero without changing sign.
    for (int i = 1; i < count; ++i) {
        double lo = knots[static_cast<std::size_t>(i - 1)], hi = knots[static_cast<std::size_t>(i)];
        if (value(hi) > tolerance(hi))
            continue;
        for (;;) {
            const double mid = lo + (hi - lo) / 2;
            if (mid == lo || mid == hi)
                break;
            if (value(mid) > tolerance(mid))
                lo = mid;
            else
                hi = mid;
        }
        return 0.9 * lo;
    }
    return 1;
}

} // namespace litepgo::fem
