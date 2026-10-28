#include "litepgo/fem/kinematics/triangle_mid_edge_kinematics.hpp"
#include "litepgo/energy/energy.hpp"

#include <Eigen/Geometry>
#include <cmath>
#include <stdexcept>
#include <unsupported/Eigen/AutoDiff>
#include <utility>

namespace litepgo {
namespace {

Eigen::Vector3d checked_normal(const Eigen::Vector3d& e1, const Eigen::Vector3d& e2) {
    const Eigen::Vector3d N = e1.cross(e2);
    const double norm_N = N.norm();
    const double scale = e1.norm() * e2.norm();
    if (!std::isfinite(norm_N) || !std::isfinite(scale) || !(norm_N > 1e-12 * scale))
        throw InfeasibleState("Triangle mid-edge kinematics: degenerate or unrepresentable face");
    return N;
}

void check_input(const Eigen::Ref<const Eigen::MatrixXd>& x, const std::array<int, 3>& opposite) {
    if (x.rows() < 3 || x.rows() > 6 || x.cols() != 3 || !x.allFinite())
        throw std::invalid_argument("Triangle mid-edge kinematics requires a finite 3-to-6-by-3 stencil");
    for (int p : opposite)
        if (p != -1 && (p < 3 || p >= x.rows()))
            throw std::invalid_argument("Triangle mid-edge kinematics: invalid opposite row");
    const Eigen::Vector3d N = checked_normal(x.row(1) - x.row(0), x.row(2) - x.row(0));
    for (int k = 0; k < 3; ++k) {
        const int p = opposite[static_cast<std::size_t>(k)];
        if (p < 0)
            continue;
        const Eigen::Vector3d N_k = checked_normal(x.row((k + 2) % 3) - x.row(p), x.row((k + 1) % 3) - x.row(p));
        const double norm_sum = (N + N_k).norm();
        if (!std::isfinite(norm_sum) || !(norm_sum > 1e-12 * (N.norm() + N_k.norm())))
            throw InfeasibleState("Triangle mid-edge kinematics: adjacent face normals cancel");
    }
}

template <class Scalar>
Eigen::Matrix<Scalar, 6, 1> eval_ab(const Eigen::Matrix<Scalar, Eigen::Dynamic, 3>& x,
                                    const std::array<int, 3>& opposite) {
    const Eigen::Matrix<Scalar, 3, 1> e1 = x.row(1) - x.row(0), e2 = x.row(2) - x.row(0);
    const Eigen::Matrix<Scalar, 3, 1> N = e1.cross(e2);
    Eigen::Matrix<Scalar, 3, 1> c = Eigen::Matrix<Scalar, 3, 1>::Zero();
    for (int k = 0; k < 3; ++k) {
        const int p = opposite[static_cast<std::size_t>(k)];
        if (p < 0)
            continue;
        const Eigen::Matrix<Scalar, 3, 1> v1 = x.row((k + 2) % 3) - x.row(p);
        const Eigen::Matrix<Scalar, 3, 1> v2 = x.row((k + 1) % 3) - x.row(p);
        const Eigen::Matrix<Scalar, 3, 1> N_k = v1.cross(v2);
        const Eigen::Matrix<Scalar, 3, 1> q = (x.row((k + 1) % 3) - x.row(k)) + (x.row((k + 2) % 3) - x.row(k));
        c[k] = -q.dot(N_k) / (N + N_k).norm();
    }
    Eigen::Matrix<Scalar, 6, 1> ab;
    ab << e1.dot(e1), e1.dot(e2), e2.dot(e2), c[0] + c[1], c[0], c[0] + c[2];
    return ab;
}

} // namespace

TriangleMidEdgeKinematics eval_triangle_mid_edge_kinematics(const Eigen::Ref<const Eigen::MatrixXd>& x,
                                                            const std::array<int, 3>& opposite, Eigen::MatrixXd* dab_dx,
                                                            std::array<Eigen::MatrixXd, 6>* d2ab_dx2) {
    check_input(x, opposite);
    Eigen::Matrix<double, 6, 1> ab;
    if (!dab_dx && !d2ab_dx2) {
        ab = eval_ab<double>(x, opposite);
    } else {
        // shortcut: all derivative requests use second-order AD; split first-order work if profiling warrants it.
        constexpr int max_dofs = 18;
        using First = Eigen::AutoDiffScalar<Eigen::Matrix<double, max_dofs, 1>>;
        using Second = Eigen::AutoDiffScalar<Eigen::Matrix<First, max_dofs, 1>>;
        Eigen::Matrix<Second, Eigen::Dynamic, 3> active(x.rows(), 3);
        const int n = static_cast<int>(x.size());
        for (int i = 0; i < x.rows(); ++i)
            for (int c = 0; c < 3; ++c) {
                Second value{First(x(i, c))};
                value.value().derivatives()[3 * i + c] = 1;
                value.derivatives()[3 * i + c].value() = 1;
                active(i, c) = value;
            }
        const auto values = eval_ab<Second>(active, opposite);
        Eigen::MatrixXd D(6, n);
        std::array<Eigen::MatrixXd, 6> H;
        for (int i = 0; i < 6; ++i) {
            ab[i] = values[i].value().value();
            D.row(i) = values[i].value().derivatives().head(n).transpose();
            if (d2ab_dx2) {
                H[static_cast<std::size_t>(i)].resize(n, n);
                for (int j = 0; j < n; ++j)
                    H[static_cast<std::size_t>(i)].row(j) =
                        values[i].derivatives()[j].derivatives().head(n).transpose();
                if (!H[static_cast<std::size_t>(i)].allFinite())
                    throw InfeasibleState("Triangle mid-edge kinematics: non-finite second derivatives");
            }
        }
        if (!ab.allFinite() || !D.allFinite())
            throw InfeasibleState("Triangle mid-edge kinematics: non-finite values or derivatives");
        if (dab_dx)
            *dab_dx = std::move(D);
        if (d2ab_dx2)
            *d2ab_dx2 = std::move(H);
    }
    if (!ab.allFinite())
        throw InfeasibleState("Triangle mid-edge kinematics: non-finite fundamental forms");
    TriangleMidEdgeKinematics result;
    result.a << ab[0], ab[1], ab[1], ab[2];
    result.b << ab[3], ab[4], ab[4], ab[5];
    return result;
}

} // namespace litepgo
