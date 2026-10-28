#include "litepgo/fem/kinematics/triangle_mid_edge_kinematics.hpp"
#include "litepgo/energy/energy.hpp"

#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace litepgo {
namespace {

constexpr double geometry_tolerance = 1e-12;

Eigen::Vector3d checked_normal(const Eigen::Vector3d& e1, const Eigen::Vector3d& e2) {
    const Eigen::Vector3d N = e1.cross(e2);
    const double norm_N = N.norm();
    const double scale = e1.norm() * e2.norm();
    if (!std::isfinite(norm_N) || !std::isfinite(scale) || !(norm_N > geometry_tolerance * scale))
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
        if (!std::isfinite(norm_sum) || !(norm_sum > geometry_tolerance * (N.norm() + N_k.norm())))
            throw InfeasibleState("Triangle mid-edge kinematics: adjacent face normals cancel");
    }
}

Eigen::Matrix3d cross_matrix(const Eigen::Vector3d& v) {
    Eigen::Matrix3d result;
    result << 0, -v[2], v[1], v[2], 0, -v[0], -v[1], v[0], 0;
    return result;
}

} // namespace

TriangleMidEdgeKinematics eval_triangle_mid_edge_kinematics(const Eigen::Ref<const Eigen::MatrixXd>& x,
                                                            const std::array<int, 3>& opposite, Eigen::MatrixXd* dab_dx,
                                                            std::array<Eigen::MatrixXd, 6>* d2ab_dx2) {
    check_input(x, opposite);
    const Eigen::Vector3d e1 = x.row(1) - x.row(0), e2 = x.row(2) - x.row(0);
    const Eigen::Vector3d N = e1.cross(e2);
    const int n = static_cast<int>(x.size());
    const bool derivatives = dab_dx || d2ab_dx2;
    constexpr std::array<int, 4> alpha{-1, 1, 0, 0}, beta{-1, 0, 1, 0};
    constexpr std::array<std::array<int, 3>, 3> b_rows{{{3, 4, 5}, {3, -1, -1}, {5, -1, -1}}};
    Eigen::MatrixXd D;
    std::array<Eigen::MatrixXd, 6> H;
    if (dab_dx) {
        D.setZero(6, n);
        D.block<1, 3>(0, 0) = -2 * e1.transpose();
        D.block<1, 3>(0, 3) = 2 * e1.transpose();
        D.block<1, 3>(1, 0) = -(e1 + e2).transpose();
        D.block<1, 3>(1, 3) = e2.transpose();
        D.block<1, 3>(1, 6) = e1.transpose();
        D.block<1, 3>(2, 0) = -2 * e2.transpose();
        D.block<1, 3>(2, 6) = 2 * e2.transpose();
    }
    if (d2ab_dx2) {
        for (auto& h : H)
            h.setZero(n, n);
        for (int u = 0; u < 3; ++u)
            for (int v = 0; v < 3; ++v) {
                H[0].block<3, 3>(3 * u, 3 * v).diagonal().setConstant(2 * alpha[u] * alpha[v]);
                H[1].block<3, 3>(3 * u, 3 * v).diagonal().setConstant(alpha[u] * beta[v] + beta[u] * alpha[v]);
                H[2].block<3, 3>(3 * u, 3 * v).diagonal().setConstant(2 * beta[u] * beta[v]);
            }
    }
    std::array<Eigen::Matrix3d, 4> dN_dx;
    if (derivatives)
        dN_dx = {cross_matrix(e2 - e1), -cross_matrix(e2), cross_matrix(e1), Eigen::Matrix3d::Zero()};
    Eigen::Vector3d c = Eigen::Vector3d::Zero();
    for (int k = 0; k < 3; ++k) {
        const int p = opposite[static_cast<std::size_t>(k)];
        if (p < 0)
            continue;
        const int j = (k + 1) % 3, l = (k + 2) % 3;
        const Eigen::Vector3d v1 = x.row(l) - x.row(p), v2 = x.row(j) - x.row(p);
        const Eigen::Vector3d N_k = v1.cross(v2), m = N + N_k;
        const Eigen::Vector3d q = (x.row(j) - x.row(k)) + (x.row(l) - x.row(k));
        const double r = m.norm(), s = q.dot(N_k), s_over_r = s / r;
        c[k] = -s_over_r;
        if (!derivatives)
            continue;

        const std::array<int, 4> rows{0, 1, 2, p};
        std::array<int, 4> d_q{1, 1, 1, 0};
        d_q[k] = -2;
        std::array<Eigen::Matrix3d, 4> dNk_over_r;
        dNk_over_r[k].setZero();
        dNk_over_r[3] = cross_matrix((v2 - v1) / r);
        dNk_over_r[l] = -cross_matrix(v2 / r);
        dNk_over_r[j] = cross_matrix(v1 / r);
        const Eigen::Vector3d m_hat = m / r;
        std::array<Eigen::Matrix3d, 4> dm_over_r;
        std::array<Eigen::Vector3d, 4> ds_over_r, dr_over_r;
        for (int u = 0; u < 4; ++u) {
            dm_over_r[u] = dN_dx[u] / r + dNk_over_r[u];
            ds_over_r[u] = d_q[u] * (N_k / r) + dNk_over_r[u].transpose() * q;
            dr_over_r[u] = dm_over_r[u].transpose() * m_hat;
            if (dab_dx) {
                const Eigen::Vector3d dc = -ds_over_r[u] + s_over_r * dr_over_r[u];
                for (int row : b_rows[k])
                    if (row >= 0)
                        D.block<1, 3>(row, 3 * rows[u]) += dc.transpose();
            }
        }
        if (!d2ab_dx2)
            continue;
        std::array<int, 4> alpha_k{}, beta_k{};
        alpha_k[3] = beta_k[3] = -1;
        alpha_k[l] = beta_k[j] = 1;
        const Eigen::Matrix3d cross_q_over_r = cross_matrix(q / r), cross_m_hat_over_r = cross_matrix(m_hat / r);
        // For a normal (x1-x0) x (x2-x0), its Hessian contracted with w has block
        // (alpha_v*beta_u-alpha_u*beta_v)*[w]_x. Here c=-s/r, s=q.N_k, r=|N+N_k|.
        for (int u = 0; u < 4; ++u)
            for (int v = u; v < 4; ++v) {
                const int hN = alpha[v] * beta[u] - alpha[u] * beta[v];
                const int hNk = alpha_k[v] * beta_k[u] - alpha_k[u] * beta_k[v];
                const Eigen::Matrix3d d2s_over_r =
                    d_q[u] * dNk_over_r[v] + d_q[v] * dNk_over_r[u].transpose() + hNk * cross_q_over_r;
                const Eigen::Matrix3d d2c =
                    -d2s_over_r + ds_over_r[u] * dr_over_r[v].transpose() + dr_over_r[u] * ds_over_r[v].transpose() +
                    s_over_r * (dm_over_r[u].transpose() * dm_over_r[v] + (hN + hNk) * cross_m_hat_over_r -
                                3 * dr_over_r[u] * dr_over_r[v].transpose());
                for (int row : b_rows[k])
                    if (row >= 0) {
                        H[row].block<3, 3>(3 * rows[u], 3 * rows[v]) += d2c;
                        if (u != v)
                            H[row].block<3, 3>(3 * rows[v], 3 * rows[u]) += d2c.transpose();
                    }
            }
    }
    TriangleMidEdgeKinematics result;
    result.a << e1.dot(e1), e1.dot(e2), e1.dot(e2), e2.dot(e2);
    result.b << c[0] + c[1], c[0], c[0], c[0] + c[2];
    if (!result.a.allFinite() || !result.b.allFinite())
        throw InfeasibleState("Triangle mid-edge kinematics: non-finite fundamental forms");
    if (dab_dx && !D.allFinite())
        throw InfeasibleState("Triangle mid-edge kinematics: non-finite first derivatives");
    if (d2ab_dx2)
        for (const auto& h : H)
            if (!h.allFinite())
                throw InfeasibleState("Triangle mid-edge kinematics: non-finite second derivatives");
    if (dab_dx)
        *dab_dx = std::move(D);
    if (d2ab_dx2)
        *d2ab_dx2 = std::move(H);
    return result;
}

double eval_triangle_mid_edge_feasible_step(const Eigen::Ref<const Eigen::MatrixXd>& x,
                                            const Eigen::Ref<const Eigen::MatrixXd>& dx,
                                            const std::array<int, 3>& opposite) {
    if (dx.rows() != x.rows() || dx.cols() != 3 || !dx.allFinite())
        throw std::invalid_argument("Triangle mid-edge feasible step: invalid position direction");
    eval_triangle_mid_edge_kinematics(x, opposite);
    double alpha = 1;
    const auto bound = [&](double margin, double rate) {
        if (!(margin > 0) || !std::isfinite(margin))
            throw InfeasibleState("Triangle mid-edge feasible step: invalid starting geometry");
        if (!std::isfinite(rate))
            alpha = 0;
        else if (rate > 0)
            alpha = std::min(alpha, 0.5 * margin / rate);
    };
    const auto normal = [&](int i, int j, int k) {
        const Eigen::Vector3d e1 = x.row(j) - x.row(i), e2 = x.row(k) - x.row(i);
        const Eigen::Vector3d de1 = dx.row(j) - dx.row(i), de2 = dx.row(k) - dx.row(i);
        const std::array<Eigen::Vector3d, 3> N{e1.cross(e2), de1.cross(e2) + e1.cross(de2), de1.cross(de2)};
        // N(t)=N0+t*N1+t^2*N2; for 0<=t<=1 its change is bounded by t*(|N1|+|N2|).
        const double scale_rate = e1.norm() * de2.norm() + e2.norm() * de1.norm() + de1.norm() * de2.norm();
        bound(N[0].norm() - geometry_tolerance * e1.norm() * e2.norm(),
              N[1].norm() + N[2].norm() + geometry_tolerance * scale_rate);
        return N;
    };
    const auto N = normal(0, 1, 2);
    for (int k = 0; k < 3; ++k) {
        const int p = opposite[static_cast<std::size_t>(k)];
        if (p < 0)
            continue;
        const auto Nk = normal(p, (k + 2) % 3, (k + 1) % 3);
        const double rate = (N[1] + Nk[1]).norm() + (N[2] + Nk[2]).norm() +
                            geometry_tolerance * (N[1].norm() + N[2].norm() + Nk[1].norm() + Nk[2].norm());
        bound((N[0] + Nk[0]).norm() - geometry_tolerance * (N[0].norm() + Nk[0].norm()), rate);
    }
    return alpha;
}

} // namespace litepgo
