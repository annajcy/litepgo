#include "litepgo/contact/detail/distance_derivatives.hpp"
#include "litepgo/contact/detail/ipc_derivatives/derivatives.hpp"

namespace litepgo::contact::detail {
void distance_derivatives(const Points& p, const Feature& f, LocalGradient& gradient, LocalHessian* hessian) {
    using namespace ipc_derivatives;
    const auto& a = p[f.ids[0]];
    const auto& b = p[f.ids[1]];
    LocalGradient g;
    LocalHessian H;
    if (f.n == 2) {
        const Eigen::Map<const Eigen::Vector3d> pa(a.data()), pb(b.data());
        g.head<6>() = point_point_distance_gradient(pa, pb);
        if (hessian) {
            Eigen::Map<Matrix6> compact(H.data());
            compact = point_point_distance_hessian(pa, pb);
        }
    } else {
        const auto& c = p[f.ids[2]];
        if (f.n == 3) {
            point_line_distance_gradient_3D(a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2], g.data());
            if (hessian)
                point_line_distance_hessian_3D(a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2], H.data());
        } else {
            const auto& d = p[f.ids[3]];
            const auto grad = f.line_line ? line_line_distance_gradient : point_plane_distance_gradient;
            grad(a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2], d[0], d[1], d[2], g.data());
            if (hessian) {
                const auto hess = f.line_line ? line_line_distance_hessian : point_plane_distance_hessian;
                hess(a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2], d[0], d[1], d[2], H.data());
            }
        }
    }
    gradient.setZero();
    if (hessian)
        hessian->setZero();
    for (int i = 0; i < 3 * f.n; ++i) {
        const int row = 3 * f.ids[i / 3] + i % 3;
        gradient[row] = g[i];
        if (hessian)
            for (int j = 0; j < 3 * f.n; ++j)
                (*hessian)(row, 3 * f.ids[j / 3] + j % 3) = H.data()[i + 3 * f.n * j];
    }
}
void cross_squarednorm_derivatives(const Points& p, LocalGradient& g, LocalHessian* H) {
    const auto& a = p[0];
    const auto& b = p[1];
    const auto& c = p[2];
    const auto& d = p[3];
    ipc_derivatives::edge_edge_cross_squarednorm_gradient(a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2], d[0],
                                                          d[1], d[2], g.data());
    if (H)
        ipc_derivatives::edge_edge_cross_squarednorm_hessian(a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2], d[0],
                                                             d[1], d[2], H->data());
}
} // namespace litepgo::contact::detail
