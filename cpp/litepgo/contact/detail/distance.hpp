#pragma once

#include <Eigen/Core>
#include <array>
#include <cmath>
#include <limits>

namespace litepgo::contact::detail {

// A local second-order scalar: analytic chain rules, at most four xyz vertices.
struct Jet {
    double value = 0;
    Eigen::Matrix<double, 12, 1> gradient = Eigen::Matrix<double, 12, 1>::Zero();
    Eigen::Matrix<double, 12, 12> hessian = Eigen::Matrix<double, 12, 12>::Zero();
    Jet() = default;
    Jet(double v) : value(v) {}
};
inline Jet operator+(const Jet& a, const Jet& b) {
    Jet r(a.value + b.value);
    r.gradient = a.gradient + b.gradient;
    r.hessian = a.hessian + b.hessian;
    return r;
}
inline Jet operator-(const Jet& a, const Jet& b) {
    Jet r(a.value - b.value);
    r.gradient = a.gradient - b.gradient;
    r.hessian = a.hessian - b.hessian;
    return r;
}
inline Jet operator*(const Jet& a, const Jet& b) {
    Jet r(a.value * b.value);
    r.gradient = a.gradient * b.value + b.gradient * a.value;
    r.hessian = a.hessian * b.value + b.hessian * a.value + a.gradient * b.gradient.transpose() +
                b.gradient * a.gradient.transpose();
    return r;
}
inline Jet compose(const Jet& a, double value, double first, double second) {
    Jet r(value);
    r.gradient = first * a.gradient;
    r.hessian = first * a.hessian + second * a.gradient * a.gradient.transpose();
    return r;
}
inline Jet operator/(const Jet& a, const Jet& b) {
    const double inv = 1 / b.value;
    return a * compose(b, inv, -inv * inv, 2 * inv * inv * inv);
}
inline Jet log(const Jet& a) {
    return compose(a, std::log(a.value), 1 / a.value, -1 / (a.value * a.value));
}
inline double scalar(double x) {
    return x;
}
inline double scalar(const Jet& x) {
    return x.value;
}
template <class S> using V3 = std::array<S, 3>;
template <class S> V3<S> sub(const V3<S>& a, const V3<S>& b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}
template <class S> V3<S> cross(const V3<S>& a, const V3<S>& b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
template <class S> S dot(const V3<S>& a, const V3<S>& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
template <class S> S pp(const V3<S>& a, const V3<S>& b) {
    auto d = sub(a, b);
    return dot(d, d);
}
template <class S> S pe(const V3<S>& p, const V3<S>& a, const V3<S>& b) {
    auto e = sub(b, a), n = cross(sub(p, a), e);
    return dot(n, n) / dot(e, e);
}
template <class S> S pt(const V3<S>& p, const V3<S>& a, const V3<S>& b, const V3<S>& c) {
    auto n = cross(sub(b, a), sub(c, a));
    auto d = dot(sub(p, a), n);
    return d * d / dot(n, n);
}
template <class S> S ee(const V3<S>& a, const V3<S>& b, const V3<S>& c, const V3<S>& d) {
    auto n = cross(sub(b, a), sub(d, c));
    auto v = dot(sub(c, a), n);
    return v * v / dot(n, n);
}

// Feature indices refer to the four vertices in a query. n=2/3/4 denotes
// point-point/point-edge/point-triangle; line_line distinguishes edge-edge.
struct Feature {
    std::array<int, 4> ids{0, 1, -1, -1};
    int n = 2;
    bool line_line = false;
};
using Points = std::array<V3<double>, 4>;
inline Feature point_segment(const Points& p, int v, int a, int b) {
    auto e = sub(p[b], p[a]);
    const double l = dot(e, e);
    const double t = l > 0 ? dot(sub(p[v], p[a]), e) / l : 0;
    if (t <= 0)
        return {{v, a, -1, -1}, 2};
    if (t >= 1)
        return {{v, b, -1, -1}, 2};
    return {{v, a, b, -1}, 3};
}
template <class S> S distance(const std::array<V3<S>, 4>& p, const Feature& f) {
    const auto& a = p[f.ids[0]];
    const auto& b = p[f.ids[1]];
    if (f.n == 2)
        return pp(a, b);
    const auto& c = p[f.ids[2]];
    if (f.n == 3)
        return pe(a, b, c);
    return f.line_line ? ee(a, b, c, p[f.ids[3]]) : pt(a, b, c, p[f.ids[3]]);
}
inline Feature point_triangle(const Points& p) {
    auto e = sub(p[2], p[1]), f = sub(p[3], p[1]), w = sub(p[0], p[1]);
    auto n = cross(e, f);
    const double nn = dot(n, n);
    if (nn > 0) {
        const double u = dot(cross(w, f), n) / nn, v = dot(cross(e, w), n) / nn;
        if (u > 0 && v > 0 && u + v < 1)
            return {{0, 1, 2, 3}, 4};
    }
    Feature best = point_segment(p, 0, 1, 2);
    double d = distance(p, best);
    for (auto ab : {std::array{2, 3}, std::array{3, 1}}) {
        auto candidate = point_segment(p, 0, ab[0], ab[1]);
        double q = distance(p, candidate);
        if (q < d) {
            best = candidate;
            d = q;
        }
    }
    return best;
}
inline Feature edge_edge(const Points& p) {
    Feature best = point_segment(p, 0, 2, 3);
    double d = distance(p, best);
    for (auto vab : {std::array{1, 2, 3}, std::array{2, 0, 1}, std::array{3, 0, 1}}) {
        auto f = point_segment(p, vab[0], vab[1], vab[2]);
        double q = distance(p, f);
        if (q < d) {
            best = f;
            d = q;
        }
    }
    auto e = sub(p[1], p[0]), f = sub(p[3], p[2]), w = sub(p[2], p[0]), n = cross(e, f);
    const double nn = dot(n, n);
    if (nn > 0) {
        double s = dot(cross(w, f), n) / nn, t = dot(cross(w, e), n) / nn;
        if (s > 0 && s < 1 && t > 0 && t < 1)
            return {{0, 1, 2, 3}, 4, true};
    }
    return best;
}
} // namespace litepgo::contact::detail
