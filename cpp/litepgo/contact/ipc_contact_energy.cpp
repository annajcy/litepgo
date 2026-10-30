#include "litepgo/contact/ipc_contact_energy.hpp"
#include "litepgo/contact/detail/broad_phase.hpp"
#include "litepgo/contact/detail/distance.hpp"

#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <set>
#include <stdexcept>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tuple>
#include <utility>

namespace litepgo::contact {
namespace {
using namespace detail;
using Ids = std::array<int, 4>;
using V = Eigen::Vector3d;
struct Pair {
    Ids ids{-1, -1, -1, -1};
    Feature feature;
    double epsilon = 0;
    int plane = -1;
    double weight = 1;
    int count() const {
        return plane >= 0 ? 1 : (epsilon > 0 ? 4 : feature.n);
    }
    auto key() const {
        return std::tuple{plane, epsilon > 0, feature.n, feature.line_line, ids};
    }
};
int checked_size(const mesh::TriangleMesh& surface) {
    surface.validate();
    if (surface.topology.num_vertices() == 0 || surface.topology.num_cells() == 0 ||
        surface.topology.num_vertices() > std::numeric_limits<int>::max() / 3)
        throw std::invalid_argument("IPCContactEnergy requires a nonempty triangle surface within the DOF index range");
    return 3 * surface.topology.num_vertices();
}
IPCContactOptions checked_options(IPCContactOptions o) {
    const double threshold = o.dhat * (o.dhat + 2 * o.min_distance);
    if (!std::isfinite(o.dhat) || o.dhat <= 0 || !std::isfinite(o.min_distance) || o.min_distance < 0 ||
        !std::isfinite(threshold) || threshold <= 0 || !std::isfinite(o.min_distance * o.min_distance) ||
        !std::isfinite(o.ccd_tolerance) || o.ccd_tolerance <= 0 || o.ccd_max_iterations <= 0)
        throw std::invalid_argument("IPCContactEnergy requires finite positive dhat, CCD tolerance/budget and "
                                    "nonnegative min_distance with representable squared thresholds");
    return o;
}
void check_x(const ConstVecRef& x, int n) {
    if (x.size() != n || !x.allFinite())
        throw std::invalid_argument("IPCContactEnergy requires num_dofs finite positions");
}
Points points(const ConstVecRef& x, const Ids& ids) {
    Points p{};
    for (int i = 0; i < 4; ++i)
        if (ids[i] >= 0)
            for (int k = 0; k < 3; ++k)
                p[i][k] = x[3 * ids[i] + k];
    return p;
}
Pair reduced(const Ids& ids, Feature f) {
    Pair p;
    p.feature = f;
    for (int i = 0; i < f.n; ++i)
        p.ids[i] = ids[f.ids[i]];
    if (f.n == 2) {
        if (p.ids[1] < p.ids[0])
            std::swap(p.ids[0], p.ids[1]);
    } else if (f.n == 3) {
        if (p.ids[2] < p.ids[1])
            std::swap(p.ids[1], p.ids[2]);
    } else if (!f.line_line)
        std::sort(p.ids.begin() + 1, p.ids.end());
    p.feature.ids = {0, 1, 2, 3};
    return p;
}
} // namespace

struct IPCContactEnergy::Impl {
    std::vector<std::array<int, 3>> faces;
    std::vector<std::array<int, 2>> edges;
    std::vector<double> edge_lengths_squared;
    std::vector<Plane> planes;
    int nv;
    IPCContactOptions options;

    Impl(const mesh::TriangleMesh& surface, IPCContactOptions o, std::vector<Plane> input)
        : planes(std::move(input)), nv(surface.topology.num_vertices()), options(o) {
        std::set<std::array<int, 3>> unique;
        std::vector<bool> used(nv, false);
        for (int i = 0; i < surface.topology.num_cells(); ++i) {
            auto f = surface.topology.cell_vertices(i);
            auto sorted = f;
            std::sort(sorted.begin(), sorted.end());
            if (!unique.insert(sorted).second)
                throw std::invalid_argument("IPCContactEnergy: duplicate face");
            faces.push_back(f);
            for (int j = 0; j < 3; ++j) {
                used[f[j]] = true;
                edges.push_back({std::min(f[j], f[(j + 1) % 3]), std::max(f[j], f[(j + 1) % 3])});
            }
        }
        if (std::find(used.begin(), used.end(), false) != used.end())
            throw std::invalid_argument("IPCContactEnergy: unused surface vertex");
        std::sort(edges.begin(), edges.end());
        edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
        for (auto e : edges)
            edge_lengths_squared.push_back(
                (surface.positions.segment<3>(3 * e[0]) - surface.positions.segment<3>(3 * e[1])).squaredNorm());
        try {
            check_geometry(surface.positions);
        } catch (const std::domain_error&) {
            throw std::invalid_argument("IPCContactEnergy: degenerate or unrepresentable rest triangle");
        }
        for (auto& p : planes) {
            double length = p.normal.stableNorm();
            if (!p.point.allFinite() || !p.normal.allFinite() || !std::isfinite(length) || length <= 0)
                throw std::invalid_argument("IPCContactEnergy planes require finite points and nonzero finite normals");
            p.normal /= length;
        }
    }
    void check_geometry(const ConstVecRef& x) const {
        for (auto f : faces) {
            const V a = x.segment<3>(3 * f[1]) - x.segment<3>(3 * f[0]),
                    b = x.segment<3>(3 * f[2]) - x.segment<3>(3 * f[0]);
            const double area = a.cross(b).squaredNorm();
            if (!(area > 0) || !std::isfinite(area))
                throw std::domain_error("IPCContactEnergy: degenerate or unrepresentable triangle");
        }
    }
    void check_planes(const ConstVecRef& x) const {
        for (const auto& p : planes)
            for (int v = 0; v < nv; ++v) {
                const double d = p.normal.dot(x.segment<3>(3 * v) - p.point);
                if (!std::isfinite(d) || !(d > options.min_distance))
                    throw std::domain_error("IPCContactEnergy: plane distance must exceed min_distance");
            }
    }
    Candidates candidates(const ConstVecRef& x, const Vec* dx, double inflation, bool intersections = false) const {
        return broad_phase(x, dx, inflation, edges, faces, intersections);
    }
    Ids edge_ids(const Ids& pair) const {
        auto a = edges[pair[0]], b = edges[pair[1]];
        return {a[0], a[1], b[0], b[1]};
    }
    std::vector<Pair> active(const ConstVecRef& x) const {
        check_x(x, 3 * nv);
        check_geometry(x);
        auto candidates_ = candidates(x, nullptr, 0.5 * (options.dhat + options.min_distance));
        const double reach = options.dhat + options.min_distance;
        std::vector<Pair> pairs;
        for (auto ids : candidates_.vf) {
            auto p = points(x, ids);
            auto f = point_triangle(p);
            if (distance(p, f) < reach * reach)
                pairs.push_back(reduced(ids, f));
        }
        for (auto indices : candidates_.ee) {
            auto ids = edge_ids(indices);
            auto p = points(x, ids);
            auto f = edge_edge(p);
            if (distance(p, f) >= reach * reach)
                continue;
            double epsilon = 1e-3 * edge_lengths_squared[indices[0]] * edge_lengths_squared[indices[1]];
            if (!std::isfinite(epsilon) || epsilon <= 0)
                throw std::domain_error("IPCContactEnergy: unrepresentable mollifier threshold");
            auto n = cross(sub(p[1], p[0]), sub(p[3], p[2]));
            if (dot(n, n) < epsilon)
                pairs.push_back(Pair{ids, f, epsilon});
            else
                pairs.push_back(reduced(ids, f));
        }
        for (int i = 0; i < static_cast<int>(planes.size()); ++i)
            for (int v = 0; v < nv; ++v) {
                if (planes[i].normal.dot(x.segment<3>(3 * v) - planes[i].point) < reach) {
                    Pair p;
                    p.ids[0] = v;
                    p.plane = i;
                    pairs.push_back(p);
                }
            }
        std::sort(pairs.begin(), pairs.end(), [](const Pair& a, const Pair& b) { return a.key() < b.key(); });
        std::vector<Pair> merged;
        for (const auto& pair : pairs) {
            if (!merged.empty() && merged.back().key() == pair.key())
                merged.back().weight += pair.weight;
            else
                merged.push_back(pair);
        }
        return merged;
    }
    template <class S> S squared_distance(const std::array<V3<S>, 4>& p, const Pair& pair) const {
        if (pair.plane < 0)
            return distance(p, pair.feature);
        const auto& plane = planes[pair.plane];
        S d = 0;
        for (int k = 0; k < 3; ++k)
            d = d + S(plane.normal[k]) * (p[0][k] - S(plane.point[k]));
        return d * d;
    }
    template <class S> S potential(const std::array<V3<S>, 4>& p, const Pair& pair) const {
        const double h = options.dhat * (options.dhat + 2 * options.min_distance);
        S d = squared_distance(p, pair) - S(options.min_distance * options.min_distance);
        if (!(scalar(d) > 0) || !std::isfinite(scalar(d)))
            throw std::domain_error("IPCContactEnergy: contact distance must exceed min_distance");
        if (scalar(d) >= h)
            return S(0);
        using std::log;
        S b = S(-1) * (d - S(h)) * (d - S(h)) * log(d / S(h));
        if (pair.epsilon > 0) {
            auto n = cross(sub(p[1], p[0]), sub(p[3], p[2]));
            S z = dot(n, n) / S(pair.epsilon);
            if (scalar(z) < 1)
                b = b * z * (S(2) - z);
        }
        return S(pair.weight) * b;
    }
    void check_intersections(const ConstVecRef& x, const Candidates& c) const {
        // Noncoplanar edge-face crossings may have positive VF and EE distances.
        for (auto indices : c.ef) {
            auto e = edges[indices[0]];
            auto f = faces[indices[1]];
            V a = x.segment<3>(3 * e[0]), b = x.segment<3>(3 * e[1]);
            V origin = x.segment<3>(3 * f[0]), u = x.segment<3>(3 * f[1]) - origin, v = x.segment<3>(3 * f[2]) - origin;
            V n = u.cross(v);
            double da = n.dot(a - origin), db = n.dot(b - origin);
            if ((da > 0 && db > 0) || (da < 0 && db < 0) || da == db)
                continue;
            V w = a + (da / (da - db)) * (b - a) - origin;
            double s = w.cross(v).dot(n) / n.squaredNorm(), t = u.cross(w).dot(n) / n.squaredNorm();
            if (s >= 0 && t >= 0 && s + t <= 1)
                throw std::domain_error("IPCContactEnergy: surface intersects itself");
        }
    }
    double advance(const ConstVecRef& x, const ConstVecRef& dx, Ids ids, bool edge) const {
        Points start = points(x, ids), velocity = points(dx, ids);
        // Remove common motion to tighten the Lipschitz bound.
        for (int k = 0; k < 3; ++k) {
            double mean = 0;
            for (int i = 0; i < 4; ++i)
                mean += velocity[i][k] / 4;
            for (int i = 0; i < 4; ++i)
                velocity[i][k] -= mean;
        }
        auto norm = [&](int i) { return std::sqrt(dot(velocity[i], velocity[i])); };
        double speed = edge ? std::max(norm(0), norm(1)) + std::max(norm(2), norm(3))
                            : norm(0) + std::max({norm(1), norm(2), norm(3)});
        if (!std::isfinite(speed))
            throw std::domain_error("IPCContactEnergy: unrepresentable CCD velocity");
        double time = 0, initial_gap = 0;
        for (long iteration = 0; iteration < options.ccd_max_iterations; ++iteration) {
            Points p = start;
            for (int i = 0; i < 4; ++i)
                for (int k = 0; k < 3; ++k)
                    p[i][k] += time * velocity[i][k];
            double d2 = distance(p, edge ? edge_edge(p) : point_triangle(p));
            const double d = std::sqrt(d2),
                         gap = (d2 - options.min_distance * options.min_distance) / (d + options.min_distance);
            if (!std::isfinite(gap) || !(gap > 0)) {
                if (time == 0)
                    throw std::domain_error("IPCContactEnergy: contact distance must exceed min_distance");
                return 0; // Fail closed if numerical evaluation loses the safe separation.
            }
            if (iteration == 0)
                initial_gap = gap;
            if (speed == 0)
                return 1;
            if (time > 0 && gap <= 0.2 * initial_gap)
                return time;
            const double increment = 0.8 * gap / speed;
            if (increment >= 1 - time)
                return 1;
            if (!(increment > 0) || time + increment == time)
                return time;
            if (time > 0 && increment < options.ccd_tolerance)
                return time;
            time += increment;
        }
        return time;
    }
};

IPCContactEnergy::IPCContactEnergy(const mesh::TriangleMesh& surface, IPCContactOptions options,
                                   std::vector<Plane> planes)
    : DynamicPatternEnergy(checked_size(surface), SpMat(surface.positions.size(), surface.positions.size())),
      rest_positions_(surface.positions), options_(checked_options(options)),
      impl_(std::make_unique<const Impl>(surface, options_, std::move(planes))) {}
IPCContactEnergy::~IPCContactEnergy() = default;
int IPCContactEnergy::num_active_pairs(const ConstVecRef& x) const {
    return static_cast<int>(impl_->active(x).size());
}
double IPCContactEnergy::min_active_distance(const ConstVecRef& x) const {
    double d = INFINITY;
    for (const auto& pair : impl_->active(x))
        d = std::min(d, impl_->squared_distance(points(x, pair.ids), pair));
    return std::sqrt(d);
}
SpMat IPCContactEnergy::pattern_at(const ConstVecRef& x) const {
    std::vector<Triplet> entries;
    for (const auto& p : impl_->active(x))
        for (int a = 0; a < p.count(); ++a)
            for (int b = 0; b < p.count(); ++b)
                for (int r = 0; r < 3; ++r)
                    for (int c = 0; c < 3; ++c)
                        entries.emplace_back(3 * p.ids[a] + r, 3 * p.ids[b] + c, 1);
    SpMat pattern(num_dofs(), num_dofs());
    pattern.setFromTriplets(entries.begin(), entries.end());
    return pattern;
}
double IPCContactEnergy::eval_values(const ConstVecRef& x, Vec* gradient, SpMat* hessian,
                                     energy::HessianKind kind) const {
    check_x(x, num_dofs());
    impl_->check_planes(x);
    auto pairs = impl_->active(x);
    double value = 0;
    // Each task owns one slot; only the fixed-order assembly below reduces values.
    std::vector<double> values(pairs.size());
    std::vector<Jet> derivatives(gradient || hessian ? pairs.size() : 0);
    tbb::parallel_for(std::size_t(0), pairs.size(), [&](std::size_t index) {
        const auto& pair = pairs[index];
        auto p = points(x, pair.ids);
        if (!gradient && !hessian) {
            values[index] = impl_->potential(p, pair);
            return;
        }
        std::array<V3<Jet>, 4> jets{};
        for (int i = 0; i < pair.count(); ++i)
            for (int k = 0; k < 3; ++k) {
                jets[i][k].value = p[i][k];
                jets[i][k].gradient[3 * i + k] = 1;
            }
        Jet result = impl_->potential(jets, pair);
        if (!result.gradient.allFinite() || !result.hessian.allFinite())
            throw std::domain_error("IPCContactEnergy: non-finite derivatives");
        values[index] = result.value;
        if (hessian) {
            Eigen::MatrixXd local = result.hessian.topLeftCorner(3 * pair.count(), 3 * pair.count());
            if (kind == energy::HessianKind::Projected) {
                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(local);
                if (solver.info() != Eigen::Success)
                    throw std::domain_error("IPCContactEnergy: Hessian projection failed");
                local = solver.eigenvectors() * solver.eigenvalues().cwiseMax(0).asDiagonal() *
                        solver.eigenvectors().transpose();
            }
            result.hessian.topLeftCorner(3 * pair.count(), 3 * pair.count()) = local;
        }
        derivatives[index] = result;
    });
    for (double local_value : values)
        value += local_value;
    if (gradient || hessian) {
        if (hessian && hessian->nonZeros() == 0 && !pairs.empty())
            throw std::logic_error("IPCContactEnergy: active pair outside Hessian pattern; call update_pattern(x)");
        struct Contribution {
            std::size_t pair;
            int vertex;
        };
        // Per-call CSR incidence preserves pair order without a position-dependent cache.
        std::vector<std::size_t> offsets(num_vertices() + 1, 0);
        for (const auto& pair : pairs)
            for (int i = 0; i < pair.count(); ++i)
                ++offsets[pair.ids[i] + 1];
        std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());
        auto next = offsets;
        std::vector<Contribution> contributions(offsets.back());
        for (std::size_t index = 0; index < pairs.size(); ++index)
            for (int i = 0; i < pairs[index].count(); ++i)
                contributions[next[pairs[index].ids[i]]++] = {index, i};
        // Like dof::Assembler, each task owns complete CSC columns and gradient DOFs.
        tbb::parallel_for(tbb::blocked_range<int>(0, num_vertices(), 32), [&](const auto& range) {
            if (hessian && hessian->nonZeros() != 0) {
                const auto* outer = hessian->outerIndexPtr();
                std::fill_n(hessian->valuePtr() + outer[3 * range.begin()],
                            outer[3 * range.end()] - outer[3 * range.begin()], 0.0);
            }
            for (int v = range.begin(); v < range.end(); ++v) {
                V g = V::Zero();
                for (std::size_t entry = offsets[v]; entry < offsets[v + 1]; ++entry) {
                    const auto [index, b] = contributions[entry];
                    const auto& pair = pairs[index];
                    const auto& local = derivatives[index];
                    if (gradient)
                        g += local.gradient.segment<3>(3 * b);
                    if (hessian)
                        for (int c = 0; c < 3; ++c) {
                            const int col = 3 * v + c;
                            const int* begin = hessian->innerIndexPtr() + hessian->outerIndexPtr()[col];
                            const int* end = hessian->innerIndexPtr() + hessian->outerIndexPtr()[col + 1];
                            for (int a = 0; a < pair.count(); ++a) {
                                const int row = 3 * pair.ids[a];
                                const int* first = std::lower_bound(begin, end, row);
                                if (end - first < 3 || first[0] != row || first[1] != row + 1 || first[2] != row + 2)
                                    throw std::logic_error("IPCContactEnergy: active pair outside Hessian pattern; "
                                                           "call update_pattern(x)");
                                for (int r = 0; r < 3; ++r)
                                    hessian->valuePtr()[first - hessian->innerIndexPtr() + r] +=
                                        local.hessian(3 * a + r, 3 * b + c);
                            }
                        }
                }
                if (gradient)
                    gradient->segment<3>(3 * v) = g;
            }
        });
    }
    if (!std::isfinite(value) || (gradient && !gradient->allFinite()))
        throw std::domain_error("IPCContactEnergy: non-finite assembly");
    if (hessian && !Eigen::Map<const Vec>(hessian->valuePtr(), hessian->nonZeros()).allFinite())
        throw std::domain_error("IPCContactEnergy: non-finite Hessian assembly");
    return value;
}
double IPCContactEnergy::feasible_step(const ConstVecRef& x, const ConstVecRef& dx) const {
    check_x(x, num_dofs());
    check_x(dx, num_dofs());
    impl_->check_geometry(x);
    impl_->check_planes(x);
    // Check initial proximity even when no swept candidate would survive filtering.
    for (const auto& p : impl_->active(x))
        if (!(impl_->squared_distance(points(x, p.ids), p) > options_.min_distance * options_.min_distance))
            throw std::domain_error("IPCContactEnergy: contact distance must exceed min_distance");
    auto initial = impl_->candidates(x, nullptr, 0, true);
    impl_->check_intersections(x, initial);
    const Vec delta = dx;
    auto swept = impl_->candidates(x, &delta, 0.5 * options_.min_distance);
    std::vector<double> steps(swept.vf.size() + swept.ee.size());
    // Every query keeps the full [0, 1] interval, independent of other results.
    tbb::parallel_for(std::size_t(0), steps.size(), [&](std::size_t i) {
        steps[i] = i < swept.vf.size() ? impl_->advance(x, dx, swept.vf[i], false)
                                       : impl_->advance(x, dx, impl_->edge_ids(swept.ee[i - swept.vf.size()]), true);
    });
    double step = 1;
    for (double candidate : steps)
        step = std::min(step, candidate);
    for (const auto& p : impl_->planes)
        for (int v = 0; v < num_vertices(); ++v) {
            const double start = p.normal.dot(x.segment<3>(3 * v) - p.point),
                         velocity = p.normal.dot(dx.segment<3>(3 * v));
            if (!std::isfinite(velocity))
                throw std::domain_error("IPCContactEnergy: non-finite plane motion");
            if (velocity < 0 && start + velocity <= options_.min_distance)
                step = std::min(step, 0.8 * (start - options_.min_distance) / (-velocity));
        }
    return step;
}
} // namespace litepgo::contact
