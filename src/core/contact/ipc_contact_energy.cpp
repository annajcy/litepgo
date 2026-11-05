#include "litepgo/contact/ipc_contact_energy.hpp"
#include "litepgo/contact/broad_phase/detail/candidates.hpp"
#include "litepgo/contact/detail/distance.hpp"
#include "litepgo/contact/detail/distance_derivatives.hpp"
#include "litepgo/contact/detail/pair_weights.hpp"

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
    double weight = 1;
    int count() const {
        return epsilon > 0 ? 4 : feature.n;
    }
    double mollifier_ratio(const Points& p) const {
        if (epsilon <= 0)
            return 1;
        const auto n = cross(sub(p[1], p[0]), sub(p[3], p[2]));
        return dot(n, n) / epsilon;
    }
    auto key() const {
        return std::tuple{epsilon > 0, feature.n, feature.line_line, ids};
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
    std::vector<double> vertex_areas, edge_areas;
    int nv;
    IPCContactOptions options;
    std::shared_ptr<const BroadPhase> broad_phase_;
    PairWeights pair_weights;

    Impl(const mesh::TriangleMesh& surface, IPCContactOptions o, std::shared_ptr<const BroadPhase> broad_phase,
         std::optional<ContactPairWeights> weights)
        : edges(contact_edges(surface.topology)), nv(surface.topology.num_vertices()), options(o),
          broad_phase_(std::move(broad_phase)),
          pair_weights(std::move(weights), nv, edges.size(), surface.topology.num_cells()) {
        if (!broad_phase_)
            throw std::invalid_argument("IPCContactEnergy requires a non-null broad phase");
        std::set<std::array<int, 3>> unique;
        std::vector<bool> used(nv, false);
        for (int i = 0; i < surface.topology.num_cells(); ++i) {
            auto f = surface.topology.cell_vertices(i);
            auto sorted = f;
            std::sort(sorted.begin(), sorted.end());
            if (!unique.insert(sorted).second)
                throw std::invalid_argument("IPCContactEnergy: duplicate face");
            faces.push_back(f);
            for (int vertex : f)
                used[vertex] = true;
        }
        if (std::find(used.begin(), used.end(), false) != used.end())
            throw std::invalid_argument("IPCContactEnergy: unused surface vertex");
        for (auto e : edges)
            edge_lengths_squared.push_back(
                (surface.positions.segment<3>(3 * e[0]) - surface.positions.segment<3>(3 * e[1])).squaredNorm());
        try {
            check_geometry(surface.positions);
        } catch (const std::domain_error&) {
            throw std::invalid_argument("IPCContactEnergy: degenerate or unrepresentable rest triangle");
        }
        if (options.use_area_weighting) {
            vertex_areas.assign(nv, 0.0);
            edge_areas.assign(edges.size(), 0.0);
            for (auto f : faces) {
                const V a = surface.positions.segment<3>(3 * f[1]) - surface.positions.segment<3>(3 * f[0]),
                        b = surface.positions.segment<3>(3 * f[2]) - surface.positions.segment<3>(3 * f[0]);
                const double third_area = a.cross(b).norm() / 6;
                for (int j = 0; j < 3; ++j) {
                    vertex_areas[f[j]] += third_area;
                    const std::array<int, 2> edge{std::min(f[j], f[(j + 1) % 3]), std::max(f[j], f[(j + 1) % 3])};
                    edge_areas[std::lower_bound(edges.begin(), edges.end(), edge) - edges.begin()] += third_area;
                }
            }
            for (const auto* areas : {&vertex_areas, &edge_areas})
                for (double area : *areas)
                    if (!std::isfinite(area) || area <= 0)
                        throw std::invalid_argument("IPCContactEnergy: unrepresentable rest area weight");
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
    Candidates candidates(const ConstVecRef& x, const Vec* dx, double inflation, bool intersections = false) const {
        return broad_phase(x, dx, inflation, edges, faces, intersections, *broad_phase_);
    }
    Ids edge_ids(const Ids& pair) const {
        auto a = edges[pair[0]], b = edges[pair[1]];
        return {a[0], a[1], b[0], b[1]};
    }
    Ids face_ids(const Ids& pair) const {
        const auto f = faces[pair[1]];
        return {pair[0], f[0], f[1], f[2]};
    }
    std::vector<Pair> active(const ConstVecRef& x) const {
        check_x(x, 3 * nv);
        check_geometry(x);
        auto candidates_ = candidates(x, nullptr, 0.5 * (options.dhat + options.min_distance));
        const double reach = options.dhat + options.min_distance;
        std::vector<Pair> pairs;
        for (auto indices : candidates_.vf) {
            const double multiplier = pair_weights.vertex_face(indices[0], indices[1]);
            if (multiplier == 0)
                continue;
            const auto ids = face_ids(indices);
            auto p = points(x, ids);
            auto f = point_triangle(p);
            if (distance(p, f) < reach * reach) {
                Pair pair = reduced(ids, f);
                pair.weight =
                    checked_pair_weight(multiplier * (options.use_area_weighting ? vertex_areas[ids[0]] / 4 : 1.0));
                pairs.push_back(pair);
            }
        }
        for (auto indices : candidates_.ee) {
            const double multiplier = pair_weights.edge_edge(indices[0], indices[1]);
            if (multiplier == 0)
                continue;
            auto ids = edge_ids(indices);
            auto p = points(x, ids);
            auto f = edge_edge(p);
            if (distance(p, f) >= reach * reach)
                continue;
            double epsilon = 1e-3 * edge_lengths_squared[indices[0]] * edge_lengths_squared[indices[1]];
            if (!std::isfinite(epsilon) || epsilon <= 0)
                throw std::domain_error("IPCContactEnergy: unrepresentable mollifier threshold");
            auto n = cross(sub(p[1], p[0]), sub(p[3], p[2]));
            Pair pair = dot(n, n) < epsilon ? Pair{ids, f, epsilon} : reduced(ids, f);
            pair.weight = checked_pair_weight(
                multiplier *
                (options.use_area_weighting ? edge_areas[indices[0]] / 4 + edge_areas[indices[1]] / 4 : 1.0));
            pairs.push_back(pair);
        }
        std::sort(pairs.begin(), pairs.end(), [](const Pair& a, const Pair& b) { return a.key() < b.key(); });
        std::vector<Pair> merged;
        for (const auto& pair : pairs) {
            if (!merged.empty() && merged.back().key() == pair.key())
                merged.back().weight = checked_pair_weight(merged.back().weight + pair.weight);
            else
                merged.push_back(pair);
        }
        return merged;
    }
    double potential(const Points& p, const Pair& pair, LocalGradient* gradient = nullptr,
                     LocalHessian* hessian = nullptr) const {
        if (gradient)
            gradient->setZero();
        if (hessian)
            hessian->setZero();
        const double h = options.dhat * (options.dhat + 2 * options.min_distance);
        const double d = distance(p, pair.feature) - options.min_distance * options.min_distance;
        if (!(d > 0) || !std::isfinite(d))
            throw std::domain_error("IPCContactEnergy: contact distance must exceed min_distance");
        if (d >= h)
            return 0;
        const double log_d = std::log(d / h), delta = d - h;
        const double b = -delta * delta * log_d;
        const double z = pair.mollifier_ratio(p), m = z < 1 ? z * (2 - z) : 1;
        if (gradient || hessian) {
            LocalGradient gd;
            LocalHessian Hd;
            distance_derivatives(p, pair.feature, gd, hessian ? &Hd : nullptr);
            const double first = -2 * delta * log_d - delta * delta / d;
            const LocalGradient gb = first * gd;
            if (gradient)
                *gradient = m * gb;
            if (hessian) {
                const double ratio = h / d;
                const double second = (ratio + 2) * ratio - 2 * log_d - 3;
                *hessian = m * (first * Hd + second * gd * gd.transpose());
            }
            if (z < 1) {
                LocalGradient gc;
                LocalHessian Hc;
                cross_squarednorm_derivatives(p, gc, hessian ? &Hc : nullptr);
                const double first_m = 2 * (1 - z) / pair.epsilon;
                const LocalGradient gm = first_m * gc;
                if (gradient)
                    *gradient += b * gm;
                if (hessian) {
                    // Product rule includes both mixed outer products; the rest threshold is constant.
                    *hessian += b * (first_m * Hc - (2 / pair.epsilon / pair.epsilon) * gc * gc.transpose()) +
                                gm * gb.transpose() + gb * gm.transpose();
                }
            }
            if (gradient)
                *gradient *= pair.weight;
            if (hessian)
                *hessian *= pair.weight;
        }
        return pair.weight * b * m;
    }
    void check_intersections(const ConstVecRef& x, const Candidates& c) const {
        // Disabled candidates bypass the clearance check, but coplanar overlaps must still be rejected.
        const auto check_touch = [&](const Ids& ids, bool edge) {
            const auto p = points(x, ids);
            if (!(distance(p, edge ? edge_edge(p) : point_triangle(p)) > 0))
                throw std::domain_error("IPCContactEnergy: surface intersects itself");
        };
        for (const auto& pair : c.vf)
            if (pair_weights.vertex_face(pair[0], pair[1]) == 0)
                check_touch(face_ids(pair), false);
        for (const auto& pair : c.ee)
            if (pair_weights.edge_edge(pair[0], pair[1]) == 0)
                check_touch(edge_ids(pair), true);
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
                                   std::shared_ptr<const BroadPhase> broad_phase,
                                   std::optional<ContactPairWeights> pair_weights)
    : DynamicPatternEnergy(checked_size(surface), SpMat(surface.positions.size(), surface.positions.size())),
      rest_positions_(surface.positions), options_(checked_options(options)),
      impl_(std::make_unique<const Impl>(surface, options_, std::move(broad_phase), std::move(pair_weights))) {}
IPCContactEnergy::~IPCContactEnergy() = default;
int IPCContactEnergy::num_active_pairs(const ConstVecRef& x) const {
    return static_cast<int>(impl_->active(x).size());
}
double IPCContactEnergy::min_active_distance(const ConstVecRef& x) const {
    double d = INFINITY;
    for (const auto& pair : impl_->active(x))
        d = std::min(d, distance(points(x, pair.ids), pair.feature));
    return std::sqrt(d);
}
ContactSnapshot IPCContactEnergy::contact_snapshot(const ConstVecRef& x) const {
    const auto pairs = impl_->active(x);
    const auto count = static_cast<Eigen::Index>(pairs.size());
    ContactSnapshot result{x.reshaped<Eigen::RowMajor>(num_vertices(), 3),
                           {},
                           ContactSnapshot::VertexIds(count, 4),
                           Vec(count),
                           Vec(count),
                           Vec(count)};
    result.types.reserve(pairs.size());
    for (Eigen::Index i = 0; i < count; ++i) {
        const auto& pair = pairs[static_cast<std::size_t>(i)];
        result.types.emplace_back(pair.epsilon > 0 || pair.feature.line_line ? "EE"
                                  : pair.feature.n == 2                      ? "VV"
                                  : pair.feature.n == 3                      ? "VE"
                                                                             : "VF");
        for (int k = 0; k < 4; ++k)
            result.vertex_ids(i, k) = pair.ids[k];
        const auto p = points(x, pair.ids);
        result.distances[i] = std::sqrt(distance(p, pair.feature));
        result.weights[i] = pair.weight;
        const double z = pair.mollifier_ratio(p);
        result.mollifiers[i] = z < 1 ? z * (2 - z) : 1;
    }
    return result;
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
    auto pairs = impl_->active(x);
    double value = 0;
    // Each task owns one slot; only the fixed-order assembly below reduces values.
    std::vector<double> values(pairs.size());
    std::vector<LocalGradient> gradients(gradient ? pairs.size() : 0);
    std::vector<LocalHessian> hessians(hessian ? pairs.size() : 0);
    tbb::parallel_for(std::size_t(0), pairs.size(), [&](std::size_t index) {
        const auto& pair = pairs[index];
        auto p = points(x, pair.ids);
        auto* g = gradient ? &gradients[index] : nullptr;
        auto* H = hessian ? &hessians[index] : nullptr;
        values[index] = impl_->potential(p, pair, g, H);
        if ((g && !g->allFinite()) || (H && !H->allFinite()))
            throw std::domain_error("IPCContactEnergy: non-finite derivatives");
        if (hessian) {
            Eigen::MatrixXd local = H->topLeftCorner(3 * pair.count(), 3 * pair.count());
            if (kind == energy::HessianKind::Projected) {
                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(local);
                if (solver.info() != Eigen::Success)
                    throw std::domain_error("IPCContactEnergy: Hessian projection failed");
                local = solver.eigenvectors() * solver.eigenvalues().cwiseMax(0).asDiagonal() *
                        solver.eigenvectors().transpose();
            }
            H->topLeftCorner(3 * pair.count(), 3 * pair.count()) = local;
        }
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
        // Like dof::EnergyAssembler, each task owns complete CSC columns and gradient DOFs.
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
                    if (gradient)
                        g += gradients[index].segment<3>(3 * b);
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
                                        hessians[index](3 * a + r, 3 * b + c);
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
    // Check initial proximity even when no swept candidate would survive filtering.
    for (const auto& p : impl_->active(x))
        if (!(distance(points(x, p.ids), p.feature) > options_.min_distance * options_.min_distance))
            throw std::domain_error("IPCContactEnergy: contact distance must exceed min_distance");
    auto initial = impl_->candidates(x, nullptr, 0, true);
    impl_->check_intersections(x, initial);
    const Vec delta = dx;
    auto swept = impl_->candidates(x, &delta, 0.5 * options_.min_distance);
    std::vector<double> steps(swept.vf.size() + swept.ee.size());
    // Every query keeps the full [0, 1] interval, independent of other results.
    tbb::parallel_for(std::size_t(0), steps.size(), [&](std::size_t i) {
        const bool vf = i < swept.vf.size();
        const auto pair = vf ? swept.vf[i] : swept.ee[i - swept.vf.size()];
        const double multiplier =
            vf ? impl_->pair_weights.vertex_face(pair[0], pair[1]) : impl_->pair_weights.edge_edge(pair[0], pair[1]);
        steps[i] = multiplier == 0 ? 1 : impl_->advance(x, dx, vf ? impl_->face_ids(pair) : impl_->edge_ids(pair), !vf);
    });
    double step = 1;
    for (double candidate : steps)
        step = std::min(step, candidate);
    return step;
}
} // namespace litepgo::contact
