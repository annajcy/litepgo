#include "litepgo/contact/detail/broad_phase.hpp"

#include <gtest/gtest.h>
#include <random>
#include <tbb/global_control.h>

namespace litepgo::contact::test {
namespace {
using Edges = std::vector<std::array<int, 2>>;
using Faces = std::vector<std::array<int, 3>>;

// Independent all-pairs oracle: no sorting, sweep axis, or parallel candidate discovery.
detail::Candidates brute_force(const Vec& x, const Vec* dx, double inflation, const Edges& edges, const Faces& faces,
                               bool intersections) {
    auto bounds = [&](const auto& ids) {
        std::array<Eigen::Vector3d, 2> b{Eigen::Vector3d::Constant(INFINITY), Eigen::Vector3d::Constant(-INFINITY)};
        for (int v : ids)
            for (int k = 0; k < 3; ++k) {
                b[0][k] = std::min(b[0][k], x[3 * v + k]);
                b[1][k] = std::max(b[1][k], x[3 * v + k]);
                if (dx) {
                    const double end = x[3 * v + k] + (*dx)[3 * v + k];
                    b[0][k] = std::min(b[0][k], end);
                    b[1][k] = std::max(b[1][k], end);
                }
            }
        for (int k = 0; k < 3; ++k) {
            b[0][k] = std::nextafter(b[0][k] - inflation, -INFINITY);
            b[1][k] = std::nextafter(b[1][k] + inflation, INFINITY);
        }
        return b;
    };
    auto overlaps = [&](const auto& a, const auto& b) {
        for (int u : a)
            for (int v : b)
                if (u == v)
                    return false;
        const auto ba = bounds(a), bb = bounds(b);
        for (int k = 0; k < 3; ++k)
            if (ba[1][k] < bb[0][k] || bb[1][k] < ba[0][k])
                return false;
        return true;
    };
    detail::Candidates result;
    for (int v = 0; v < x.size() / 3; ++v)
        for (auto f : faces)
            if (overlaps(std::array{v}, f))
                result.vf.push_back({v, f[0], f[1], f[2]});
    for (int e = 0; e < static_cast<int>(edges.size()); ++e) {
        for (int f = e + 1; f < static_cast<int>(edges.size()); ++f)
            if (overlaps(edges[e], edges[f]))
                result.ee.push_back({e, f, -1, -1});
        if (intersections)
            for (int f = 0; f < static_cast<int>(faces.size()); ++f)
                if (overlaps(edges[e], faces[f]))
                    result.ef.push_back({e, f, -1, -1});
    }
    for (auto* list : {&result.vf, &result.ee, &result.ef})
        std::sort(list->begin(), list->end());
    return result;
}

void compare(const Vec& x, const Vec* dx, double inflation, const Edges& edges, const Faces& faces,
             bool intersections) {
    const auto expected = brute_force(x, dx, inflation, edges, faces, intersections);
    for (int threads : {1, 4, 8}) {
        SCOPED_TRACE(threads);
        const tbb::global_control control(tbb::global_control::max_allowed_parallelism, threads);
        const auto actual = detail::broad_phase(x, dx, inflation, edges, faces, intersections);
        EXPECT_EQ(actual.vf, expected.vf);
        EXPECT_EQ(actual.ee, expected.ee);
        EXPECT_EQ(actual.ef, expected.ef);
    }
}
} // namespace

TEST(ContactBroadPhase, MatchesBruteForceForAnisotropicDegenerateAndSweptBoxes) {
    std::mt19937 random(7319);
    std::uniform_real_distribution<double> unit(-1, 1);
    Edges edges;
    Faces faces;
    for (int i = 0; i < 96; i += 3) {
        faces.push_back({i, i + 1, i + 2});
        edges.insert(edges.end(), {{i, i + 1}, {i + 1, i + 2}, {i, i + 2}});
    }
    // Include shared-vertex primitives as well as disconnected triangles.
    faces.push_back({0, 3, 6});
    edges.push_back({0, 3});
    for (int trial = 0; trial < 12; ++trial) {
        SCOPED_TRACE(trial);
        Vec x(288), dx(288);
        for (int v = 0; v < 96; ++v)
            for (int k = 0; k < 3; ++k) {
                const double scale = k == trial % 3 ? 100 : 0.01;
                x[3 * v + k] = trial >= 9 ? 0 : unit(random) * scale;
                dx[3 * v + k] = unit(random) * scale;
                // Zero-width slabs and coincident endpoints exercise sweep ties.
                if (trial >= 6 && k == (trial + 1) % 3)
                    x[3 * v + k] = dx[3 * v + k] = 0;
            }
        for (const Vec* motion : std::array<const Vec*, 2>{nullptr, &dx})
            for (double inflation : {0.0, 0.005 / 2, 0.003 / 2, (0.005 + 0.003) / 2})
                for (bool intersections : {false, true})
                    compare(x, motion, inflation, edges, faces, intersections);
    }
}

TEST(ContactBroadPhase, RetainsTouchingRoundedAndSweptPairs) {
    const Faces faces{{0, 1, 2}};
    const Edges edges{{0, 1}, {1, 2}, {0, 2}, {3, 4}};
    Vec x(15);
    x << 0, 0, 0, 1, 0, 0, 0, 1, 0, 0.25, 0.25, 1, 0.25, 0.25, 2;
    Vec dx = Vec::Zero(15);
    dx[11] = -2; // Crosses the face, though neither endpoint touches it.
    auto swept = detail::broad_phase(x, &dx, 0, edges, faces, true);
    EXPECT_NE(std::find(swept.vf.begin(), swept.vf.end(), std::array{3, 0, 1, 2}), swept.vf.end());
    EXPECT_NE(std::find(swept.ef.begin(), swept.ef.end(), std::array{3, 0, -1, -1}), swept.ef.end());
    compare(x, &dx, 0, edges, faces, true);
    for (double height : {0.0, std::nextafter(0.0, 1.0), 0.125, std::nextafter(0.125, 1.0)}) {
        x[11] = x[14] = height;
        const double inflation = height < 0.1 ? 0 : 0.0625;
        const auto candidates = detail::broad_phase(x, nullptr, inflation, edges, faces, true);
        EXPECT_NE(std::find(candidates.vf.begin(), candidates.vf.end(), std::array{3, 0, 1, 2}), candidates.vf.end());
        compare(x, nullptr, inflation, edges, faces, true);
    }
}

TEST(ContactBroadPhase, EmptyAndUnrepresentableBounds) {
    compare(Vec(), nullptr, 0, {}, {}, true);
    Vec x = Vec::Constant(3, std::numeric_limits<double>::max());
    Vec dx = x;
    EXPECT_THROW(detail::broad_phase(x, &dx, 0, {}, {}), std::domain_error);
    EXPECT_THROW(detail::broad_phase(x, nullptr, std::numeric_limits<double>::max(), {}, {}), std::domain_error);
    compare(x, nullptr, 0, {}, {}, true); // Outward rounding may reach infinity.
}
} // namespace litepgo::contact::test
