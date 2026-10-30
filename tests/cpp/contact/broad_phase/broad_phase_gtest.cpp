#include "litepgo/contact/broad_phase/detail/candidates.hpp"
#include "litepgo/contact/broad_phase/sweep_and_prune.hpp"

#include "litepgo/contact/ipc_contact_energy.hpp"
#include <cstring>
#include <future>
#include <gtest/gtest.h>
#include <mutex>
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
        const auto actual = detail::broad_phase(x, dx, inflation, edges, faces, intersections, SweepAndPrune{});
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
    auto swept = detail::broad_phase(x, &dx, 0, edges, faces, true, SweepAndPrune{});
    EXPECT_NE(std::find(swept.vf.begin(), swept.vf.end(), std::array{3, 0, 1, 2}), swept.vf.end());
    EXPECT_NE(std::find(swept.ef.begin(), swept.ef.end(), std::array{3, 0, -1, -1}), swept.ef.end());
    compare(x, &dx, 0, edges, faces, true);
    for (double height : {0.0, std::nextafter(0.0, 1.0), 0.125, std::nextafter(0.125, 1.0)}) {
        x[11] = x[14] = height;
        const double inflation = height < 0.1 ? 0 : 0.0625;
        const auto candidates = detail::broad_phase(x, nullptr, inflation, edges, faces, true, SweepAndPrune{});
        EXPECT_NE(std::find(candidates.vf.begin(), candidates.vf.end(), std::array{3, 0, 1, 2}), candidates.vf.end());
        compare(x, nullptr, inflation, edges, faces, true);
    }
}

TEST(ContactBroadPhase, EmptyAndUnrepresentableBounds) {
    compare(Vec(), nullptr, 0, {}, {}, true);
    Vec x = Vec::Constant(3, std::numeric_limits<double>::max());
    Vec dx = x;
    EXPECT_THROW(detail::broad_phase(x, &dx, 0, {}, {}, false, SweepAndPrune{}), std::domain_error);
    EXPECT_THROW(detail::broad_phase(x, nullptr, std::numeric_limits<double>::max(), {}, {}, false, SweepAndPrune{}),
                 std::domain_error);
    compare(x, nullptr, 0, {}, {}, true); // Outward rounding may reach infinity.
}

namespace {
class NoisyBroadPhase final : public BroadPhase {
  public:
    std::vector<BoxPair> detect(const BroadPhaseQuery& q) const override {
        std::vector<BoxPair> pairs;
        for (std::size_t i = 0; i < q.boxes.size(); ++i)
            for (std::size_t j = i + 1; j < q.boxes.size(); ++j) {
                // Deliberate false positives: disallowed kinds, topology neighbors and separated boxes.
                pairs.push_back({j, i});
                pairs.push_back({i, j});
            }
        pairs.push_back({0, 0});
        return pairs;
    }
};
class InvalidBroadPhase final : public BroadPhase {
  public:
    std::vector<BoxPair> detect(const BroadPhaseQuery& q) const override {
        return {{0, q.boxes.size()}};
    }
};
class RecordingBroadPhase final : public BroadPhase {
  public:
    mutable std::mutex mutex;
    mutable std::vector<std::pair<std::vector<AABB>, PairKindMask>> queries;
    std::vector<BoxPair> detect(const BroadPhaseQuery& q) const override {
        {
            std::lock_guard lock(mutex);
            queries.emplace_back(std::vector<AABB>(q.boxes.begin(), q.boxes.end()), q.allowed_pairs);
        }
        return SweepAndPrune{}.detect(q);
    }
};
mesh::TriangleMesh surface() {
    Vec x(18);
    x << 0, 0, 0, 10, 0, 0, 0, 10, 0, 2, 2, 0.5, 6, 2, 4.5, 2, 6, 4.5;
    return {x, mesh::TriangleTopology(6, {{0, 1, 2}, {3, 4, 5}})};
}
} // namespace

TEST(ContactBroadPhase, PublicSAPRespectsEveryMaskAndOriginalIndices) {
    using K = PrimitiveKind;
    std::vector<AABB> boxes;
    for (int i = 0; i < 90; ++i) {
        const double x = double((i * 17) % 31);
        boxes.push_back({Eigen::Vector3d(x, 0, 0), Eigen::Vector3d(x + 1, 1, 1), static_cast<K>(i % 3)});
    }
    const SweepAndPrune sap;
    for (int bits = 0; bits < 8; ++bits) {
        const PairKindMask mask{bool(bits & 1), bool(bits & 2), bool(bits & 4)};
        std::vector<BoxPair> expected;
        for (std::size_t i = 0; i < boxes.size(); ++i)
            for (std::size_t j = i + 1; j < boxes.size(); ++j) {
                auto a = boxes[i].kind, b = boxes[j].kind;
                if (a > b)
                    std::swap(a, b);
                const bool permitted = ((bits & 1) && a == K::Vertex && b == K::Face) ||
                                       ((bits & 2) && a == K::Edge && b == K::Edge) ||
                                       ((bits & 4) && a == K::Edge && b == K::Face);
                if (permitted && boxes[i].lo[0] <= boxes[j].hi[0] && boxes[j].lo[0] <= boxes[i].hi[0])
                    expected.push_back({i, j});
            }
        for (int threads : {1, 4, 8}) {
            const tbb::global_control control(tbb::global_control::max_allowed_parallelism, threads);
            auto actual = sap.detect({boxes, mask});
            for (auto& pair : actual)
                if (pair[0] > pair[1])
                    std::swap(pair[0], pair[1]);
            std::sort(actual.begin(), actual.end());
            EXPECT_EQ(actual, expected);
        }
    }
    // Infinite singleton bounds and outward-rounded extrema must not make the axis NaN.
    for (double endpoint : std::array<double, 3>{-INFINITY, INFINITY, std::numeric_limits<double>::max()}) {
        std::vector<AABB> infinite{
            {Eigen::Vector3d::Constant(endpoint), Eigen::Vector3d::Constant(endpoint), K::Vertex},
            {Eigen::Vector3d::Constant(endpoint), Eigen::Vector3d::Constant(endpoint), K::Face}};
        const auto pairs = sap.detect({infinite, {true, false, false}});
        ASSERT_EQ(pairs.size(), 1);
        EXPECT_NE(pairs[0][0], pairs[0][1]);
    }
    boxes[0].lo[0] = NAN;
    EXPECT_THROW(sap.detect({boxes, {true, true, true}}), std::invalid_argument);
    boxes[0].lo[0] = boxes[0].hi[0] + 1;
    EXPECT_THROW(sap.detect({boxes, {true, true, true}}), std::invalid_argument);
    boxes[0].lo[0] = 0;
    boxes[0].kind = static_cast<K>(42);
    EXPECT_THROW(sap.detect({boxes, {true, true, true}}), std::invalid_argument);
}

TEST(ContactBroadPhase, InjectedBackendNormalizesCandidatesAndOwnsLifetime) {
    const auto mesh = surface();
    const IPCContactOptions options{1.0};
    IPCContactEnergy reference(mesh, options, std::make_shared<SweepAndPrune>());
    const Vec x = mesh.positions;
    reference.update_pattern(x);
    const Vec g = reference.eval_gradient(x);
    const SpMat h = reference.eval_hessian(x);
    Vec dx = Vec::Zero(x.size());
    dx.segment<3>(9) << 0, 0, -2;
    std::shared_ptr<const BroadPhase> sap = std::make_shared<SweepAndPrune>();
    std::shared_ptr<const BroadPhase> noisy = std::make_shared<NoisyBroadPhase>();
    for (const auto& backend : {sap, noisy}) {
        IPCContactEnergy energy(mesh, options, backend, {});
        energy.update_pattern(x);
        EXPECT_EQ(energy.eval_value(x), reference.eval_value(x));
        const Vec actual_g = energy.eval_gradient(x);
        const SpMat actual_h = energy.eval_hessian(x);
        ASSERT_EQ(actual_h.nonZeros(), h.nonZeros());
        EXPECT_EQ(std::memcmp(actual_h.outerIndexPtr(), h.outerIndexPtr(), sizeof(int) * (h.cols() + 1)), 0);
        EXPECT_EQ(std::memcmp(actual_h.innerIndexPtr(), h.innerIndexPtr(), sizeof(int) * h.nonZeros()), 0);
        EXPECT_EQ(std::memcmp(actual_g.data(), g.data(), sizeof(double) * g.size()), 0);
        EXPECT_EQ(std::memcmp(actual_h.valuePtr(), h.valuePtr(), sizeof(double) * h.nonZeros()), 0);
        EXPECT_EQ(energy.max_feasible_step(x, dx), reference.max_feasible_step(x, dx));
    }
    std::weak_ptr<const BroadPhase> weak = noisy;
    auto retained = std::make_unique<IPCContactEnergy>(mesh, options, noisy, std::vector<Plane>{});
    noisy.reset();
    EXPECT_FALSE(weak.expired());
    EXPECT_EQ(retained->eval_value(x), reference.eval_value(x));
    retained.reset();
    EXPECT_TRUE(weak.expired());
    IPCContactEnergy invalid(mesh, options, std::make_shared<InvalidBroadPhase>(), {});
    EXPECT_THROW(invalid.eval_value(x), std::invalid_argument);
    EXPECT_THROW(invalid.update_pattern(x), std::invalid_argument);
    EXPECT_THROW(invalid.max_feasible_step(x, dx), std::invalid_argument);
    auto solve = [&] {
        IPCContactEnergy energy(mesh, options, sap, {});
        return energy.eval_value(x);
    };
    auto first = std::async(std::launch::async, solve), second = std::async(std::launch::async, solve);
    EXPECT_EQ(first.get(), reference.eval_value(x));
    EXPECT_EQ(second.get(), reference.eval_value(x));
}

TEST(ContactBroadPhase, ContactSuppliesStaticIntersectionAndSweptQueries) {
    const auto mesh = surface();
    auto backend = std::make_shared<RecordingBroadPhase>();
    IPCContactEnergy energy(mesh, IPCContactOptions{1.0, 0.1}, backend, {});
    const Vec x = mesh.positions;
    energy.eval_value(x);
    ASSERT_EQ(backend->queries.size(), 1);
    const auto& [boxes, mask] = backend->queries.front();
    EXPECT_TRUE(mask.vertex_face);
    EXPECT_TRUE(mask.edge_edge);
    EXPECT_FALSE(mask.edge_face);
    ASSERT_EQ(boxes.size(), 14); // 6 vertices, 6 edges, 2 faces
    EXPECT_EQ(boxes[3].lo[2], std::nextafter(0.5 - 0.55, -INFINITY));
    backend->queries.clear();
    Vec dx = Vec::Zero(x.size());
    dx.segment<3>(9) << 0, 0, -2;
    energy.max_feasible_step(x, dx);
    ASSERT_EQ(backend->queries.size(), 3);
    EXPECT_FALSE(backend->queries[0].second.edge_face);
    EXPECT_TRUE(backend->queries[1].second.edge_face);
    EXPECT_FALSE(backend->queries[2].second.edge_face);
    EXPECT_EQ(backend->queries[1].first[3].lo[2], std::nextafter(0.5, -INFINITY));
    EXPECT_EQ(backend->queries[2].first[3].lo[2], std::nextafter(-1.5 - 0.05, -INFINITY));
}
} // namespace litepgo::contact::test
