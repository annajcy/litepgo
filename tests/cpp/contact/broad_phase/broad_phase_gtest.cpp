#include "litepgo/contact/broad_phase/detail/candidates.hpp"
#include "litepgo/contact/broad_phase/spatial_hash.hpp"
#include <memory>

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
        for (int f = 0; f < static_cast<int>(faces.size()); ++f)
            if (overlaps(std::array{v}, faces[f]))
                result.vf.push_back({v, f, -1, -1});
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
        const SpatialHash automatic, fixed(10.0);
        for (const BroadPhase* backend : std::array<const BroadPhase*, 2>{&automatic, &fixed}) {
            const auto actual = detail::broad_phase(x, dx, inflation, edges, faces, intersections, *backend);
            EXPECT_EQ(actual.vf, expected.vf);
            EXPECT_EQ(actual.ee, expected.ee);
            EXPECT_EQ(actual.ef, expected.ef);
        }
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
    auto swept = detail::broad_phase(x, &dx, 0, edges, faces, true, SpatialHash{});
    EXPECT_NE(std::find(swept.vf.begin(), swept.vf.end(), std::array{3, 0, -1, -1}), swept.vf.end());
    EXPECT_NE(std::find(swept.ef.begin(), swept.ef.end(), std::array{3, 0, -1, -1}), swept.ef.end());
    compare(x, &dx, 0, edges, faces, true);
    for (double height : {0.0, std::nextafter(0.0, 1.0), 0.125, std::nextafter(0.125, 1.0)}) {
        x[11] = x[14] = height;
        const double inflation = height < 0.1 ? 0 : 0.0625;
        const auto candidates = detail::broad_phase(x, nullptr, inflation, edges, faces, true, SpatialHash{});
        EXPECT_NE(std::find(candidates.vf.begin(), candidates.vf.end(), std::array{3, 0, -1, -1}), candidates.vf.end());
        compare(x, nullptr, inflation, edges, faces, true);
    }
}

TEST(ContactBroadPhase, EmptyAndUnrepresentableBounds) {
    compare(Vec(), nullptr, 0, {}, {}, true);
    Vec x = Vec::Constant(3, std::numeric_limits<double>::max());
    Vec dx = x;
    EXPECT_THROW(detail::broad_phase(x, &dx, 0, {}, {}, false, SpatialHash{}), std::domain_error);
    EXPECT_THROW(detail::broad_phase(x, nullptr, std::numeric_limits<double>::max(), {}, {}, false, SpatialHash{}),
                 std::domain_error);
    EXPECT_THROW(detail::broad_phase(x, nullptr, 0, {}, {}, true, SpatialHash{}), std::overflow_error);
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
        return SpatialHash{}.detect(q);
    }
};
mesh::TriangleMesh surface() {
    Vec x(18);
    x << 0, 0, 0, 10, 0, 0, 0, 10, 0, 2, 2, 0.5, 6, 2, 4.5, 2, 6, 4.5;
    return {x, mesh::TriangleTopology(6, {{0, 1, 2}, {3, 4, 5}})};
}
} // namespace

TEST(ContactBroadPhase, PublicSpatialHashRespectsEveryMaskAndOriginalIndices) {
    using K = PrimitiveKind;
    std::vector<AABB> boxes;
    for (int i = 0; i < 90; ++i) {
        const double x = double((i * 17) % 31);
        boxes.push_back({Eigen::Vector3d(x, 0, 0), Eigen::Vector3d(x + 1, 1, 1), static_cast<K>(i % 3)});
    }
    const SpatialHash backend_default;
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
            auto actual = backend_default.detect({boxes, mask});
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
        EXPECT_THROW(backend_default.detect({infinite, {true, false, false}}), std::overflow_error);
    }
    boxes[0].lo[0] = NAN;
    EXPECT_THROW(backend_default.detect({boxes, {true, true, true}}), std::invalid_argument);
    boxes[0].lo[0] = boxes[0].hi[0] + 1;
    EXPECT_THROW(backend_default.detect({boxes, {true, true, true}}), std::invalid_argument);
    boxes[0].lo[0] = 0;
    boxes[0].kind = static_cast<K>(42);
    EXPECT_THROW(backend_default.detect({boxes, {true, true, true}}), std::invalid_argument);
}

TEST(ContactBroadPhase, InjectedBackendNormalizesCandidatesAndOwnsLifetime) {
    const auto mesh = surface();
    const IPCContactOptions options{1.0};
    IPCContactEnergy reference(mesh, options, std::make_shared<SpatialHash>());
    const Vec x = mesh.positions;
    const Vec g = reference.evaluate(x)->gradient();
    const SpMat h = reference.evaluate(x)->hessian();
    Vec dx = Vec::Zero(x.size());
    dx.segment<3>(9) << 0, 0, -2;
    std::shared_ptr<const BroadPhase> backend_default = std::make_shared<SpatialHash>();
    std::shared_ptr<const BroadPhase> noisy = std::make_shared<NoisyBroadPhase>();
    std::shared_ptr<const BroadPhase> hash = std::make_shared<SpatialHash>(0.5);
    for (const auto& backend : {backend_default, noisy, hash}) {
        IPCContactEnergy energy(mesh, options, backend);
        EXPECT_EQ(energy.evaluate(x)->value(), reference.evaluate(x)->value());
        const Vec actual_g = energy.evaluate(x)->gradient();
        const SpMat actual_h = energy.evaluate(x)->hessian();
        ASSERT_EQ(actual_h.nonZeros(), h.nonZeros());
        EXPECT_EQ(std::memcmp(actual_h.outerIndexPtr(), h.outerIndexPtr(), sizeof(int) * (h.cols() + 1)), 0);
        EXPECT_EQ(std::memcmp(actual_h.innerIndexPtr(), h.innerIndexPtr(), sizeof(int) * h.nonZeros()), 0);
        EXPECT_EQ(std::memcmp(actual_g.data(), g.data(), sizeof(double) * g.size()), 0);
        EXPECT_EQ(std::memcmp(actual_h.valuePtr(), h.valuePtr(), sizeof(double) * h.nonZeros()), 0);
        EXPECT_EQ(energy.evaluate(x)->max_feasible_step(dx), reference.evaluate(x)->max_feasible_step(dx));
    }
    std::weak_ptr<const BroadPhase> weak = noisy;
    auto retained = std::make_unique<IPCContactEnergy>(mesh, options, noisy);
    noisy.reset();
    EXPECT_FALSE(weak.expired());
    EXPECT_EQ(retained->evaluate(x)->value(), reference.evaluate(x)->value());
    retained.reset();
    EXPECT_TRUE(weak.expired());
    IPCContactEnergy invalid(mesh, options, std::make_shared<InvalidBroadPhase>());
    EXPECT_THROW(invalid.evaluate(x)->value(), std::invalid_argument);
    EXPECT_THROW(invalid.evaluate(x)->hessian(), std::invalid_argument);
    EXPECT_THROW(invalid.evaluate(x)->max_feasible_step(dx), std::invalid_argument);
    auto solve = [&] {
        IPCContactEnergy energy(mesh, options, backend_default);
        return energy.evaluate(x)->value();
    };
    auto first = std::async(std::launch::async, solve), second = std::async(std::launch::async, solve);
    EXPECT_EQ(first.get(), reference.evaluate(x)->value());
    EXPECT_EQ(second.get(), reference.evaluate(x)->value());
}

TEST(ContactBroadPhase, ContactSuppliesStaticIntersectionAndSweptQueries) {
    const auto mesh = surface();
    auto backend = std::make_shared<RecordingBroadPhase>();
    IPCContactEnergy energy(mesh, IPCContactOptions{1.0, 0.1}, backend);
    const Vec x = mesh.positions;
    energy.evaluate(x)->value();
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
    energy.evaluate(x)->max_feasible_step(dx);
    ASSERT_EQ(backend->queries.size(), 3);
    EXPECT_FALSE(backend->queries[0].second.edge_face);
    EXPECT_TRUE(backend->queries[1].second.edge_face);
    EXPECT_FALSE(backend->queries[2].second.edge_face);
    EXPECT_EQ(backend->queries[1].first[3].lo[2], std::nextafter(0.5, -INFINITY));
    EXPECT_EQ(backend->queries[2].first[3].lo[2], std::nextafter(-1.5 - 0.05, -INFINITY));
}

namespace {
std::vector<BoxPair> canonical(std::vector<BoxPair> pairs) {
    for (auto& pair : pairs)
        if (pair[0] > pair[1])
            std::swap(pair[0], pair[1]);
    std::sort(pairs.begin(), pairs.end());
    return pairs; // Do not unique: duplicate emission is a test failure.
}
std::vector<BoxPair> box_oracle(const BroadPhaseQuery& q) {
    std::vector<BoxPair> pairs;
    for (std::size_t i = 0; i < q.boxes.size(); ++i)
        for (std::size_t j = i + 1; j < q.boxes.size(); ++j) {
            const auto& a = q.boxes[i];
            const auto& b = q.boxes[j];
            if (q.allowed_pairs.allows(a.kind, b.kind) && !(a.lo.array() > b.hi.array()).any() &&
                !(b.lo.array() > a.hi.array()).any())
                pairs.push_back({i, j});
        }
    return pairs;
}
} // namespace
TEST(SpatialHash, MatchesBruteForceWithNegativeBoundaryDegenerateAndSweptBoxes) {
    std::mt19937 rng(31973);
    std::uniform_real_distribution<double> coord(-3, 3), width(0, 1);
    std::vector<AABB> boxes;
    for (int i = 0; i < 180; ++i) {
        Eigen::Vector3d lo, hi;
        for (int k = 0; k < 3; ++k) {
            lo[k] = i % 4 == 0 ? std::floor(coord(rng)) : coord(rng);
            hi[k] = lo[k] + (i % 5 == 0 ? 0 : width(rng));
            if (i % 7 == 0) { // Full swept bounds in both signs.
                const double motion = coord(rng);
                lo[k] += std::min(0.0, motion);
                hi[k] += std::max(0.0, motion);
            }
        }
        boxes.push_back({lo, hi, static_cast<PrimitiveKind>(i % 3)});
    }
    // Contact exactly on a cell boundary and just either side of zero.
    for (double t : {-1.0, 0.0, 1.0, std::nextafter(0.0, -1.0), std::nextafter(1.0, 2.0)})
        for (auto kind : {PrimitiveKind::Vertex, PrimitiveKind::Edge, PrimitiveKind::Face})
            boxes.push_back({Eigen::Vector3d::Constant(t), Eigen::Vector3d::Constant(t), kind});
    for (int bits = 0; bits < 8; ++bits) {
        const BroadPhaseQuery q{boxes, {bool(bits & 1), bool(bits & 2), bool(bits & 4)}};
        const auto expected = box_oracle(q);
        for (auto h : std::array<std::optional<double>, 4>{std::nullopt, 0.5, 1.0, 2.0}) {
            SpatialHash hash(h);
            for (int threads : {1, 4, 8}) {
                const tbb::global_control control(tbb::global_control::max_allowed_parallelism, threads);
                EXPECT_EQ(canonical(hash.detect(q)), expected);
            }
        }
    }
    SpatialHash shared;
    const BroadPhaseQuery q{boxes, {true, true, true}};
    auto run = [&] { return canonical(shared.detect(q)); };
    auto a = std::async(std::launch::async, run), b = std::async(std::launch::async, run);
    EXPECT_EQ(a.get(), box_oracle(q));
    EXPECT_EQ(b.get(), box_oracle(q));
}
TEST(SpatialHash, RejectsUnrepresentableGridsAndValidatesCompleteInput) {
    using V = Eigen::Vector3d;
    using K = PrimitiveKind;
    for (double h : std::array<double, 4>{0, -1, INFINITY, NAN})
        EXPECT_THROW(SpatialHash{h}, std::invalid_argument);
    EXPECT_TRUE(SpatialHash{}.detect({{}, {true, true, true}}).empty());
    std::vector<std::vector<AABB>> cases{
        {{V::Constant(-INFINITY), V::Constant(INFINITY), K::Face}, {V::Zero(), V::Zero(), K::Vertex}},
        {{V::Constant(-0x1p63), V::Constant(-0x1p63), K::Face},
         {V::Constant(-0x1p63), V::Constant(-0x1p63), K::Vertex}},
        {{V::Constant(0x1p63), V::Constant(0x1p63), K::Face}, {V::Constant(0x1p63), V::Constant(0x1p63), K::Vertex}},
        {{V::Constant(-0x1p63), V::Constant(0x1p62), K::Face}, {V::Zero(), V::Zero(), K::Vertex}},
        {{V::Zero(), V::Constant(100), K::Face}, {V::Zero(), V::Constant(100), K::Edge}}, // Total record budget.
        {{V::Zero(), V::Constant(10000), K::Face}, {V::Zero(), V::Zero(), K::Vertex}},    // Product budget.
        {{V::Zero(), V::Zero(), K::Face}, {V::Zero(), V::Zero(), K::Vertex}}              // No positive auto extent.
    };
    EXPECT_THROW(SpatialHash(1).detect({cases[0], {true, true, true}}), std::overflow_error);
    EXPECT_EQ(canonical(SpatialHash(1).detect({cases[1], {true, true, true}})),
              box_oracle({cases[1], {true, true, true}}));
    EXPECT_THROW(SpatialHash(1).detect({cases[2], {true, true, true}}), std::overflow_error);
    for (int i : {3, 4, 5})
        EXPECT_THROW(SpatialHash(1).detect({cases[i], {true, true, true}}), std::length_error);
    EXPECT_EQ(canonical(SpatialHash{}.detect({cases[6], {true, true, true}})),
              box_oracle({cases[6], {true, true, true}}));
    for (const auto& first : cases) {
        auto boxes = first;
        boxes.push_back({V::Constant(NAN), V::Zero(), K::Vertex});
        EXPECT_THROW(SpatialHash(1).detect({boxes, {true, true, true}}), std::invalid_argument);
        boxes.back() = {V::Zero(), V::Zero(), static_cast<K>(42)};
        EXPECT_THROW(SpatialHash{}.detect({boxes, {true, true, true}}), std::invalid_argument);
        boxes.back() = {V::Ones(), V::Zero(), K::Vertex};
        EXPECT_THROW(SpatialHash(1).detect({boxes, {true, true, true}}), std::invalid_argument);
    }
}
TEST(SpatialHash, OwnerCellEmitsPairsOnlyOnceAcrossManyCells) {
    std::vector<AABB> boxes(20, {Eigen::Vector3d::Zero(), Eigen::Vector3d(99, 0, 0), PrimitiveKind::Edge});
    const BroadPhaseQuery query{boxes, {false, true, false}};
    EXPECT_EQ(canonical(SpatialHash(1).detect(query)), box_oracle(query));
}
} // namespace litepgo::contact::test
