#pragma once

#include <Eigen/Core>
#include <array>
#include <cstddef>
#include <span>
#include <utility>
#include <vector>

namespace litepgo::contact {

enum class PrimitiveKind { Vertex, Edge, Face };

struct AABB {
    Eigen::Vector3d lo, hi;
    PrimitiveKind kind;
};

// Only contact-relevant combinations are supported; VV, VE and FF are excluded.
struct PairKindMask {
    bool vertex_face = false;
    bool edge_edge = false;
    bool edge_face = false;
    bool allows(PrimitiveKind a, PrimitiveKind b) const {
        if (a > b)
            std::swap(a, b);
        return (vertex_face && a == PrimitiveKind::Vertex && b == PrimitiveKind::Face) ||
               (edge_edge && a == PrimitiveKind::Edge && b == PrimitiveKind::Edge) ||
               (edge_face && a == PrimitiveKind::Edge && b == PrimitiveKind::Face);
    }
};

struct BroadPhaseQuery {
    // Inclusive, already inflated/outward-rounded bounds. No NaNs; lo <= hi.
    // Infinities are allowed because finite extrema can round outward to infinity.
    std::span<const AABB> boxes;
    PairKindMask allowed_pairs;
};
using BoxPair = std::array<std::size_t, 2>;

class BroadPhase {
  public:
    virtual ~BroadPhase() = default;
    // Return input-box indices, never indices into a reordered scratch array.
    // Must include every overlapping distinct pair allowed by the mask. Conservative
    // false positives, reversed pairs and duplicates are permitted. Contact normalizes
    // them and rejects invalid indices. Missing overlaps violate the safety contract.
    // Implementations must support concurrent const calls; retain no query references.
    virtual std::vector<BoxPair> detect(const BroadPhaseQuery& query) const = 0;
};

} // namespace litepgo::contact
