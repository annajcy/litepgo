#pragma once

#include "litepgo/contact/broad_phase/broad_phase.hpp"
#include <optional>

namespace litepgo::contact {

// Sparse hash grid, with per-call scratch and no retained query state. Auto sizing
// starts with the median positive edge/face AABB extent. Grids exceeding the
// spatial index's estimated memory budget are coarsened without dropping pairs.
// Unrepresentable coordinates and budgets too small for the input still throw.
class SpatialHash final : public BroadPhase {
  public:
    explicit SpatialHash(std::optional<double> cell_size = std::nullopt);
    std::vector<BoxPair> detect(const BroadPhaseQuery& query) const override;

  private:
    const std::optional<double> cell_size_;
};
} // namespace litepgo::contact
