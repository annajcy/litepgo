#pragma once

#include "litepgo/contact/broad_phase/broad_phase.hpp"
#include <optional>

namespace litepgo::contact {

// Sparse hash grid, with per-call scratch and no retained query state. Auto sizing
// uses the median positive edge/face AABB extent. Unrepresentable coordinates or
// more than 2^20 box-cell records fall back to SAP for the entire query.
class SpatialHash final : public BroadPhase {
  public:
    explicit SpatialHash(std::optional<double> cell_size = std::nullopt);
    std::vector<BoxPair> detect(const BroadPhaseQuery& query) const override;

  private:
    const std::optional<double> cell_size_;
};
} // namespace litepgo::contact
