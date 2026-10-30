#pragma once

#include "litepgo/contact/broad_phase/broad_phase.hpp"

namespace litepgo::contact {

// Immutable backend; all scratch is per call. Chooses the widest scene axis,
// scans overlapping intervals with TBB, and rejects other-axis separation.
class SweepAndPrune final : public BroadPhase {
  public:
    std::vector<BoxPair> detect(const BroadPhaseQuery& query) const override;
};

} // namespace litepgo::contact
