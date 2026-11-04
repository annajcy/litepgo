#pragma once
#include <memory>
#include <utility>
namespace litepgo::energy {
class ParametricEnergy;
namespace detail {
// Exposes native ownership edges to binding GC; not a numerical interface.
struct ParametricView {
    explicit ParametricView(std::shared_ptr<ParametricEnergy> model) : model_(std::move(model)) {}
    virtual ~ParametricView() = default;
    std::shared_ptr<ParametricEnergy> model_;
};
} // namespace detail
} // namespace litepgo::energy
