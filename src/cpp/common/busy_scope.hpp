#pragma once

#include <atomic>
#include <stdexcept>
#include <string>

namespace litepgo::detail {

// Marks an object busy for one call, so a second call that overlaps it throws
// std::logic_error instead of racing on the object's scratch state.
class BusyScope {
  public:
    BusyScope(std::atomic<bool>& busy, const char* owner) : busy_(busy) {
        if (busy_.exchange(true)) {
            throw std::logic_error(std::string(owner) + " is already in use by another call");
        }
    }
    ~BusyScope() {
        busy_.store(false);
    }
    BusyScope(const BusyScope&) = delete;
    BusyScope& operator=(const BusyScope&) = delete;

  private:
    std::atomic<bool>& busy_;
};

} // namespace litepgo::detail
