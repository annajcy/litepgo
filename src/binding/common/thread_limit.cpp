#include "bindings.hpp"
#include <tbb/global_control.h>

#include <memory>
#include <stdexcept>

namespace litepgo::common::binding {
namespace {
struct TbbThreadLimit {
    std::unique_ptr<tbb::global_control> control;

    explicit TbbThreadLimit(size_t threads) {
        if (threads == 0) {
            throw std::invalid_argument("threads must be positive");
        }
        control = std::make_unique<tbb::global_control>(tbb::global_control::max_allowed_parallelism, threads);
    }
};
} // namespace

void bind_thread_limit(nanobind::module_& m) {
    nanobind::class_<TbbThreadLimit>(m, "_TbbThreadLimit")
        .def(nanobind::init<size_t>())
        .def("close", [](TbbThreadLimit& limit) { limit.control.reset(); });
    m.def("_active_tbb_thread_limit",
          [] { return tbb::global_control::active_value(tbb::global_control::max_allowed_parallelism); });
}
} // namespace litepgo::common::binding
