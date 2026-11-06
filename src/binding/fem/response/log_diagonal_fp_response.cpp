#include "litepgo/fem/response/log_diagonal_fp_response.hpp"
#include "bindings.hpp"
namespace nb = nanobind;
using namespace nb::literals;
namespace litepgo::fem::binding {
void bind_log_diagonal_fp_response(nb::module_& m) {
    nb::class_<LogDiagonalFpResponse, energy::Response>(m, "LogDiagonalFpResponse")
        .def(nb::init<int>(), "num_stretches"_a = 3);
}
} // namespace litepgo::fem::binding
