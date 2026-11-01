#include "litepgo/energy/response/fixed_pattern_response.hpp"
#include "litepgo/bindings/bindings.hpp"
#include "litepgo/bindings/detail/energy_helpers.hpp"
#include "litepgo/bindings/detail/sparse_matrix.hpp"
#include <nanobind/eigen/dense.h>
namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::energy::binding {
void bind_fixed_pattern_response(nb::module_& m) {
    using linalg::binding::SparseArgument;
    nb::class_<FixedPatternResponse, Response, PyResponse<FixedPatternResponse>>(m, "FixedPatternResponse")
        .def(nb::init<const SparseArgument&>(), "pattern"_a,
             "Subclass and override eval_values(x, need_jacobian). Return value or (value, J); None means infeasible.");
}
} // namespace litepgo::energy::binding
