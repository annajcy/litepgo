#include "litepgo/energy/response/fixed_pattern_response.hpp"
#include "bindings.hpp"
#include "detail/energy_helpers.hpp"
#include "detail/sparse_matrix.hpp"
#include <nanobind/eigen/dense.h>
namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::energy::binding {
void bind_fixed_pattern_response(nb::module_& m) {
    using linalg::binding::SparseArgument;
    nb::class_<FixedPatternResponse, Response, PyResponse<FixedPatternResponse>>(m, "FixedPatternResponse")
        .def(
            "eval_jvp_values",
            [](const FixedPatternResponse& r, const ConstVecRef& x, const ConstVecRef& d) {
                return dynamic_cast<const PyResponse<FixedPatternResponse>&>(r).default_jvp(x, d);
            },
            "x"_a, "dx"_a)
        .def(
            "eval_vjp_values",
            [](const FixedPatternResponse& r, const ConstVecRef& x, const ConstVecRef& d) {
                return dynamic_cast<const PyResponse<FixedPatternResponse>&>(r).default_vjp(x, d);
            },
            "x"_a, "gy"_a)
        .def(nb::init<const SparseArgument&, const std::optional<SparseArgument>&>(), "pattern"_a,
             "hessian_pattern"_a = nb::none(),
             "Subclass and override eval_values(x, need_jacobian). Return value or (value, J); None means infeasible.");
}
} // namespace litepgo::energy::binding
