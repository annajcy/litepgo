#include "litepgo/energy/response/dynamic_pattern_response.hpp"
#include "litepgo/bindings/bindings.hpp"
#include "litepgo/bindings/detail/energy_helpers.hpp"
#include "litepgo/bindings/detail/sparse_matrix.hpp"
#include <nanobind/eigen/dense.h>
#include <nanobind/trampoline.h>
namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::energy::binding {
namespace {
class PyDynamicResponse : public PyResponse<DynamicPatternResponse> {
  public:
    using PyResponse<DynamicPatternResponse>::PyResponse;

  private:
    SpMat pattern_at(const ConstVecRef& x) const override {
        nb::detail::ticket ticket(nb_trampoline, "pattern_at", nb::detail::str_hash("pattern_at"), true);
        return nb::cast<linalg::binding::SparseArgument>(nb_trampoline.base().attr(ticket.key)(nb::cast(Vec(x))))
            .matrix;
    }
};
} // namespace

void bind_dynamic_pattern_response(nb::module_& m) {
    using linalg::binding::SparseArgument;
    nb::class_<DynamicPatternResponse, Response, PyDynamicResponse>(m, "DynamicPatternResponse")
        .def(nb::init<const SparseArgument&>(), "initial_pattern"_a,
             "Subclass and override pattern_at(x) and eval_values(x, need_jacobian). Dimensions remain fixed.");
}
} // namespace litepgo::energy::binding
