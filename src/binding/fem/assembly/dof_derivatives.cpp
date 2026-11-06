#include "litepgo/fem/assembly/dof_derivatives.hpp"
#include "bindings.hpp"

#include <nanobind/eigen/dense.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::fem::binding {
void bind_dof_derivatives(nb::module_& m) {
    m.def("eval_gradient_wrt_dofs", &eval_gradient_wrt_dofs, "dvalue_dcoeffs"_a, "gradient"_a,
          "Chain rule for a gradient with respect to a field value (matrices flatten by column).");
    m.def("eval_hessian_wrt_dofs", &eval_hessian_wrt_dofs, "dvalue_dcoeffs"_a, "hessian"_a,
          "Chain rule for a Hessian with respect to a field value (exact for affine fields).");
}
} // namespace litepgo::fem::binding
