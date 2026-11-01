#include "litepgo/fem/kinematics/positive_det_feasible_step.hpp"
#include "litepgo/bindings/bindings.hpp"

#include <nanobind/eigen/dense.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::fem::binding {
void bind_positive_det_feasible_step(nb::module_& m) {
    m.def("eval_positive_det_feasible_step", &eval_positive_det_feasible_step, "A"_a, "delta_A"_a,
          "Safe bound in [0, 1] keeping det(A + t delta_A) > 0 on the whole segment.");
}
} // namespace litepgo::fem::binding
