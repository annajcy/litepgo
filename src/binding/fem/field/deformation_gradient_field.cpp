#include "litepgo/fem/field/deformation_gradient_field.hpp"
#include "bindings.hpp"

#include <nanobind/eigen/dense.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::fem::binding {
void bind_deformation_gradient_field(nb::module_& m) {
    nb::class_<DeformationGradientField3D, Field3D>(m, "DeformationGradientField3D")
        .def(nb::init<const Field3D&, const Field3D&>(), "u_field"_a, "X_field"_a, nb::keep_alive<1, 2>(),
             nb::keep_alive<1, 3>(), "Column-flattened F = I + du/dX; keeps both fields alive.");
}
} // namespace litepgo::fem::binding
