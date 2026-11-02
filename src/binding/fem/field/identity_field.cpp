#include "litepgo/fem/field/identity_field.hpp"
#include "bindings.hpp"

#include <nanobind/eigen/dense.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::fem::binding {
void bind_identity_field(nb::module_& m) {
    nb::class_<IdentityField2D, Field2D>(m, "IdentityField2D")
        .def(nb::init<Eigen::VectorXd>(), "coeffs"_a, "A constant field whose coefficients are its value.");
    nb::class_<IdentityField3D, Field3D>(m, "IdentityField3D")
        .def(nb::init<Eigen::VectorXd>(), "coeffs"_a, "A constant field whose coefficients are its value.");
}
} // namespace litepgo::fem::binding
