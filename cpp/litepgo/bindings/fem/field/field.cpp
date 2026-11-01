#include "litepgo/fem/field/field.hpp"
#include "litepgo/bindings/bindings.hpp"

#include <nanobind/eigen/dense.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::fem::binding {
namespace {
template <int Dim> void bind_field(nb::module_& m, const char* name) {
    using Field = FieldT<Dim>;
    nb::class_<Field>(m, name, "Vector-valued field, affine in its coefficients; gradients flatten by column.")
        .def_prop_ro("num_components", &Field::num_components)
        .def_prop_ro("num_dofs", &Field::num_dofs)
        .def("eval_value", &Field::eval_value, "xi"_a)
        .def("eval_dvalue_dcoeffs", &Field::eval_dvalue_dcoeffs, "xi"_a)
        .def("eval_dvalue_dxi", &Field::eval_dvalue_dxi, "xi"_a)
        .def("eval_dgrad_xi_dcoeffs", &Field::eval_dgrad_xi_dcoeffs, "xi"_a)
        .def("eval_d2value_dxi2", &Field::eval_d2value_dxi2, "xi"_a)
        .def("eval_dhess_xi_dcoeffs", &Field::eval_dhess_xi_dcoeffs, "xi"_a);
}
} // namespace

void bind_fem_field_field(nb::module_& m) {
    bind_field<2>(m, "Field2D");
    bind_field<3>(m, "Field3D");
}
} // namespace litepgo::fem::binding
