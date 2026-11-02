#include "litepgo/fem/field/interpolated_field.hpp"
#include "bindings.hpp"
#include "detail/templates.hpp"
#include "litepgo/fem/basis/2d/quad_hermite_basis.hpp"
#include "litepgo/fem/basis/2d/triangle_p1_basis.hpp"
#include "litepgo/fem/basis/3d/hex_hermite_basis.hpp"
#include "litepgo/fem/basis/3d/hex_q1_basis.hpp"
#include "litepgo/fem/basis/3d/tet_p1_basis.hpp"

#include <nanobind/eigen/dense.h>

#include <string>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::fem::binding {
namespace {
using bindings::register_instance;

template <class B, int Dim> void bind_interpolated_field(nb::module_& m, const char* basis_name) {
    using Field = InterpolatedFieldT<B, Dim>;
    const std::string template_name = "InterpolatedField" + std::to_string(Dim) + "D";
    const std::string name = template_name + "[" + basis_name + "]";
    nb::class_<Field, FieldT<Dim>>(m, name.c_str())
        .def(nb::init<Eigen::MatrixXd>(), "coeffs"_a, "One row per basis function, one column per component.");
    register_instance(m, template_name.c_str(), nb::make_tuple(nb::type<B>()), nb::type<Field>());
}
} // namespace

void bind_fem_field_interpolated_field(nb::module_& m) {
    bind_interpolated_field<TriangleP1Basis, 2>(m, "TriangleP1Basis");
    bind_interpolated_field<QuadHermiteBasis, 2>(m, "QuadHermiteBasis");
    bind_interpolated_field<TetP1Basis, 3>(m, "TetP1Basis");
    bind_interpolated_field<HexQ1Basis, 3>(m, "HexQ1Basis");
    bind_interpolated_field<HexHermiteBasis, 3>(m, "HexHermiteBasis");
}
} // namespace litepgo::fem::binding
