#pragma once
#include "detail/templates.hpp"
#include "litepgo/dof/constant_dof_map.hpp"
#include "litepgo/dof/element_dof_map.hpp"
#include "litepgo/dof/vertex_dof_map.hpp"
#include "litepgo/energy/response/mesh_response.hpp"
#include "litepgo/fem/basis/basis.hpp"
#include "litepgo/fem/local/local_response.hpp"
#include <memory>
#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <string>
#include <utility>
#include <vector>
namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::fem::binding {
using bindings::register_instance;
using Responses = std::vector<std::shared_ptr<energy::Response>>;
template <class B, int Dim> void bind_basis(nb::module_& m, const char* name) {
    using Coordinate = typename CoefficientInfo<Dim>::Coordinate;
    nb::class_<B> cls(m, name,
                      "Scalar shape functions evaluated by static methods at one reference coordinate xi. "
                      "num_basis_functions counts scalar functions, independently of field components. "
                      "All output rows follow coefficient-row order. Domain-exterior coordinates extrapolate "
                      "without clamping.");
    cls.attr("num_basis_functions") = B::num_basis_functions;
    cls.def_static("coefficient_info", &B::coefficient_info, "basis_index"_a,
                   "Coefficient sample: reference point and per-axis derivative orders. All-zero orders mean a value. "
                   "Invalid indices raise IndexError.")
        .def_static(
            "eval_N", [](const Coordinate& xi) { return Eigen::VectorXd(B::eval_N(xi)); }, "xi"_a,
            "Shape-function values, shape (num_basis_functions,).")
        .def_static(
            "eval_dN_dxi", [](const Coordinate& xi) { return Eigen::MatrixXd(B::eval_dN_dxi(xi)); }, "xi"_a,
            "Reference-coordinate derivatives, shape (num_basis_functions, D); column j differentiates xi[j].")
        .def_static(
            "eval_d2N_dxi2", [](const Coordinate& xi) { return Eigen::MatrixXd(B::eval_d2N_dxi2(xi)); }, "xi"_a,
            "Second reference-coordinate derivatives, shape (num_basis_functions, D*(D+1)/2). "
            "Columns: (11,12,22) in 2D or (11,12,13,22,23,33) in 3D; mixed entries are stored once, unscaled.");
}

template <class Q> void bind_quadrature(nb::module_& m, const char* name) {
    nb::class_<Q> cls(m, name,
                      "Fixed reference quadrature with static points() and weights() methods. "
                      "Weights include reference-domain measure; physical geometry is applied by the integrator.");
    cls.attr("num_points") = Q::num_points;
    cls.def_static(
           "points", [] { return Eigen::MatrixXd(Q::points()); },
           "Copy of reference coordinates, shape (num_points, D), including (num_points, 1) in 1D.")
        .def_static(
            "weights", [] { return Eigen::VectorXd(Q::weights()); },
            "Copy of reference integration weights, shape (num_points,). Entry q corresponds to points()[q].");
}

inline nb::tuple value_and_derivatives(double value, bool gradient, const Eigen::VectorXd& g, bool hessian,
                                       const Eigen::MatrixXd& H) {
    return nb::make_tuple(value, gradient ? nb::cast(g) : nb::none(), hessian ? nb::cast(H) : nb::none());
}

inline std::string class_name(nb::handle cls) {
    return nb::cast<std::string>(cls.attr("__name__"));
}

template <class ParameterMap, class Force, class Cls>
void bind_force_init(Cls& cls, const char* fixed_name, const char* doc) {
    cls.def(
        "__init__",
        [](Force* self, const ParameterMap& parameter_map, const dof::VertexDofMap& displacement_map,
           const Responses& rest_position_fields, Vec displacement, const Responses& fixed, int batch_size) {
            new (self) Force(parameter_map, displacement_map, rest_position_fields, std::move(displacement), fixed,
                             batch_size);
        },
        "parameter_map"_a, "displacement_map"_a, "rest_position_fields"_a, "displacement"_a, nb::arg(fixed_name),
        "batch_size"_a = 256, doc);
}

template <class Force, class M, class B, class Q>
void bind_force(nb::module_& m, const char* template_name, const char* fixed_name, const char* doc) {
    const nb::tuple key = nb::make_tuple(nb::type<M>(), nb::type<B>(), nb::type<Q>());
    const std::string name = std::string(template_name) + "[" + class_name(key[0]) + ", " + class_name(key[1]) + ", " +
                             class_name(key[2]) + "]";
    auto cls = nb::class_<Force, energy::MeshResponse>(m, name.c_str());
    bind_force_init<dof::ConstantDofMap, Force>(cls, fixed_name, doc);
    bind_force_init<dof::ElementDofMap, Force>(cls, fixed_name, doc);
    register_instance(m, template_name, key, nb::type<Force>());
}
} // namespace litepgo::fem::binding
