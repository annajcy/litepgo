#include "litepgo/fem/kinematics/basis_dof_transform.hpp"
#include "bindings.hpp"
#include "detail/fem_helpers.hpp"
#include "detail/templates.hpp"

#include <nanobind/eigen/dense.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include <optional>
#include <string>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::fem::binding {
namespace {

template <class B> void bind_basis_dof_transform(nb::module_& m) {
    using Transform = BasisDofTransform<B>;
    const nb::tuple key = nb::make_tuple(nb::type<B>());
    const std::string name = "BasisDofTransform[" + class_name(key[0]) + "]";
    nb::class_<Transform> cls(m, name.c_str());
    cls.attr("dofs_per_vertex") = Transform::dofs_per_vertex;
    cls.def(nb::init<energy::Response&>(), "geometry"_a,
            "Caches the scaling for one element's geometry; keeps no reference.")
        .def(
            "to_coefficients",
            [](const Transform& transform, const Eigen::MatrixXd& local) { return transform.to_coefficients(local); },
            "local"_a, "Basis coefficients from physical DOFs gathered with dofs_per_vertex DOFs per vertex.")
        .def(
            "pullback",
            [](const Transform& transform, std::optional<Eigen::VectorXd> gradient,
               std::optional<Eigen::MatrixXd> hessian) {
                transform.pullback(gradient ? &*gradient : nullptr, hessian ? &*hessian : nullptr);
                return nb::make_tuple(gradient ? nb::cast(*gradient) : nb::none(),
                                      hessian ? nb::cast(*hessian) : nb::none());
            },
            "gradient"_a = nb::none(), "hessian"_a = nb::none(),
            "(gradient or None, Hessian or None) with respect to coefficients, pulled back to physical DOFs.");
    register_instance(m, "BasisDofTransform", key, nb::type<Transform>());
}
} // namespace

void bind_fem_kinematics_basis_dof_transform(nb::module_& m) {
    bind_basis_dof_transform<TetP1Basis>(m);
    bind_basis_dof_transform<HexQ1Basis>(m);
    bind_basis_dof_transform<HexHermiteBasis>(m);
}
} // namespace litepgo::fem::binding
