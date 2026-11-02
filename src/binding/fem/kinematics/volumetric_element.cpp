#include "litepgo/fem/kinematics/volumetric_element.hpp"
#include "bindings.hpp"
#include "detail/fem_helpers.hpp"
#include "detail/templates.hpp"
#include "litepgo/fem/local/local_energy.hpp"
#include "litepgo/fem/local/local_response.hpp"
#include "litepgo/fem/material/volumetric/neo_hookean.hpp"
#include "litepgo/fem/material/volumetric/volumetric_material_local_energy.hpp"
#include "litepgo/fem/quadrature/3d/gauss_legendre_3d.hpp"
#include "litepgo/fem/quadrature/3d/tet_centroid_quadrature.hpp"

#include <nanobind/eigen/dense.h>
#include <nanobind/stl/string.h>

#include <string>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::fem::binding {
namespace {

template <class Q, class... Materials> void bind_volumetric_element(nb::module_& m) {
    using Element = VolumetricElement3D<Q>;
    const nb::tuple key = nb::make_tuple(nb::type<Q>());
    const std::string name = "VolumetricElement3D[" + class_name(key[0]) + "]";
    nb::class_<Element> cls(m, name.c_str());
    cls.def(nb::init<const Field3D&>(), "X_field"_a, "Caches the reference geometry of X_field; keeps no reference.")
        .def(
            "eval",
            [](const Element& element, const LocalEnergy3D& local, bool gradient, bool hessian) {
                Eigen::VectorXd g;
                Eigen::MatrixXd H;
                const double value = element.eval(local, gradient ? &g : nullptr, hessian ? &H : nullptr);
                return value_and_derivatives(value, gradient, g, hessian, H);
            },
            "local_energy"_a, "gradient"_a = false, "hessian"_a = false,
            "Integrated (value, gradient or None, Hessian or None) in the local energy's DOF order.");
    cls.def(
        "eval",
        [](const Element& element, const LocalResponse3D& local, bool jacobian) {
            Eigen::MatrixXd J;
            const Eigen::VectorXd value = element.eval(local, jacobian ? &J : nullptr);
            return nb::make_tuple(value, jacobian ? nb::cast(J) : nb::none());
        },
        "local_response"_a, "jacobian"_a = false, "Integrated (value, Jacobian or None) in local response order.");
    // max_feasible_step is a C++ member template over local energies with feasible steps.
    (cls.def(
         "max_feasible_step",
         [](const Element& element, const VolumetricMaterialLocalEnergyF3D<Materials>& local,
            const Eigen::VectorXd& delta_coeffs) { return element.max_feasible_step(local, delta_coeffs); },
         "local_energy"_a, "delta_coeffs"_a, "Smallest feasible step over the quadrature points."),
     ...);
    register_instance(m, "VolumetricElement3D", key, nb::type<Element>());
}
} // namespace

void bind_fem_kinematics_volumetric_element(nb::module_& m) {
    bind_volumetric_element<TetCentroidQuadrature, NeoHookean>(m);
    bind_volumetric_element<GaussLegendre3D<2>, NeoHookean>(m);
    bind_volumetric_element<GaussLegendre3D<4>, NeoHookean>(m);
}
} // namespace litepgo::fem::binding
