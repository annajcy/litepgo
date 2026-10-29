#include "litepgo/bindings/bindings.hpp"
#include "litepgo/bindings/templates.hpp"
#include "litepgo/fem/kinematics/basis_dof_transform.hpp"
#include "litepgo/fem/kinematics/positive_det_feasible_step.hpp"
#include "litepgo/fem/kinematics/triangle_mid_edge_kinematics.hpp"
#include "litepgo/fem/kinematics/volumetric_element.hpp"
#include "litepgo/fem/local/dof_derivatives.hpp"
#include "litepgo/fem/local/local_energy.hpp"
#include "litepgo/fem/material/shell/koiter_stvk.hpp"
#include "litepgo/fem/material/shell/shell_material_local_energy.hpp"
#include "litepgo/fem/material/volumetric/neo_hookean.hpp"
#include "litepgo/fem/material/volumetric/volumetric_material_local_energy.hpp"
#include "litepgo/fem/quadrature/3d/gauss_legendre_3d.hpp"
#include "litepgo/fem/quadrature/3d/tet_centroid_quadrature.hpp"

#include <nanobind/eigen/dense.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include <array>
#include <optional>
#include <string>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::fem::binding {
namespace {
using bindings::register_instance;

std::string class_name(nb::handle cls) {
    return nb::cast<std::string>(cls.attr("__name__"));
}

// value, gradient and Hessian, with None for each derivative not requested.
nb::tuple value_and_derivatives(double value, bool gradient, const Eigen::VectorXd& g, bool hessian,
                                const Eigen::MatrixXd& H) {
    return nb::make_tuple(value, gradient ? nb::cast(g) : nb::none(), hessian ? nb::cast(H) : nb::none());
}

template <int Dim> void bind_local_energy(nb::module_& m, const char* name) {
    using Local = LocalEnergyT<Dim>;
    using Coordinate = typename Local::Coordinate;
    nb::class_<Local>(m, name, "Pointwise density with derivatives with respect to one active coefficient block.")
        .def_prop_ro("num_dofs", &Local::num_dofs)
        .def_prop_ro("has_gradient", &Local::has_gradient)
        .def_prop_ro("has_hessian", &Local::has_hessian)
        .def("eval_value", &Local::eval_value, "xi"_a)
        .def("eval_gradient", &Local::eval_gradient, "xi"_a)
        .def("eval_hessian", &Local::eval_hessian, "xi"_a)
        .def(
            "accumulate",
            [](const Local& local, const Coordinate& xi, double weight, bool gradient, bool hessian) {
                Eigen::VectorXd g = Eigen::VectorXd::Zero(gradient ? local.num_dofs() : 0);
                Eigen::MatrixXd H =
                    Eigen::MatrixXd::Zero(hessian ? local.num_dofs() : 0, hessian ? local.num_dofs() : 0);
                const double value = local.accumulate(xi, weight, gradient ? &g : nullptr, hessian ? &H : nullptr);
                return value_and_derivatives(value, gradient, g, hessian, H);
            },
            "xi"_a, "weight"_a, "gradient"_a = false, "hessian"_a = false,
            "Weighted (value, gradient or None, Hessian or None) at xi; C++ adds these into caller buffers.");
}

template <template <class> class Local, class M> void bind_volumetric_local(nb::module_& m, const char* template_name) {
    using Energy = Local<M>;
    const nb::tuple key = nb::make_tuple(nb::type<M>());
    const std::string name = std::string(template_name) + "[" + class_name(key[0]) + "]";
    nb::class_<Energy, VolumetricMaterialLocalEnergy3D<M>>(m, name.c_str())
        .def(nb::init<const Field3D&, const Field3D&, const Field3D&>(), "F"_a, "Fp"_a, "theta"_a,
             nb::keep_alive<1, 2>(), nb::keep_alive<1, 3>(), nb::keep_alive<1, 4>(),
             "Borrows the three fields and keeps them alive.");
    register_instance(m, template_name, key, nb::type<Energy>());
}

template <class M> void bind_volumetric_locals(nb::module_& m) {
    using Base = VolumetricMaterialLocalEnergy3D<M>;
    const nb::tuple key = nb::make_tuple(nb::type<M>());
    const std::string name = "VolumetricMaterialLocalEnergy3D[" + class_name(key[0]) + "]";
    nb::class_<Base, LocalEnergy3D>(m, name.c_str(), "Material density whose derivatives follow one input field.")
        .def("eval_feasible_step", &Base::eval_feasible_step, "xi"_a, "delta_coeffs"_a,
             "Largest alpha in [0, 1] keeping the material inputs valid along the active coefficients.");
    register_instance(m, "VolumetricMaterialLocalEnergy3D", key, nb::type<Base>());
    bind_volumetric_local<VolumetricMaterialLocalEnergyF3D, M>(m, "VolumetricMaterialLocalEnergyF3D");
    bind_volumetric_local<VolumetricMaterialLocalEnergyFp3D, M>(m, "VolumetricMaterialLocalEnergyFp3D");
    bind_volumetric_local<VolumetricMaterialLocalEnergyTheta3D, M>(m, "VolumetricMaterialLocalEnergyTheta3D");
}

template <class M> void bind_shell_local(nb::module_& m) {
    using Energy = ShellMaterialLocalEnergy2D<M>;
    const nb::tuple key = nb::make_tuple(nb::type<M>());
    const std::string name = "ShellMaterialLocalEnergy2D[" + class_name(key[0]) + "]";
    nb::class_<Energy, LocalEnergy2D>(m, name.c_str())
        .def(nb::init<const TriangleMidEdgeRestData&, Eigen::MatrixXd, double, Eigen::VectorXd>(), "rest"_a,
             "u_local"_a, "thickness"_a, "theta"_a,
             "Density per reference area for stencil displacements u_local (one row per stencil vertex).");
    register_instance(m, "ShellMaterialLocalEnergy2D", key, nb::type<Energy>());
}

// Physical displacement DOFs (one row per vertex) <-> basis coefficients (one row per basis function).
template <class B> void bind_dof_transform(nb::module_& m) {
    using Transform = BasisDofTransform<B>;
    const nb::tuple key = nb::make_tuple(nb::type<B>());
    const std::string name = "BasisDofTransform[" + class_name(key[0]) + "]";
    nb::class_<Transform> cls(m, name.c_str());
    cls.attr("dofs_per_vertex") = Transform::dofs_per_vertex;
    cls.def(nb::init<const Field3D&>(), "X_field"_a,
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
    // max_feasible_step is a C++ member template over local energies with feasible steps.
    (cls.def(
         "max_feasible_step",
         [](const Element& element, const VolumetricMaterialLocalEnergy3D<Materials>& local,
            const Eigen::VectorXd& delta_coeffs) { return element.max_feasible_step(local, delta_coeffs); },
         "local_energy"_a, "delta_coeffs"_a, "Smallest feasible step over the quadrature points."),
     ...);
    register_instance(m, "VolumetricElement3D", key, nb::type<Element>());
}
} // namespace

void bind_local(nb::module_& m) {
    bind_local_energy<2>(m, "LocalEnergy2D");
    bind_local_energy<3>(m, "LocalEnergy3D");
    bind_volumetric_locals<NeoHookean>(m);
    bind_shell_local<KoiterStVK>(m);

    bind_dof_transform<TetP1Basis>(m);
    bind_dof_transform<HexQ1Basis>(m);
    bind_dof_transform<HexHermiteBasis>(m);

    bind_volumetric_element<TetCentroidQuadrature, NeoHookean>(m);
    bind_volumetric_element<GaussLegendre3D<2>, NeoHookean>(m);
    bind_volumetric_element<GaussLegendre3D<4>, NeoHookean>(m);

    nb::class_<TriangleMidEdgeKinematics>(m, "TriangleMidEdgeKinematics")
        .def_ro("a", &TriangleMidEdgeKinematics::a, "First fundamental form.")
        .def_ro("b", &TriangleMidEdgeKinematics::b, "Second fundamental form, b_ab = n.dot(x_,ab).");
    m.def(
        "eval_triangle_mid_edge_kinematics",
        [](const Eigen::MatrixXd& x, const std::array<int, 3>& opposite, bool gradient, bool hessian) {
            Eigen::MatrixXd dab_dx;
            std::array<Eigen::MatrixXd, 6> d2ab_dx2;
            const auto kinematics = eval_triangle_mid_edge_kinematics(x, opposite, gradient ? &dab_dx : nullptr,
                                                                      hessian ? &d2ab_dx2 : nullptr);
            return nb::make_tuple(kinematics, gradient ? nb::cast(dab_dx) : nb::none(),
                                  hessian ? nb::cast(d2ab_dx2) : nb::none());
        },
        "x"_a, "opposite"_a, "gradient"_a = false, "hessian"_a = false,
        "(kinematics, dab_dx or None, d2ab_dx2 or None); rows/components use (a11,a12,a22,b11,b12,b22) and x "
        "flattens by row.");
    m.def("eval_triangle_mid_edge_feasible_step", &eval_triangle_mid_edge_feasible_step, "x"_a, "dx"_a, "opposite"_a);
    m.def("eval_positive_det_feasible_step", &eval_positive_det_feasible_step, "A"_a, "delta_A"_a,
          "Safe bound in [0, 1] keeping det(A + t delta_A) > 0 on the whole segment.");
    m.def("eval_gradient_wrt_dofs", &eval_gradient_wrt_dofs, "dvalue_dcoeffs"_a, "gradient"_a,
          "Chain rule for a gradient with respect to a field value (matrices flatten by column).");
    m.def("eval_hessian_wrt_dofs", &eval_hessian_wrt_dofs, "dvalue_dcoeffs"_a, "hessian"_a,
          "Chain rule for a Hessian with respect to a field value (exact for affine fields).");
}

} // namespace litepgo::fem::binding
