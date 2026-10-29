#include "litepgo/bindings/bindings.hpp"
#include "litepgo/bindings/templates.hpp"
#include "litepgo/fem/basis/2d/quad_hermite_basis.hpp"
#include "litepgo/fem/basis/2d/triangle_p1_basis.hpp"
#include "litepgo/fem/basis/3d/hex_hermite_basis.hpp"
#include "litepgo/fem/basis/3d/hex_q1_basis.hpp"
#include "litepgo/fem/basis/3d/tet_p1_basis.hpp"
#include "litepgo/fem/field/deformation_gradient_field.hpp"
#include "litepgo/fem/field/identity_field.hpp"
#include "litepgo/fem/field/interpolated_field.hpp"
#include "litepgo/fem/kinematics/triangle_mid_edge_kinematics.hpp"
#include "litepgo/fem/material/shell/koiter_stvk.hpp"
#include "litepgo/fem/material/volumetric/neo_hookean.hpp"
#include "litepgo/fem/quadrature/2d/gauss_legendre_2d.hpp"
#include "litepgo/fem/quadrature/2d/triangle_centroid_quadrature.hpp"
#include "litepgo/fem/quadrature/3d/gauss_legendre_3d.hpp"
#include "litepgo/fem/quadrature/3d/tet_centroid_quadrature.hpp"

#include <nanobind/eigen/dense.h>
#include <nanobind/stl/array.h>

#include <string>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::fem::binding {
namespace {
using bindings::register_instance;

// Basis classes are template-argument tags with their static evaluations.
template <class B, int Dim> void bind_basis(nb::module_& m, const char* name) {
    using Coordinate = Eigen::Matrix<double, Dim, 1>;
    nb::class_<B> cls(m, name);
    cls.attr("num_basis_functions") = B::num_basis_functions;
    cls.def_static(
           "eval_N", [](const Coordinate& xi) { return Eigen::VectorXd(B::eval_N(xi)); }, "xi"_a)
        .def_static(
            "eval_dN_dxi", [](const Coordinate& xi) { return Eigen::MatrixXd(B::eval_dN_dxi(xi)); }, "xi"_a)
        .def_static(
            "eval_d2N_dxi2", [](const Coordinate& xi) { return Eigen::MatrixXd(B::eval_d2N_dxi2(xi)); }, "xi"_a,
            "Second derivatives, upper triangle in row order: (11,12,22) or (11,12,13,22,23,33).")
        .def_static(
            "reference_node", [](int basis_index) { return Coordinate(B::reference_node(basis_index)); },
            "basis_index"_a);
}

// Quadrature classes are template-argument tags with their points and weights.
template <class Q> void bind_quadrature(nb::module_& m, const char* name) {
    nb::class_<Q> cls(m, name);
    cls.attr("num_points") = Q::num_points;
    cls.def_static(
           "points", [] { return Eigen::MatrixXd(Q::points()); }, "One reference point per row.")
        .def_static("weights", [] { return Eigen::VectorXd(Q::weights()); });
}

template <int PointsPerAxis> void bind_gauss_legendre(nb::module_& m) {
    const std::string n = std::to_string(PointsPerAxis);
    const std::string name2 = "GaussLegendre2D[" + n + "]", name3 = "GaussLegendre3D[" + n + "]";
    bind_quadrature<GaussLegendre2D<PointsPerAxis>>(m, name2.c_str());
    bind_quadrature<GaussLegendre3D<PointsPerAxis>>(m, name3.c_str());
    register_instance(m, "GaussLegendre2D", nb::make_tuple(PointsPerAxis), nb::type<GaussLegendre2D<PointsPerAxis>>());
    register_instance(m, "GaussLegendre3D", nb::make_tuple(PointsPerAxis), nb::type<GaussLegendre3D<PointsPerAxis>>());
}

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

template <class B, int Dim> void bind_interpolated_field(nb::module_& m, const char* basis_name) {
    using Field = InterpolatedFieldT<B, Dim>;
    const std::string template_name = "InterpolatedField" + std::to_string(Dim) + "D";
    const std::string name = template_name + "[" + basis_name + "]";
    nb::class_<Field, FieldT<Dim>>(m, name.c_str())
        .def(nb::init<Eigen::MatrixXd>(), "coeffs"_a, "One row per basis function, one column per component.");
    register_instance(m, template_name.c_str(), nb::make_tuple(nb::type<B>()), nb::type<Field>());
}

VolumetricMaterialInputs volumetric_inputs(const Eigen::Matrix3d& F, const Eigen::Matrix3d& Fp,
                                           const Eigen::VectorXd& theta) {
    return {F, Fp, theta};
}

// Volumetric materials are tags with their pointwise density and derivatives.
template <class M> void bind_volumetric_material(nb::module_& m, const char* name) {
    nb::class_<M> cls(m, name);
    cls.attr("num_parameters") = M::num_parameters;
#define LITEPGO_VOLUMETRIC(fn)                                                                                         \
    cls.def_static(                                                                                                    \
        #fn,                                                                                                           \
        [](const Eigen::Matrix3d& F, const Eigen::Matrix3d& Fp, const Eigen::VectorXd& theta) {                        \
            return M::fn(volumetric_inputs(F, Fp, theta));                                                             \
        },                                                                                                             \
        "F"_a, "Fp"_a, "theta"_a)
    LITEPGO_VOLUMETRIC(eval_W);
    LITEPGO_VOLUMETRIC(eval_dW_dF);
    LITEPGO_VOLUMETRIC(eval_d2W_dF2);
    LITEPGO_VOLUMETRIC(eval_dW_dFp);
    LITEPGO_VOLUMETRIC(eval_d2W_dFp2);
    LITEPGO_VOLUMETRIC(eval_dW_dtheta);
    LITEPGO_VOLUMETRIC(eval_d2W_dtheta2);
#undef LITEPGO_VOLUMETRIC
    cls.def_static(
        "eval_feasible_step",
        [](const Eigen::Matrix3d& F, const Eigen::Matrix3d& Fp, const Eigen::VectorXd& theta, const Eigen::Matrix3d& dF,
           const Eigen::Matrix3d& dFp, const Eigen::VectorXd& dtheta) {
            return M::eval_feasible_step(volumetric_inputs(F, Fp, theta), volumetric_inputs(dF, dFp, dtheta));
        },
        "F"_a, "Fp"_a, "theta"_a, "dF"_a, "dFp"_a, "dtheta"_a,
        "Largest alpha in [0, 1] keeping the inputs valid along inputs + t * increments.");
}

// Shell materials are tags with their pointwise density per reference area.
template <class M> void bind_shell_material(nb::module_& m, const char* name) {
    nb::class_<M> cls(m, name);
    cls.attr("num_parameters") = M::num_parameters;
#define LITEPGO_SHELL(fn)                                                                                              \
    cls.def_static(                                                                                                    \
        #fn,                                                                                                           \
        [](const Eigen::Matrix2d& a, const Eigen::Matrix2d& b, const Eigen::Matrix2d& a_bar,                           \
           const Eigen::Matrix2d& b_bar, double thickness,                                                             \
           const Eigen::VectorXd& theta) { return M::fn(ShellMaterialInputs{a, b, a_bar, b_bar, thickness, theta}); }, \
        "a"_a, "b"_a, "a_bar"_a, "b_bar"_a, "thickness"_a, "theta"_a)
    LITEPGO_SHELL(eval_W);
    LITEPGO_SHELL(eval_dW_dab);
    LITEPGO_SHELL(eval_d2W_dab2);
#undef LITEPGO_SHELL
}
} // namespace

void bind(nb::module_& m) {
    m.doc() = "Finite elements (cpp/litepgo/fem). Template instantiations are in _templates.";

    bind_basis<TriangleP1Basis, 2>(m, "TriangleP1Basis");
    bind_basis<QuadHermiteBasis, 2>(m, "QuadHermiteBasis");
    bind_basis<TetP1Basis, 3>(m, "TetP1Basis");
    bind_basis<HexQ1Basis, 3>(m, "HexQ1Basis");
    bind_basis<HexHermiteBasis, 3>(m, "HexHermiteBasis");

    bind_quadrature<TriangleCentroidQuadrature>(m, "TriangleCentroidQuadrature");
    bind_quadrature<TetCentroidQuadrature>(m, "TetCentroidQuadrature");
    bind_gauss_legendre<2>(m);
    bind_gauss_legendre<4>(m);

    bind_field<2>(m, "Field2D");
    bind_field<3>(m, "Field3D");
    nb::class_<IdentityField2D, Field2D>(m, "IdentityField2D")
        .def(nb::init<Eigen::VectorXd>(), "coeffs"_a, "A constant field whose coefficients are its value.");
    nb::class_<IdentityField3D, Field3D>(m, "IdentityField3D")
        .def(nb::init<Eigen::VectorXd>(), "coeffs"_a, "A constant field whose coefficients are its value.");
    bind_interpolated_field<TriangleP1Basis, 2>(m, "TriangleP1Basis");
    bind_interpolated_field<QuadHermiteBasis, 2>(m, "QuadHermiteBasis");
    bind_interpolated_field<TetP1Basis, 3>(m, "TetP1Basis");
    bind_interpolated_field<HexQ1Basis, 3>(m, "HexQ1Basis");
    bind_interpolated_field<HexHermiteBasis, 3>(m, "HexHermiteBasis");
    nb::class_<DeformationGradientField3D, Field3D>(m, "DeformationGradientField3D")
        .def(nb::init<const Field3D&, const Field3D&>(), "u_field"_a, "X_field"_a, nb::keep_alive<1, 2>(),
             nb::keep_alive<1, 3>(), "Column-flattened F = I + du/dX; keeps both fields alive.");

    bind_volumetric_material<NeoHookean>(m, "NeoHookean");
    bind_shell_material<KoiterStVK>(m, "KoiterStVK");

    nb::class_<TriangleMidEdgeRestData>(m, "TriangleMidEdgeRestData")
        .def(nb::init<Eigen::MatrixXd, std::array<int, 3>>(), "X"_a, "opposite_rows"_a,
             "Reference stencil positions (central face first) and the local row across each edge, or -1.")
        .def_ro("X_local", &TriangleMidEdgeRestData::X_local)
        .def_ro("opposite", &TriangleMidEdgeRestData::opposite)
        .def_ro("a_bar", &TriangleMidEdgeRestData::a_bar)
        .def_ro("b_bar", &TriangleMidEdgeRestData::b_bar)
        .def_ro("sqrt_det_a_bar", &TriangleMidEdgeRestData::sqrt_det_a_bar);

    bind_energies(m);
}

} // namespace litepgo::fem::binding
