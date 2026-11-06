#include "litepgo/fem/kinematics/volumetric_element.hpp"
#include "bindings.hpp"
#include "detail/fem_helpers.hpp"
#include "detail/templates.hpp"
#include "litepgo/fem/quadrature/3d/gauss_legendre_3d.hpp"
#include "litepgo/fem/quadrature/3d/tet_centroid_quadrature.hpp"

#include <nanobind/eigen/dense.h>
#include <nanobind/stl/string.h>

#include <string>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::fem::binding {
namespace {

template <class Q> void bind_volumetric_element(nb::module_& m) {
    using Element = VolumetricElement3D<Q>;
    const nb::tuple key = nb::make_tuple(nb::type<Q>());
    const std::string name = "VolumetricElement3D[" + class_name(key[0]) + "]";
    nb::class_<Element> cls(m, name.c_str());
    cls.def(nb::init<energy::Response&>(), "geometry"_a, "Snapshots physical quadrature weights.")
        .def(
            "points",
            [](const Element& element) {
                Eigen::MatrixXd points(Q::num_points, 3);
                for (int q = 0; q < Q::num_points; ++q)
                    points.row(q) = element.point(q).xi.transpose();
                return points;
            },
            "Copy of cached reference coordinates, shape (Q,3), in quadrature order.")
        .def(
            "weights",
            [](const Element& element) {
                Vec weights(Q::num_points);
                for (int q = 0; q < Q::num_points; ++q)
                    weights[q] = element.point(q).w_dV;
                return weights;
            },
            "Physical weights, including reference quadrature weights and the geometry determinant.");
    cls.def(
        "max_feasible_step",
        [](const Element& element, nb::callable step) {
            return element.max_feasible_step(
                [&](int, const typename Element::Point& point) { return nb::cast<double>(step(point.xi)); });
        },
        "step"_a, "Conservative bound over quadrature points; step(xi) returns a bound in [0,1].");
    register_instance(m, "VolumetricElement3D", key, nb::type<Element>());
}
} // namespace

void bind_fem_kinematics_volumetric_element(nb::module_& m) {
    bind_volumetric_element<TetCentroidQuadrature>(m);
    bind_volumetric_element<GaussLegendre3D<2>>(m);
    bind_volumetric_element<GaussLegendre3D<4>>(m);
}
} // namespace litepgo::fem::binding
