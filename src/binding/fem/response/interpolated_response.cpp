#include "litepgo/fem/response/interpolated_response.hpp"
#include "bindings.hpp"
#include "detail/templates.hpp"
#include "litepgo/fem/basis/1d/cubic_hermite_basis.hpp"
#include "litepgo/fem/basis/1d/linear_lagrange_basis.hpp"
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
template <class B> void bind(nb::module_& m, const char* basis_name) {
    using R = InterpolatedResponse<B>;
    const std::string name = std::string("InterpolatedResponse[") + basis_name + "]";
    nb::class_<R, energy::Response>(m, name.c_str())
        .def(nb::init<Eigen::MatrixXd>(), "coefficients"_a,
             "R(xi) = coefficients.T @ N(xi). Owns finite coefficient snapshots; derivatives are with respect to "
             "reference coordinates, not coefficients. Rows follow basis order, columns are output components.");
    bindings::register_instance(m, "InterpolatedResponse", nb::make_tuple(nb::type<B>()), nb::type<R>());
}
} // namespace
void bind_interpolated_response(nb::module_& m) {
    bind<LinearLagrangeBasis>(m, "LinearLagrangeBasis");
    bind<CubicHermiteBasis>(m, "CubicHermiteBasis");
    bind<TriangleP1Basis>(m, "TriangleP1Basis");
    bind<QuadHermiteBasis>(m, "QuadHermiteBasis");
    bind<TetP1Basis>(m, "TetP1Basis");
    bind<HexQ1Basis>(m, "HexQ1Basis");
    bind<HexHermiteBasis>(m, "HexHermiteBasis");
}
} // namespace litepgo::fem::binding
