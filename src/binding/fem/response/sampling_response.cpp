#include "litepgo/fem/response/sampling_response.hpp"
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
template <class B> void bind(nb::module_& m) {
    const auto key = nb::make_tuple(nb::type<B>());
    const auto add = [&](const char* name, auto factory, const char* doc) {
        const std::string instance_name =
            std::string(name) + "[" + nb::str(nb::type<B>().attr("__name__")).c_str() + "]";
        auto function = nb::cpp_function(
            [factory](const Eigen::MatrixXd& points, int components) {
                return new energy::AffineResponse(factory(points, components));
            },
            nb::scope(m), nb::name(instance_name.c_str()), "points"_a, "components"_a, nb::rv_policy::take_ownership,
            doc);
        bindings::register_instance(m, name, key, function);
    };
    add("value_sampling_response", &value_sampling_response<B>,
        "AffineResponse from row-major coefficients to values at fixed (Q,D) reference points; "
        "output order: point, component.");
    add("gradient_sampling_response", &gradient_sampling_response<B>,
        "AffineResponse from row-major coefficients to reference gradients at fixed (Q,D) points; "
        "output order: point, coordinate direction, component.");
    add("hessian_sampling_response", &hessian_sampling_response<B>,
        "AffineResponse from row-major coefficients to reference Hessian samples; output order: "
        "point, Basis upper-triangle slot, component. Mixed entries are unscaled. "
        "The response's own coefficient Hessian is zero.");
}
} // namespace
void bind_sampling_responses(nb::module_& m) {
    bind<LinearLagrangeBasis>(m);
    bind<CubicHermiteBasis>(m);
    bind<TriangleP1Basis>(m);
    bind<QuadHermiteBasis>(m);
    bind<TetP1Basis>(m);
    bind<HexQ1Basis>(m);
    bind<HexHermiteBasis>(m);
}
} // namespace litepgo::fem::binding
