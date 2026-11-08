#include "bindings.hpp"
#include "litepgo/fem/basis/basis.hpp"
#include <nanobind/eigen/dense.h>

namespace nb = nanobind;

namespace litepgo::fem::binding {
namespace {
template <int Dim> void bind_info(nb::module_& m, const char* name) {
    using Info = CoefficientInfo<Dim>;
    nb::class_<Info>(m, name, "Reference sample defining one interpolation coefficient.")
        .def_prop_ro(
            "point", [](const Info& info) { return info.point; },
            "Reference coordinate: a scalar in 1D, a vector in 2D/3D.")
        .def_prop_ro(
            "derivative_order",
            [](const Info& info) {
                nb::list orders;
                for (int order : info.derivative_order)
                    orders.append(order);
                return nb::tuple(orders);
            },
            "Per-axis reference derivative orders as a tuple. All zeros select the function value.");
}
} // namespace

void bind_coefficient_info(nb::module_& m) {
    bind_info<1>(m, "CoefficientInfo1D");
    bind_info<2>(m, "CoefficientInfo2D");
    bind_info<3>(m, "CoefficientInfo3D");
}
} // namespace litepgo::fem::binding
