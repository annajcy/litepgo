#include "litepgo/fem/response/deformation_gradient_response.hpp"
#include "bindings.hpp"
#include <nanobind/stl/shared_ptr.h>
namespace nb = nanobind;
using namespace nb::literals;
namespace litepgo::fem::binding {
void bind_deformation_gradient_response(nb::module_& m) {
    nb::class_<DeformationGradientResponse, energy::Response>(m, "DeformationGradientResponse")
        .def(nb::init<std::shared_ptr<energy::Response>, std::shared_ptr<energy::Response>>(), "displacement"_a,
             "geometry"_a,
             "Spatial xi -> vec(I + Du DX^-1), packed column-major. Owns child responses, which must stay fixed. "
             "Both children must map R3 to R3 and support Hessians. Exact spatial Jacobians; no Hessian support.");
}
} // namespace litepgo::fem::binding
