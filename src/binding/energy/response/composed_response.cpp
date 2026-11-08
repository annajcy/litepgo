#include "litepgo/energy/response/composed_response.hpp"
#include "bindings.hpp"
#include "litepgo/dof/constant_dof_map.hpp"
#include "litepgo/dof/element_dof_map.hpp"
#include "litepgo/dof/stencil_dof_map.hpp"
#include "litepgo/dof/vertex_dof_map.hpp"
#include "litepgo/energy/response/elementwise_response.hpp"
#include "litepgo/energy/response/exp_response.hpp"
#include <nanobind/stl/shared_ptr.h>
namespace nb = nanobind;
using namespace nb::literals;
namespace litepgo::energy::binding {
void bind_composed_response(nb::module_& m) {
    nb::class_<ComposedResponse, Response>(m, "ComposedResponse")
        .def(nb::init<std::shared_ptr<Response>, std::shared_ptr<Response>>(), "outer"_a, "inner"_a);
    auto element = nb::class_<ElementwiseResponse, Response>(m, "ElementwiseResponse");
    auto bind = [&]<class Map>() {
        element.def(nb::init<const Map&, std::shared_ptr<Response>>(), "input_map"_a, "transform"_a);
    };
    bind.template operator()<dof::ConstantDofMap>();
    bind.template operator()<dof::ElementDofMap>();
    bind.template operator()<dof::VertexDofMap>();
    bind.template operator()<dof::StencilDofMap>();
    nb::class_<ExpResponse, Response>(m, "ExpResponse").def(nb::init<int>(), "size"_a);
}
} // namespace litepgo::energy::binding
