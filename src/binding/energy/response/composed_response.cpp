#include "litepgo/energy/response/composed_response.hpp"
#include "bindings.hpp"
#include "litepgo/dof/constant_dof_map.hpp"
#include "litepgo/dof/element_dof_map.hpp"
#include "litepgo/dof/stencil_dof_map.hpp"
#include "litepgo/dof/vertex_dof_map.hpp"
#include "litepgo/energy/response/elementwise_response.hpp"
#include "litepgo/energy/response/exp_response.hpp"
#include <nanobind/stl/shared_ptr.h>
#include <utility>
namespace nb = nanobind;
using namespace nb::literals;
namespace litepgo::energy::binding {
template <class T> struct ResponseOwnerGC {
    static int traverse(PyObject* self, visitproc visit, void* arg) {
        Py_VISIT(Py_TYPE(self));
        if (!nb::inst_ready(self))
            return 0;
        for (const auto& child : nb::inst_ptr<T>(self)->children_) {
            auto owner = nb::find(child.get());
            Py_VISIT(owner.ptr());
        }
        return 0;
    }
    static int clear(PyObject* self) {
        if (nb::inst_ready(self)) {
            auto owners = std::exchange(nb::inst_ptr<T>(self)->children_, {});
        }
        return 0;
    }
    static PyType_Slot* slots() {
        static PyType_Slot slots[] = {{Py_tp_traverse, reinterpret_cast<void*>(traverse)},
                                      {Py_tp_clear, reinterpret_cast<void*>(clear)},
                                      {0, nullptr}};
        return slots;
    }
};
void bind_composed_response(nb::module_& m) {
    nb::class_<ComposedResponse, Response>(m, "ComposedResponse",
                                           nb::type_slots(ResponseOwnerGC<ComposedResponse>::slots()))
        .def(nb::init<std::shared_ptr<Response>, std::shared_ptr<Response>>(), "outer"_a, "inner"_a);
    auto element = nb::class_<ElementwiseResponse, Response>(
        m, "ElementwiseResponse", nb::type_slots(ResponseOwnerGC<ElementwiseResponse>::slots()));
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
