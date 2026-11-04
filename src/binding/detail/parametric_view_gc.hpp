#pragma once
#include "litepgo/energy/parametric/parametric_energy.hpp"
#include "litepgo/energy/parametric/parametric_view.hpp"
#include <nanobind/nanobind.h>
#include <utility>

namespace litepgo::energy::binding {
template <class Base> struct ParametricViewGC {
    static int traverse(PyObject* self, visitproc visit, void* arg) {
        Py_VISIT(Py_TYPE(self));
        if (!nanobind::inst_ready(self))
            return 0;
        const auto* view = dynamic_cast<const detail::ParametricView*>(nanobind::inst_ptr<Base>(self));
        if (view) {
            auto model = nanobind::find(view->model_.get());
            Py_VISIT(model.ptr());
        }
        return 0;
    }
    static int clear(PyObject* self) {
        if (!nanobind::inst_ready(self))
            return 0;
        auto* view = dynamic_cast<detail::ParametricView*>(nanobind::inst_ptr<Base>(self));
        if (view) {
            auto owner = std::exchange(view->model_, {});
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
} // namespace litepgo::energy::binding
