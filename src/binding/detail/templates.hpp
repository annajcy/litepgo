#pragma once

#include <nanobind/nanobind.h>

namespace litepgo::bindings {

// Records a class or function as the instantiation template_name[key...] in the submodule's
// _templates dict ({template_name: {key: cls}}). Python wraps each entry so that
// Template[args] returns the bound callable, mirroring the C++ template arguments.
inline void register_instance(nanobind::module_& m, const char* template_name, const nanobind::tuple& key,
                              nanobind::handle cls) {
    if (!nanobind::hasattr(m, "_templates"))
        m.attr("_templates") = nanobind::dict();
    nanobind::dict templates = nanobind::borrow<nanobind::dict>(m.attr("_templates"));
    if (!templates.contains(template_name))
        templates[template_name] = nanobind::dict();
    nanobind::borrow<nanobind::dict>(templates[template_name])[key] = cls;
}

} // namespace litepgo::bindings
