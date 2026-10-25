#pragma once

#include <nanobind/nanobind.h>

// One function per C++ directory, each filling the _core submodule of the same name.
namespace litepgo::bindings {

void bind_energy(nanobind::module_& m);
void bind_linalg(nanobind::module_& m);
void bind_solver(nanobind::module_& m);

} // namespace litepgo::bindings
