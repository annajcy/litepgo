#pragma once

#include <nanobind/nanobind.h>

// Each bound module fills the _core submodule of the same name from
// litepgo::<module>::binding::bind.
namespace litepgo::io::binding {
void bind(nanobind::module_& m);
} // namespace litepgo::io::binding

namespace litepgo::energy::binding {
void bind(nanobind::module_& m);
} // namespace litepgo::energy::binding

namespace litepgo::linalg::binding {
void bind(nanobind::module_& m);
} // namespace litepgo::linalg::binding

namespace litepgo::solver::binding {
void bind(nanobind::module_& m);
} // namespace litepgo::solver::binding
