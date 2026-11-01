#pragma once

#include <nanobind/nanobind.h>

// Each bound module fills the _core submodule of the same name from
// litepgo::<module>::binding::bind.
namespace litepgo::mesh::binding {
void bind(nanobind::module_& m);
} // namespace litepgo::mesh::binding

namespace litepgo::dof::binding {
void bind(nanobind::module_& m);
} // namespace litepgo::dof::binding

namespace litepgo::energy::binding {
void bind(nanobind::module_& m);
} // namespace litepgo::energy::binding

namespace litepgo::fem::binding {
void bind(nanobind::module_& m);
// Local energies, elements and kinematics (fem_local.cpp), and the energy template
// instantiations (fem_energy.cpp), compiled separately.
void bind_local(nanobind::module_& m);
void bind_energies(nanobind::module_& m);
} // namespace litepgo::fem::binding

namespace litepgo::linalg::binding {
void bind(nanobind::module_& m);
} // namespace litepgo::linalg::binding

namespace litepgo::contact::binding {
void bind(nanobind::module_& m);
} // namespace litepgo::contact::binding

namespace litepgo::solver::binding {
void bind(nanobind::module_& m);
} // namespace litepgo::solver::binding

namespace litepgo::spatial::binding {
void bind(nanobind::module_& m);
}

namespace litepgo::response::binding {
void bind(nanobind::module_& m);
}
