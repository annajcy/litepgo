#include "litepgo/bindings/bindings.hpp"
#include "litepgo/core/build_info.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <utility>

namespace nb = nanobind;

NB_MODULE(_core, m) {
    m.doc() = "litepgo native extension";

    m.def("build_info", [] {
        const litepgo::BuildInfo info = litepgo::build_info();
        nb::dict result;
        result["version"] = info.version;
        result["eigen_version"] = info.eigen_version;
        result["tbb_header_version"] = info.tbb_header_version;
        result["tbb_runtime_version"] = info.tbb_runtime_version;
        result["sparse_backends"] = info.sparse_backends;
        result["ipc_toolkit_version"] = info.ipc_toolkit_version;
        return result;
    });

    // Bases come first: fem uses mesh and dof types, and its energies derive from energy.Energy.
    for (const auto& [name, bind] :
         {std::pair{"mesh", &litepgo::mesh::binding::bind}, std::pair{"dof", &litepgo::dof::binding::bind},
          std::pair{"linalg", &litepgo::linalg::binding::bind}, std::pair{"energy", &litepgo::energy::binding::bind},
          std::pair{"fem", &litepgo::fem::binding::bind}, std::pair{"solver", &litepgo::solver::binding::bind}}) {
        nb::module_ submodule = m.def_submodule(name);
        bind(submodule);
    }
#if defined(LITEPGO_HAS_CONTACT)
    nb::module_ contact = m.def_submodule("contact");
    litepgo::contact::binding::bind(contact);
#endif
}
