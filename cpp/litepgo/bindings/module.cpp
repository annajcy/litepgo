#include "litepgo/bindings/bindings.h"
#include "litepgo/core/build_info.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

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
        return result;
    });

    nb::module_ linalg = m.def_submodule("linalg");
    litepgo::bindings::bind_linalg(linalg);
}
