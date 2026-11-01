#include "litepgo/common/build_info.hpp"
#include "litepgo/bindings/bindings.hpp"
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
namespace nb = nanobind;
namespace litepgo::common::binding {
void bind_build_info(nb::module_& m) {
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
}
} // namespace litepgo::common::binding
