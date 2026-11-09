#include "litepgo/fem/assembly/shell_fem_force_fp_response.hpp"
#include "bindings.hpp"
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>
namespace nb = nanobind;
using namespace nb::literals;
namespace litepgo::fem::binding {
void bind_fem_shell_fem_force_fp_response(nb::module_& m) {
    auto cls = nb::class_<ShellFemForceFpResponse, energy::MeshResponse>(m, "ShellFemForceFpResponse");
    auto bind_map = [&]<class Map>() {
        cls.def(nb::init<std::shared_ptr<const ShellBatchMaterial>, const Map&, const dof::StencilDofMap&,
                         const std::vector<TriangleMidEdgeRestData>&, const Vec&, const std::vector<double>&,
                         const std::vector<Vec>&, int>(),
                "material"_a, "parameter_map"_a, "displacement_map"_a, "rest_data"_a, "displacement"_a, "thickness"_a,
                "theta"_a, "batch_size"_a = 256);
    };
    bind_map.template operator()<dof::ConstantDofMap>();
    bind_map.template operator()<dof::ElementDofMap>();
}
} // namespace litepgo::fem::binding
