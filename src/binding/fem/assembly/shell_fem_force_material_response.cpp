#include "litepgo/fem/assembly/shell_fem_force_material_response.hpp"
#include "bindings.hpp"
#include "detail/templates.hpp"
#include "litepgo/fem/material/shell/koiter_stvk.hpp"
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace nb::literals;
namespace litepgo::fem::binding {
void bind_fem_shell_fem_force_material_response(nb::module_& m) {
    using Force = ShellFemForceMaterialResponse<KoiterStVK>;
    auto cls = nb::class_<Force, energy::MeshResponse>(m, "ShellFemForceMaterialResponse[KoiterStVK]");
    auto bind_map = [&]<class Map>() {
        cls.def(nb::init<const Map&, const dof::StencilDofMap&, const std::vector<TriangleMidEdgeRestData>&, const Vec&,
                         const std::vector<double>&, int>(),
                "parameter_map"_a, "displacement_map"_a, "rest_data"_a, "displacement"_a, "thickness"_a,
                "batch_size"_a = 256, "Snapshots fixed inputs and caches per-face kinematics.");
    };
    bind_map.template operator()<dof::ConstantDofMap>();
    bind_map.template operator()<dof::ElementDofMap>();
    bindings::register_instance(m, "ShellFemForceMaterialResponse", nb::make_tuple(nb::type<KoiterStVK>()),
                                nb::type<Force>());
}
} // namespace litepgo::fem::binding
