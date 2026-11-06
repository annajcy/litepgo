#include "litepgo/fem/assembly/shell_fem_force_fp_response.hpp"
#include "bindings.hpp"
#include "detail/fem_helpers.hpp"
#include "detail/templates.hpp"
#include "litepgo/fem/material/shell/koiter_stvk.hpp"
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace nb::literals;
namespace litepgo::fem::binding {
namespace {
template <class M> void bind_shell_fp_force(nb::module_& m) {
    using Force = ShellFemForceFpResponse<M>;
    const nb::tuple key = nb::make_tuple(material_key<M>());
    const std::string name = "ShellFemForceFpResponse[" + class_name(key[0]) + "]";
    auto cls = nb::class_<Force, energy::MeshResponse>(m, name.c_str());
    const char* doc = "Snapshots fixed inputs and caches per-face kinematics.";
    auto bind_map = [&]<class Map>() {
        if constexpr (is_py_material<M>)
            cls.def(
                "__init__",
                [](nb::pointer_and_handle<Force> self, nb::handle material, const Map& parameter_map,
                   const dof::StencilDofMap& map, const std::vector<TriangleMidEdgeRestData>& rest, const Vec& u,
                   const std::vector<double>& thickness, const std::vector<Vec>& theta, int batch_size) {
                    new (self.p) Force(parameter_map, map, rest, u, thickness, theta, batch_size,
                                       owned_material<M>(self, material));
                },
                "material"_a, "parameter_map"_a, "displacement_map"_a, "rest_data"_a, "displacement"_a, "thickness"_a,
                "theta"_a, "batch_size"_a = 256, doc);
        else
            cls.def(nb::init<const Map&, const dof::StencilDofMap&, const std::vector<TriangleMidEdgeRestData>&,
                             const Vec&, const std::vector<double>&, const std::vector<Vec>&, int>(),
                    "parameter_map"_a, "displacement_map"_a, "rest_data"_a, "displacement"_a, "thickness"_a, "theta"_a,
                    "batch_size"_a = 256, doc);
    };
    bind_map.template operator()<dof::ConstantDofMap>();
    bind_map.template operator()<dof::ElementDofMap>();
    bindings::register_instance(m, "ShellFemForceFpResponse", key, nb::type<Force>());
}
} // namespace

void bind_fem_shell_fem_force_fp_response(nb::module_& m) {
    bind_shell_fp_force<KoiterStVK>(m);
    bind_shell_fp_force<PyShellMaterial>(m);
}
} // namespace litepgo::fem::binding
