#include "litepgo/fem/assembly/shell_fem_energy.hpp"
#include "bindings.hpp"
#include "detail/fem_helpers.hpp"
#include "detail/templates.hpp"
#include "litepgo/fem/material/shell/koiter_stvk.hpp"
#include <nanobind/eigen/sparse.h>

#include <nanobind/eigen/dense.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <string>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::fem::binding {
namespace {

template <class M> void bind_shell_fem_energy(nb::module_& m) {
    using Energy = ShellFemEnergy<M>;
    const nb::tuple key = nb::make_tuple(material_key<M>());
    const std::string name = "ShellFemEnergy[" + class_name(key[0]) + "]";
    const char* doc =
        "Fixed per-face inputs; optional Fp changes natural forms, preserving original areas and thickness.";
    nb::class_<Energy, energy::MeshEnergy> cls(m, name.c_str());
    if constexpr (is_py_material<M>)
        cls.def(
            "__init__",
            [](nb::pointer_and_handle<Energy> self, nb::handle material, const dof::StencilDofMap& map,
               std::vector<TriangleMidEdgeRestData> rest, std::vector<double> thickness,
               std::vector<Eigen::VectorXd> theta, const std::vector<Eigen::Matrix3d>& Fp, int batch_size) {
                new (self.p) Energy(map, std::move(rest), std::move(thickness), std::move(theta), Fp, batch_size,
                                    owned_material<M>(self, material));
            },
            "material"_a, "displacement_map"_a, "rest_data"_a, "thickness"_a, "theta"_a,
            "Fp"_a = std::vector<Eigen::Matrix3d>{}, "batch_size"_a = 256, doc);
    else
        cls.def(nb::init<const dof::StencilDofMap&, std::vector<TriangleMidEdgeRestData>, std::vector<double>,
                         std::vector<Eigen::VectorXd>, const std::vector<Eigen::Matrix3d>&, int>(),
                "displacement_map"_a, "rest_data"_a, "thickness"_a, "theta"_a, "Fp"_a = std::vector<Eigen::Matrix3d>{},
                "batch_size"_a = 256, doc);
    register_instance(m, "ShellFemEnergy", key, nb::type<Energy>());
}
} // namespace

void bind_fem_shell_fem_energy(nb::module_& m) {
    bind_shell_fem_energy<KoiterStVK>(m);
    bind_shell_fem_energy<PyShellMaterial>(m);
}
} // namespace litepgo::fem::binding
