#include "litepgo/fem/material/shell/shell_material_local_energy.hpp"
#include "litepgo/bindings/bindings.hpp"
#include "litepgo/bindings/detail/fem_helpers.hpp"
#include "litepgo/bindings/detail/templates.hpp"
#include "litepgo/fem/local/local_energy.hpp"
#include "litepgo/fem/material/shell/koiter_stvk.hpp"

#include <nanobind/eigen/dense.h>
#include <nanobind/stl/string.h>

#include <string>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::fem::binding {
namespace {

template <class M> void bind_shell_material_local_energy(nb::module_& m) {
    using Energy = ShellMaterialLocalEnergy2D<M>;
    const nb::tuple key = nb::make_tuple(nb::type<M>());
    const std::string name = "ShellMaterialLocalEnergy2D[" + class_name(key[0]) + "]";
    nb::class_<Energy, LocalEnergy2D>(m, name.c_str())
        .def(nb::init<const TriangleMidEdgeRestData&, Eigen::MatrixXd, double, Eigen::VectorXd>(), "rest"_a,
             "u_local"_a, "thickness"_a, "theta"_a,
             "Density per reference area for stencil displacements u_local (one row per stencil vertex).");
    register_instance(m, "ShellMaterialLocalEnergy2D", key, nb::type<Energy>());
}
} // namespace

void bind_fem_material_shell_shell_material_local_energy(nb::module_& m) {
    bind_shell_material_local_energy<KoiterStVK>(m);
}
} // namespace litepgo::fem::binding
