#include "litepgo/fem/shell_fem_energy.hpp"
#include "litepgo/bindings/bindings.hpp"
#include "litepgo/bindings/detail/fem_helpers.hpp"
#include "litepgo/bindings/detail/templates.hpp"
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
    const nb::tuple key = nb::make_tuple(nb::type<M>());
    const std::string name = "ShellFemEnergy[" + class_name(key[0]) + "]";
    nb::class_<Energy, energy::Energy>(m, name.c_str())
        .def(nb::init<const dof::StencilDofMap&, std::vector<TriangleMidEdgeRestData>, std::vector<double>,
                      std::vector<Eigen::VectorXd>, const std::vector<Eigen::Matrix3d>&>(),
             "displacement_map"_a, "rest_data"_a, "thickness"_a, "theta"_a, "Fp"_a = std::vector<Eigen::Matrix3d>{},
             "Fixed per-face inputs; optional Fp changes natural forms, preserving original areas and thickness.");
    register_instance(m, "ShellFemEnergy", key, nb::type<Energy>());
}
} // namespace

void bind_fem_shell_fem_energy(nb::module_& m) {
    bind_shell_fem_energy<KoiterStVK>(m);
}
} // namespace litepgo::fem::binding
