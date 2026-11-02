#include "litepgo/fem/material/volumetric/volumetric_material_local_energy.hpp"
#include "bindings.hpp"
#include "detail/fem_helpers.hpp"
#include "detail/templates.hpp"
#include "litepgo/fem/local/local_energy.hpp"
#include "litepgo/fem/material/volumetric/neo_hookean.hpp"

#include <nanobind/eigen/dense.h>
#include <nanobind/stl/string.h>

#include <string>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::fem::binding {
namespace {

template <class M> void bind_volumetric_locals(nb::module_& m) {
    using Energy = VolumetricMaterialLocalEnergyF3D<M>;
    const nb::tuple key = nb::make_tuple(nb::type<M>());
    const std::string name = "VolumetricMaterialLocalEnergyF3D[" + class_name(key[0]) + "]";
    nb::class_<Energy, LocalEnergy3D>(m, name.c_str())
        .def(nb::init<const Field3D&, const Field3D&, const Field3D&>(), "F"_a, "Fp"_a, "theta"_a,
             nb::keep_alive<1, 2>(), nb::keep_alive<1, 3>(), nb::keep_alive<1, 4>(),
             "Borrows the three fields and keeps them alive.")
        .def("eval_feasible_step", &Energy::eval_feasible_step, "xi"_a, "delta_coeffs"_a);
    register_instance(m, "VolumetricMaterialLocalEnergyF3D", key, nb::type<Energy>());
}
} // namespace

void bind_fem_material_volumetric_volumetric_material_local_energy(nb::module_& m) {
    bind_volumetric_locals<NeoHookean>(m);
}
} // namespace litepgo::fem::binding
