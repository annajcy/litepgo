#include "litepgo/fem/assembly/volumetric_fem_energy.hpp"
#include "bindings.hpp"
#include "detail/fem_helpers.hpp"
#include "detail/templates.hpp"
#include "litepgo/fem/material/volumetric/neo_hookean.hpp"
#include "litepgo/fem/quadrature/3d/gauss_legendre_3d.hpp"
#include "litepgo/fem/quadrature/3d/tet_centroid_quadrature.hpp"
#include <nanobind/eigen/sparse.h>

#include <nanobind/eigen/dense.h>
#include <nanobind/stl/string.h>

#include <string>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::fem::binding {
namespace {

template <class M, class B, class Q> struct BindEnergy {
    static void apply(nb::module_& m) {
        using Energy = VolumetricFemEnergy<M, B, Q>;
        const nb::tuple key = nb::make_tuple(material_key<M>(), nb::type<B>(), nb::type<Q>());
        const std::string name =
            "VolumetricFemEnergy[" + class_name(key[0]) + ", " + class_name(key[1]) + ", " + class_name(key[2]) + "]";
        nb::class_<Energy, energy::MeshEnergy> cls(m, name.c_str());
        const char* doc = "One spatial Response per element in map order for each of rest positions, Fp (9 components) "
                          "and theta; the energy snapshots their values at construction.";
        if constexpr (is_py_material<M>)
            cls.def(
                "__init__",
                [](nb::pointer_and_handle<Energy> self, nb::handle material, const dof::VertexDofMap& displacement_map,
                   const Responses& rest_position_fields, const Responses& Fp, const Responses& theta, int batch_size) {
                    new (self.p) Energy(displacement_map, rest_position_fields, Fp, theta, batch_size,
                                        owned_material<M>(self, material));
                },
                "material"_a, "displacement_map"_a, "rest_position_fields"_a, "Fp"_a, "theta"_a, "batch_size"_a = 256,
                doc);
        else
            cls.def(
                "__init__",
                [](Energy* self, const dof::VertexDofMap& displacement_map, const Responses& rest_position_fields,
                   const Responses& Fp, const Responses& theta, int batch_size) {
                    new (self) Energy(displacement_map, rest_position_fields, Fp, theta, batch_size);
                },
                "displacement_map"_a, "rest_position_fields"_a, "Fp"_a, "theta"_a, "batch_size"_a = 256, doc);
        register_instance(m, "VolumetricFemEnergy", key, nb::type<Energy>());
    }
};
} // namespace

void bind_fem_volumetric_fem_energy(nb::module_& m) {
    for_each_volumetric_instance<BindEnergy>(m);
}
} // namespace litepgo::fem::binding
