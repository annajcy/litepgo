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

template <class B, class Q> struct BindEnergy {
    static void apply(nb::module_& m) {
        using Energy = VolumetricFemEnergy<B, Q>;
        const nb::tuple key = nb::make_tuple(nb::type<B>(), nb::type<Q>());
        const std::string name = "VolumetricFemEnergy[" + class_name(key[0]) + ", " + class_name(key[1]) + "]";
        nb::class_<Energy, energy::MeshEnergy>(m, name.c_str())
            .def(nb::init<std::shared_ptr<const VolumetricBatchMaterial>, const dof::VertexDofMap&, const Responses&,
                          const Responses&, const Responses&, int>(),
                 "material"_a, "displacement_map"_a, "rest_position_fields"_a, "Fp"_a, "theta"_a, "batch_size"_a = 256);
        register_instance(m, "VolumetricFemEnergy", key, nb::type<Energy>());
    }
};
} // namespace

void bind_fem_volumetric_fem_energy(nb::module_& m) {
    for_each_volumetric_instance<BindEnergy>(m);
}
} // namespace litepgo::fem::binding
