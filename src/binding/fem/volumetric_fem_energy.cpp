#include "litepgo/fem/volumetric_fem_energy.hpp"
#include "bindings.hpp"
#include "detail/fem_helpers.hpp"
#include "detail/templates.hpp"
#include "litepgo/fem/field/shared_field.hpp"
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

template <class M, class B, class Q> void bind_volumetric_fem_energy(nb::module_& m) {
    using Energy = VolumetricFemEnergy<M, B, Q, SharedField3D, SharedField3D, SharedField3D>;
    const nb::tuple key = nb::make_tuple(nb::type<M>(), nb::type<B>(), nb::type<Q>());
    const std::string name =
        "VolumetricFemEnergy[" + class_name(key[0]) + ", " + class_name(key[1]) + ", " + class_name(key[2]) + "]";
    nb::class_<Energy, energy::MeshEnergy>(m, name.c_str())
        .def(
            "__init__",
            [](Energy* self, const dof::VertexDofMap& displacement_map, const Fields& rest_position_fields,
               const Fields& Fp, const Fields& theta) {
                new (self) Energy(displacement_map, shared(rest_position_fields), shared(Fp), shared(theta));
            },
            "displacement_map"_a, "rest_position_fields"_a, "Fp"_a, "theta"_a,
            "One Field3D per element in map order for each of rest positions, Fp (9 components) and theta; the "
            "energy shares ownership of every field.");
    register_instance(m, "VolumetricFemEnergy", key, nb::type<Energy>());
}
} // namespace

void bind_fem_volumetric_fem_energy(nb::module_& m) {
    bind_volumetric_fem_energy<NeoHookean, TetP1Basis, TetCentroidQuadrature>(m);
    bind_volumetric_fem_energy<NeoHookean, HexQ1Basis, GaussLegendre3D<2>>(m);
    bind_volumetric_fem_energy<NeoHookean, HexQ1Basis, GaussLegendre3D<4>>(m);
    bind_volumetric_fem_energy<NeoHookean, HexHermiteBasis, GaussLegendre3D<2>>(m);
    bind_volumetric_fem_energy<NeoHookean, HexHermiteBasis, GaussLegendre3D<4>>(m);
}
} // namespace litepgo::fem::binding
