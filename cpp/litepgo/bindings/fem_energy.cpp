#include "litepgo/bindings/bindings.hpp"
#include "litepgo/bindings/templates.hpp"
#include "litepgo/fem/field/shared_field.hpp"
#include "litepgo/fem/material/shell/koiter_stvk.hpp"
#include "litepgo/fem/material/volumetric/neo_hookean.hpp"
#include "litepgo/fem/quadrature/3d/gauss_legendre_3d.hpp"
#include "litepgo/fem/quadrature/3d/tet_centroid_quadrature.hpp"
#include "litepgo/fem/shell_fem_energy.hpp"
#include "litepgo/fem/volumetric_fem_energy.hpp"

#include <nanobind/eigen/dense.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <memory>
#include <string>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::fem::binding {
namespace {
using bindings::register_instance;

// Field types are erased: Python passes any Field3D, held through SharedField3D.
using Fields = std::vector<std::shared_ptr<Field3D>>;

std::vector<SharedField3D> shared(const Fields& fields) {
    std::vector<SharedField3D> result;
    result.reserve(fields.size());
    for (const auto& field : fields)
        result.emplace_back(field);
    return result;
}

std::string class_name(nb::handle cls) {
    return nb::cast<std::string>(cls.attr("__name__"));
}

template <class M, class B, class Q> void bind_volumetric(nb::module_& m) {
    using Energy = VolumetricFemEnergy<M, B, Q, SharedField3D, SharedField3D, SharedField3D>;
    const nb::tuple key = nb::make_tuple(nb::type<M>(), nb::type<B>(), nb::type<Q>());
    const std::string name =
        "VolumetricFemEnergy[" + class_name(key[0]) + ", " + class_name(key[1]) + ", " + class_name(key[2]) + "]";
    nb::class_<Energy, energy::Energy>(m, name.c_str())
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

template <class M> void bind_shell(nb::module_& m) {
    using Energy = ShellFemEnergy<M>;
    const nb::tuple key = nb::make_tuple(nb::type<M>());
    const std::string name = "ShellFemEnergy[" + class_name(key[0]) + "]";
    nb::class_<Energy, energy::Energy>(m, name.c_str())
        .def(nb::init<const dof::StencilDofMap&, std::vector<TriangleMidEdgeRestData>, std::vector<double>,
                      std::vector<Eigen::VectorXd>>(),
             "displacement_map"_a, "rest_data"_a, "thickness"_a, "theta"_a,
             "One rest data, positive thickness and parameter vector per face, in map order.");
    register_instance(m, "ShellFemEnergy", key, nb::type<Energy>());
}
} // namespace

void bind_energies(nb::module_& m) {
    // Add an instantiation here to make it available as Template[args] in Python.
    bind_volumetric<NeoHookean, TetP1Basis, TetCentroidQuadrature>(m);
    bind_volumetric<NeoHookean, HexQ1Basis, GaussLegendre3D<2>>(m);
    bind_volumetric<NeoHookean, HexQ1Basis, GaussLegendre3D<4>>(m);
    bind_volumetric<NeoHookean, HexHermiteBasis, GaussLegendre3D<2>>(m);
    bind_volumetric<NeoHookean, HexHermiteBasis, GaussLegendre3D<4>>(m);
    bind_shell<KoiterStVK>(m);
}

} // namespace litepgo::fem::binding
