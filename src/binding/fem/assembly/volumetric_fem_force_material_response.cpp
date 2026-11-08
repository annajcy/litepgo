#include "litepgo/fem/assembly/volumetric_fem_force_material_response.hpp"
#include "bindings.hpp"
#include "detail/fem_helpers.hpp"
#include "litepgo/fem/material/volumetric/neo_hookean.hpp"
#include "litepgo/fem/quadrature/3d/gauss_legendre_3d.hpp"
#include "litepgo/fem/quadrature/3d/tet_centroid_quadrature.hpp"
#include <nanobind/eigen/sparse.h>

#include <nanobind/eigen/dense.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::fem::binding {
namespace {

template <class M, class B, class Q> struct BindForce {
    static void apply(nb::module_& m) {
        bind_force<VolumetricFemForceMaterialResponse<M, B, Q>, M, B, Q>(
            m, "VolumetricFemForceMaterialResponse", "Fp",
            "Fixed displacement/geometry/Fp; force and material-parameter Jacobian.");
    }
};
} // namespace

void bind_fem_volumetric_fem_force_material_response(nb::module_& m) {
    for_each_volumetric_instance<BindForce>(m);
}
} // namespace litepgo::fem::binding
