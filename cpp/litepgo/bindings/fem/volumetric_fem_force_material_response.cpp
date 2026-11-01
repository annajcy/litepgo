#include "litepgo/fem/volumetric_fem_force_material_response.hpp"
#include "litepgo/bindings/bindings.hpp"
#include "litepgo/bindings/detail/fem_helpers.hpp"
#include "litepgo/fem/field/shared_field.hpp"
#include "litepgo/fem/material/volumetric/neo_hookean.hpp"
#include "litepgo/fem/quadrature/3d/gauss_legendre_3d.hpp"
#include "litepgo/fem/quadrature/3d/tet_centroid_quadrature.hpp"
#include <nanobind/eigen/sparse.h>

#include <nanobind/eigen/dense.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::fem::binding {
namespace {

template <class M, class B, class Q> void bind_volumetric_fem_force_material_response(nb::module_& m) {
    using Force = VolumetricFemForceMaterialResponse<M, B, Q, SharedField3D, SharedField3D>;
    bind_force<Force, M, B, Q>(m, "VolumetricFemForceMaterialResponse", "Fp",
                               "Fixed displacement/geometry/Fp; force and material-parameter Jacobian.");
}
} // namespace

void bind_fem_volumetric_fem_force_material_response(nb::module_& m) {
    bind_volumetric_fem_force_material_response<NeoHookean, TetP1Basis, TetCentroidQuadrature>(m);
    bind_volumetric_fem_force_material_response<NeoHookean, HexQ1Basis, GaussLegendre3D<2>>(m);
    bind_volumetric_fem_force_material_response<NeoHookean, HexQ1Basis, GaussLegendre3D<4>>(m);
    bind_volumetric_fem_force_material_response<NeoHookean, HexHermiteBasis, GaussLegendre3D<2>>(m);
    bind_volumetric_fem_force_material_response<NeoHookean, HexHermiteBasis, GaussLegendre3D<4>>(m);
}
} // namespace litepgo::fem::binding
