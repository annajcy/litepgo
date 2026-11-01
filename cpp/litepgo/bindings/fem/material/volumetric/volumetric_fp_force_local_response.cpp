#include "litepgo/fem/material/volumetric/volumetric_fp_force_local_response.hpp"
#include "litepgo/bindings/bindings.hpp"
#include "litepgo/bindings/detail/fem_helpers.hpp"
#include "litepgo/fem/local/local_response.hpp"
#include "litepgo/fem/material/volumetric/neo_hookean.hpp"

#include <nanobind/eigen/dense.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::fem::binding {
namespace {} // namespace

void bind_volumetric_fp_force_local_response(nb::module_& m) {
    bind_force_local_response<NeoHookean, VolumetricFpForceLocalResponse3D<NeoHookean>>(
        m, "VolumetricFpForceLocalResponse3D", "Force density and Fp-coefficient Jacobian at fixed F and theta.");
}
} // namespace litepgo::fem::binding
