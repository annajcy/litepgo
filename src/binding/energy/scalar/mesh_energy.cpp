#include "litepgo/energy/scalar/mesh_energy.hpp"
#include "bindings.hpp"
#include "detail/mesh_trampolines.hpp"
#include "litepgo/dof/constant_dof_map.hpp"
#include "litepgo/dof/element_dof_map.hpp"
#include "litepgo/dof/stencil_dof_map.hpp"
#include "litepgo/dof/vertex_dof_map.hpp"
#include "litepgo/energy/scalar/energy.hpp"

#include <nanobind/eigen/dense.h>
#include <nanobind/stl/vector.h>

#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::energy::binding {
void bind_mesh_energy(nb::module_& m) {
    auto base = nb::class_<MeshEnergy, Energy, PyMeshEnergy>(m, "MeshEnergy");
    base.def(
        "eval_feasible_step_batch",
        [](const MeshEnergy& e, const std::vector<int>& ids, const BatchMatrix& x, const BatchMatrix& d) {
            return dynamic_cast<const PyMeshEnergy&>(e).default_step(ids, x, d);
        },
        "ids"_a, "x"_a, "dx"_a);

    auto bind_map = [&]<class Map>() { base.def(nb::init<const Map&, int>(), "map"_a, "batch_size"_a = 256); };
    bind_map.template operator()<dof::StencilDofMap>();
    bind_map.template operator()<dof::ConstantDofMap>();
    bind_map.template operator()<dof::ElementDofMap>();
    bind_map.template operator()<dof::VertexDofMap>();
}
} // namespace litepgo::energy::binding
