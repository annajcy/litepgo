#include "litepgo/energy/response/mesh_response.hpp"
#include "bindings.hpp"
#include "detail/mesh_trampolines.hpp"
#include "litepgo/dof/constant_dof_map.hpp"
#include "litepgo/dof/element_dof_map.hpp"
#include "litepgo/dof/stencil_dof_map.hpp"
#include "litepgo/dof/vertex_dof_map.hpp"
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/vector.h>
#include <vector>
namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::energy::binding {
void bind_energy_response_mesh_response(nb::module_& m) {
    auto base = nb::class_<MeshResponse, Response, PyMeshResponse>(m, "MeshResponse");
    base.def(
        "eval_jvp_batch",
        [](const MeshResponse& r, const std::vector<int>& ids, const BatchMatrix& x, const BatchMatrix& d) {
            return dynamic_cast<const PyMeshResponse&>(r).default_jvp(ids, x, d);
        },
        "ids"_a, "x"_a, "dx"_a);
    base.def(
        "eval_vjp_batch",
        [](const MeshResponse& r, const std::vector<int>& ids, const BatchMatrix& x, const BatchMatrix& d) {
            return dynamic_cast<const PyMeshResponse&>(r).default_vjp(ids, x, d);
        },
        "ids"_a, "x"_a, "gy"_a);

    auto bind_input = [&]<class Input>() {
        auto bind_output = [&]<class Output>() {
            base.def(nb::init<const Input&, const Output&, int>(), "input_map"_a, "output_map"_a, "batch_size"_a = 256);
        };
        bind_output.template operator()<dof::ConstantDofMap>();
        bind_output.template operator()<dof::ElementDofMap>();
        bind_output.template operator()<dof::VertexDofMap>();
        bind_output.template operator()<dof::StencilDofMap>();
    };
    bind_input.template operator()<dof::ConstantDofMap>();
    bind_input.template operator()<dof::ElementDofMap>();
    bind_input.template operator()<dof::VertexDofMap>();
    bind_input.template operator()<dof::StencilDofMap>();
}
} // namespace litepgo::energy::binding
