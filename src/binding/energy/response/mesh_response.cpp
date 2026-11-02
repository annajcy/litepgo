#include "litepgo/energy/response/mesh_response.hpp"
#include "bindings.hpp"
#include "detail/sparse_matrix.hpp"
#include "litepgo/dof/constant_dof_map.hpp"
#include "litepgo/dof/element_dof_map.hpp"
#include "litepgo/dof/stencil_dof_map.hpp"
#include "litepgo/dof/vertex_dof_map.hpp"
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/vector.h>
#include <stdexcept>
#include <vector>
namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::energy::binding {
namespace {
template <class Input, class Output, class Cls> void bind_mesh_response(Cls& cls) {
    cls.def(
        "__init__",
        [](MeshResponse* self, const Input& input, const Output& output, nb::callable evaluator, int batch_size) {
            new (self) MeshResponse(
                input, output,
                [evaluator](std::span<const int> ids, const BatchMatrix& x, BatchMatrix* J) -> BatchMatrix {
                    nb::gil_scoped_acquire acquire;
                    nb::object value = evaluator(std::vector<int>(ids.begin(), ids.end()),
                                                 nb::cast(x, nb::rv_policy::copy), J != nullptr);
                    if (value.is_none())
                        throw std::domain_error("MeshResponse: infeasible batch");
                    if (!J)
                        return nb::cast<BatchMatrix>(value);
                    if (!nb::isinstance<nb::tuple>(value) || nb::len(value) != 2)
                        throw nb::type_error("Expected (batch_values, batch_Jacobians)");
                    nb::tuple pair = nb::cast<nb::tuple>(value);
                    *J = nb::cast<BatchMatrix>(pair[1]);
                    return nb::cast<BatchMatrix>(pair[0]);
                },
                batch_size);
        },
        "input_map"_a, "output_map"_a, "eval_batch"_a, "batch_size"_a = 256);
}

template <class Input, class Cls> void bind_batch_outputs(Cls& cls) {
    bind_mesh_response<Input, dof::ConstantDofMap>(cls);
    bind_mesh_response<Input, dof::ElementDofMap>(cls);
    bind_mesh_response<Input, dof::VertexDofMap>(cls);
    bind_mesh_response<Input, dof::StencilDofMap>(cls);
}
} // namespace

void bind_energy_response_mesh_response(nb::module_& m) {
    using linalg::binding::SparseArgument;
    auto batch = nb::class_<MeshResponse, Response>(m, "MeshResponse");
    bind_batch_outputs<dof::ConstantDofMap>(batch);
    bind_batch_outputs<dof::ElementDofMap>(batch);
    bind_batch_outputs<dof::VertexDofMap>(batch);
    bind_batch_outputs<dof::StencilDofMap>(batch);
}
} // namespace litepgo::energy::binding
