#include "litepgo/energy/scalar/mesh_energy.hpp"
#include "litepgo/bindings/bindings.hpp"
#include "litepgo/dof/constant_dof_map.hpp"
#include "litepgo/dof/element_dof_map.hpp"
#include "litepgo/dof/stencil_dof_map.hpp"
#include "litepgo/dof/vertex_dof_map.hpp"
#include "litepgo/energy/scalar/energy.hpp"
#include <functional>
#include <nanobind/stl/function.h>
#include <nanobind/stl/tuple.h>
#include <tuple>

#include <nanobind/eigen/dense.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::energy::binding {
void bind_mesh_energy(nb::module_& m) {
    using BatchResult = std::tuple<Vec, std::optional<BatchMatrix>, std::optional<BatchMatrix>>;
    using BatchCallback = std::function<std::optional<BatchResult>(std::vector<int>, BatchMatrix, bool, bool)>;
    using BatchStep = std::function<Vec(std::vector<int>, BatchMatrix, BatchMatrix)>;
    auto mesh = nb::class_<MeshEnergy, Energy>(m, "MeshEnergy");
    auto bind_map = [&]<class Map>() {
        mesh.def(
            "__init__",
            [](MeshEnergy* self, const Map& map, BatchCallback callback, int batch_size,
               std::optional<BatchStep> step) {
                if (!callback || (step && !*step))
                    throw std::invalid_argument("Expected batch callbacks");
                auto evaluate = [callback](std::span<const int> ids, const BatchMatrix& x, BatchMatrix* g,
                                           BatchMatrix* h) {
                    auto result = callback(std::vector<int>(ids.begin(), ids.end()), x, g != nullptr, h != nullptr);
                    if (!result)
                        throw std::domain_error("MeshEnergy: infeasible batch");
                    auto [values, gradients, hessians] = std::move(*result);
                    if (g) {
                        if (!gradients)
                            throw std::invalid_argument("Missing batch gradients");
                        *g = std::move(*gradients);
                    }
                    if (h) {
                        if (!hessians)
                            throw std::invalid_argument("Missing batch Hessians");
                        *h = std::move(*hessians);
                    }
                    return values;
                };
                MeshEnergy::Step bound;
                if (step)
                    bound = [step = *step](std::span<const int> ids, const BatchMatrix& x, const BatchMatrix& dx) {
                        return step(std::vector<int>(ids.begin(), ids.end()), x, dx);
                    };
                new (self) MeshEnergy(map, std::move(evaluate), batch_size, std::move(bound));
            },
            "map"_a, "eval_batch"_a, "batch_size"_a = 256, "step"_a = nb::none());
    };
    bind_map.template operator()<dof::StencilDofMap>();
    bind_map.template operator()<dof::ConstantDofMap>();
    bind_map.template operator()<dof::ElementDofMap>();
    bind_map.template operator()<dof::VertexDofMap>();
}
} // namespace litepgo::energy::binding
