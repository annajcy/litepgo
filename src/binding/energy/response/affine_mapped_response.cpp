#include "litepgo/energy/response/affine_mapped_response.hpp"
#include "bindings.hpp"
#include "detail/sparse_matrix.hpp"

#include <nanobind/eigen/dense.h>
#include <nanobind/stl/optional.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::energy::binding {
void bind_affine_mapped_response(nb::module_& m) {
    nb::class_<AffineMappedResponse, Response>(m, "AffineMappedResponse")
        .def(
            "__init__",
            [](AffineMappedResponse* self, const linalg::binding::SparseArgument& A, std::optional<Vec> offset) {
                new (self) AffineMappedResponse(A.matrix, offset.value_or(Vec()));
            },
            "A"_a, "offset"_a = nb::none(),
            "R(x) = A @ x + offset. Owns snapshots; omitted or empty offset means zero. Exact Hessians are zero.");
}
} // namespace litepgo::energy::binding
