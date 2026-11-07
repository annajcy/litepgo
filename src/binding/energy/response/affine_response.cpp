#include "litepgo/energy/response/affine_response.hpp"
#include "bindings.hpp"
#include "detail/sparse_matrix.hpp"
#include "litepgo/energy/response/constant_response.hpp"
#include "litepgo/energy/response/identity_response.hpp"

#include <nanobind/eigen/dense.h>
#include <nanobind/stl/optional.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::energy::binding {
void bind_affine_response(nb::module_& m) {
    nb::class_<AffineResponse, Response>(m, "AffineResponse")
        .def(
            "__init__",
            [](AffineResponse* self, const linalg::binding::SparseArgument& A, std::optional<Vec> offset) {
                new (self) AffineResponse(A.matrix, offset.value_or(Vec()));
            },
            "A"_a, "offset"_a = nb::none(),
            "R(x) = A @ x + offset. Owns snapshots; omitted or empty offset means zero. Exact Hessians are zero.");
    nb::class_<IdentityResponse, Response>(m, "IdentityResponse")
        .def(nb::init<int>(), "size"_a, "R(x) = x. Direct products return the supplied direction or weights.");
    nb::class_<ConstantResponse, Response>(m, "ConstantResponse")
        .def(nb::init<int, Vec>(), "num_inputs"_a, "value"_a,
             "R(x) = value. Owns a snapshot; all derivatives are zero.");
}
} // namespace litepgo::energy::binding
