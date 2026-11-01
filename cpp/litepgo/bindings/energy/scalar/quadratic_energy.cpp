#include "litepgo/energy/scalar/quadratic_energy.hpp"
#include "litepgo/bindings/bindings.hpp"
#include "litepgo/bindings/detail/sparse_matrix.hpp"
#include "litepgo/energy/scalar/energy.hpp"

#include <nanobind/eigen/dense.h>
#include <nanobind/stl/optional.h>

#include <optional>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::energy::binding {
void bind_quadratic_energy(nb::module_& m) {
    nb::class_<QuadraticEnergy, Energy>(m, "QuadraticEnergy")
        .def(
            "__init__",
            [](QuadraticEnergy* self, const linalg::binding::SparseArgument& A, std::optional<Vec> b) {
                new (self) QuadraticEnergy(A.matrix, b.value_or(Vec()));
            },
            "A"_a, "b"_a = nb::none(), "E(x) = 1/2 x^T A x + b^T x with a symmetric A.");
}
} // namespace litepgo::energy::binding
