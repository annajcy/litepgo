#include "litepgo/energy/scalar/vertex_pulling_energy.hpp"
#include "litepgo/bindings/bindings.hpp"
#include "litepgo/energy/scalar/energy.hpp"

#include <nanobind/eigen/dense.h>
#include <nanobind/stl/vector.h>

#include <array>

#include <stdexcept>
#include <utility>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::energy::binding {
void bind_vertex_pulling_energy(nb::module_& m) {
    nb::class_<VertexPullingEnergy, Energy>(m, "VertexPullingEnergy")
        .def(
            "__init__",
            [](VertexPullingEnergy* self, int num_dofs,
               const Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor>& position_dofs,
               const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& targets, double stiffness) {
                std::vector<std::array<int, 3>> dofs(static_cast<std::size_t>(position_dofs.rows()));
                for (Eigen::Index i = 0; i < position_dofs.rows(); ++i)
                    for (int c = 0; c < 3; ++c)
                        dofs[static_cast<std::size_t>(i)][static_cast<std::size_t>(c)] = position_dofs(i, c);
                if (targets.rows() != position_dofs.rows())
                    throw std::invalid_argument("VertexPullingEnergy requires one target row per pulled vertex");
                new (self)
                    VertexPullingEnergy(num_dofs, std::move(dofs), targets.reshaped<Eigen::RowMajor>(), stiffness);
            },
            "num_dofs"_a, "position_dofs"_a, "targets"_a, "stiffness"_a,
            "E(x) = k/2 sum_i |x[position_dofs[i]] - targets[i]|^2 from (n, 3) DOF and target arrays.");
}
} // namespace litepgo::energy::binding
