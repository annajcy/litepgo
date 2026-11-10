#include "litepgo/fem/assembly/shell_fem_energy.hpp"
#include "bindings.hpp"
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>
namespace nb = nanobind;
using namespace nb::literals;
namespace litepgo::fem::binding {
void bind_fem_shell_fem_energy(nb::module_& m) {
    nb::class_<ShellFemEnergy, energy::MeshEnergy>(m, "ShellFemEnergy")
        .def(nb::init<std::shared_ptr<const ShellBatchMaterial>, const dof::StencilDofMap&,
                      std::vector<TriangleMidEdgeRestData>, std::vector<double>, std::vector<Vec>,
                      const std::vector<Eigen::Matrix3d>&, int>(),
             "material"_a, "displacement_map"_a, "rest_data"_a, "thickness"_a, "theta"_a, "Fp"_a, "batch_size"_a = 256);
}
} // namespace litepgo::fem::binding
