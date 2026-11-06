#include "litepgo/solver/static_equilibrium_response.hpp"
#include "bindings.hpp"
#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/shared_ptr.h>
namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::solver::binding {
void bind_static_equilibrium_response(nb::module_& m) {
    nb::class_<StaticEquilibriumResponse, energy::Response>(m, "StaticEquilibriumResponse")
        .def(nb::init<int, dof::DofReduction, Vec, NewtonOptions, StaticEquilibriumResponse::MakeEnergy,
                      StaticEquilibriumResponse::MakeResponse>(),
             "num_parameters"_a, "fixed"_a, "initial"_a, "options"_a, "make_energy"_a, "make_response"_a);
}
} // namespace litepgo::solver::binding
