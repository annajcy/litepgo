#include "litepgo/solver/static_equilibrium_response.hpp"
#include "bindings.hpp"
#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/shared_ptr.h>
#include <utility>
namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::solver::binding {
namespace {
class PythonStaticEquilibriumResponse final : public StaticEquilibriumResponse {
  public:
    PythonStaticEquilibriumResponse(int num_parameters, dof::DofReduction fixed, Vec initial, NewtonOptions options,
                                    MakeEnergy make_energy, MakeResponse make_response)
        : StaticEquilibriumResponse(
              num_parameters, std::move(fixed), std::move(initial), options,
              make_energy ? MakeEnergy([this](const Vec& p) { return energy_factory_(p); }) : MakeEnergy{},
              make_response ? MakeResponse([this](const Vec& u) { return response_factory_(u); }) : MakeResponse{}),
          energy_factory_(std::move(make_energy)), response_factory_(std::move(make_response)) {}

    static int traverse(PyObject* self, visitproc visit, void* arg) noexcept {
        Py_VISIT(Py_TYPE(self));
        if (!nb::inst_ready(self))
            return 0;
        const auto* response = nb::inst_ptr<PythonStaticEquilibriumResponse>(self);
        const auto make_energy = nb::find(response->energy_factory_);
        Py_VISIT(make_energy.ptr());
        const auto make_response = nb::find(response->response_factory_);
        Py_VISIT(make_response.ptr());
        return 0;
    }

    static int clear(PyObject* self) noexcept {
        if (!nb::inst_ready(self))
            return 0;
        auto* response = nb::inst_ptr<PythonStaticEquilibriumResponse>(self);
        MakeEnergy make_energy;
        MakeResponse make_response;
        // Empty both fields before releasing Python references that may run finalizers.
        response->energy_factory_.swap(make_energy);
        response->response_factory_.swap(make_response);
        return 0;
    }

  private:
    // Only the Python wrapper owns the callables so its GC slots can release them.
    MakeEnergy energy_factory_;
    MakeResponse response_factory_;
};
} // namespace

void bind_static_equilibrium_response(nb::module_& m) {
    static const PyType_Slot slots[] = {
        {Py_tp_traverse, reinterpret_cast<void*>(PythonStaticEquilibriumResponse::traverse)},
        {Py_tp_clear, reinterpret_cast<void*>(PythonStaticEquilibriumResponse::clear)},
        {0, nullptr},
    };
    nb::class_<PythonStaticEquilibriumResponse, energy::Response>(m, "StaticEquilibriumResponse", nb::type_slots(slots))
        .def(nb::init<int, dof::DofReduction, Vec, NewtonOptions, StaticEquilibriumResponse::MakeEnergy,
                      StaticEquilibriumResponse::MakeResponse>(),
             "num_parameters"_a, "fixed"_a, "initial"_a, "options"_a, "make_energy"_a, "make_response"_a);
}
} // namespace litepgo::solver::binding
