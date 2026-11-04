#include "litepgo/energy/parametric/parametric_energy.hpp"
#include "bindings.hpp"
#include "detail/energy_helpers.hpp"
#include <nanobind/stl/shared_ptr.h>

namespace nb = nanobind;
using namespace nb::literals;
namespace litepgo::energy::binding {
class PyParametricEnergy : public ParametricEnergy {
  public:
    NB_TRAMPOLINE(ParametricEnergy);
    PyParametricEnergy(const linalg::binding::SparseArgument& K, const linalg::binding::SparseArgument& B, bool fixed)
        : ParametricEnergy(K.matrix, B.matrix, fixed) {}
    Vec default_jvp(const ConstVecRef& u, const ConstVecRef& p, const ConstVecRef& v) const {
        return ParametricEnergy::eval_parameter_jvp(u, p, v);
    }
    Vec default_vjp(const ConstVecRef& u, const ConstVecRef& p, const ConstVecRef& w) const {
        return ParametricEnergy::eval_parameter_vjp(u, p, w);
    }
    SpMat default_state_pattern(const ConstVecRef& u, const ConstVecRef& p) const {
        return ParametricEnergy::state_pattern_at(u, p);
    }
    SpMat default_parameter_pattern(const ConstVecRef& u, const ConstVecRef& p) const {
        return ParametricEnergy::parameter_pattern_at(u, p);
    }

  protected:
    double eval_state_values(const ConstVecRef& u, const ConstVecRef& p, Vec* g, SpMat* K,
                             HessianKind kind) const override {
        nb::detail::ticket ticket(nb_trampoline, "eval_state_values", nb::detail::str_hash("eval_state_values"), true);
        auto result =
            nb_trampoline.base().attr(ticket.key)(Vec(u), Vec(p), g != nullptr, K ? nb::cast(kind) : nb::none());
        if (result.is_none())
            throw std::domain_error("ParametricEnergy: infeasible");
        if (!nb::isinstance<nb::tuple>(result) || nb::len(result) != 3)
            throw nb::type_error("Expected (value, gradient, Hessian)");
        auto tuple = nb::cast<nb::tuple>(result);
        if (g)
            *g = nb::cast<Vec>(tuple[1]);
        if (K) {
            const auto source = nb::cast<linalg::binding::SparseArgument>(tuple[2]).matrix;
            copy_derivative_values(source, *K, "ParametricEnergy: invalid Hessian",
                                   "ParametricEnergy: Hessian outside pattern");
        }
        return nb::cast<double>(tuple[0]);
    }
    void eval_parameter_jacobian(const ConstVecRef& u, const ConstVecRef& p, SpMat& B) const override {
        nb::detail::ticket ticket(nb_trampoline, "eval_parameter_jacobian",
                                  nb::detail::str_hash("eval_parameter_jacobian"), true);
        const auto source =
            nb::cast<linalg::binding::SparseArgument>(nb_trampoline.base().attr(ticket.key)(Vec(u), Vec(p))).matrix;
        copy_derivative_values(source, B, "ParametricEnergy: invalid parameter Jacobian",
                               "ParametricEnergy: parameter Jacobian outside pattern");
    }
    Vec eval_parameter_jvp(const ConstVecRef& u, const ConstVecRef& p, const ConstVecRef& v) const override {
        NB_OVERRIDE(eval_parameter_jvp, Vec(u), Vec(p), Vec(v));
    }
    Vec eval_parameter_vjp(const ConstVecRef& u, const ConstVecRef& p, const ConstVecRef& w) const override {
        NB_OVERRIDE(eval_parameter_vjp, Vec(u), Vec(p), Vec(w));
    }
    SpMat state_pattern_at(const ConstVecRef& u, const ConstVecRef& p) const override {
        nb::detail::ticket ticket(nb_trampoline, "state_pattern_at", nb::detail::str_hash("state_pattern_at"), false);
        if (!ticket.key.is_valid())
            return ParametricEnergy::state_pattern_at(u, p);
        return nb::cast<linalg::binding::SparseArgument>(nb_trampoline.base().attr(ticket.key)(Vec(u), Vec(p))).matrix;
    }
    SpMat parameter_pattern_at(const ConstVecRef& u, const ConstVecRef& p) const override {
        nb::detail::ticket ticket(nb_trampoline, "parameter_pattern_at", nb::detail::str_hash("parameter_pattern_at"),
                                  false);
        if (!ticket.key.is_valid())
            return ParametricEnergy::parameter_pattern_at(u, p);
        return nb::cast<linalg::binding::SparseArgument>(nb_trampoline.base().attr(ticket.key)(Vec(u), Vec(p))).matrix;
    }
    double eval_feasible_step(const ConstVecRef& u, const ConstVecRef& p, const ConstVecRef& du) const override {
        NB_OVERRIDE(eval_feasible_step, Vec(u), Vec(p), Vec(du));
    }
};
void bind_parametric_energy(nb::module_& m) {
    using linalg::binding::SparseArgument;
    nb::class_<ParametricEnergy, PyParametricEnergy>(m, "ParametricEnergy")
        .def(nb::init<const SparseArgument&, const SparseArgument&, bool>(), "state_pattern"_a, "parameter_pattern"_a,
             "fixed_pattern"_a = true)
        .def_prop_ro("num_dofs", &ParametricEnergy::num_dofs)
        .def_prop_ro("num_parameters", &ParametricEnergy::num_parameters)
        .def(
            "energy_at",
            [](std::shared_ptr<ParametricEnergy> model, const ConstVecRef& p) { return model->energy_at(p); }, "p"_a)
        .def(
            "force_at",
            [](std::shared_ptr<ParametricEnergy> model, const ConstVecRef& u) { return model->force_at(u); }, "u"_a)
        .def(
            "eval_parameter_jvp",
            [](const ParametricEnergy& model, const ConstVecRef& u, const ConstVecRef& p, const ConstVecRef& v) {
                return dynamic_cast<const PyParametricEnergy&>(model).default_jvp(u, p, v);
            },
            "u"_a, "p"_a, "v"_a)
        .def(
            "eval_parameter_vjp",
            [](const ParametricEnergy& model, const ConstVecRef& u, const ConstVecRef& p, const ConstVecRef& w) {
                return dynamic_cast<const PyParametricEnergy&>(model).default_vjp(u, p, w);
            },
            "u"_a, "p"_a, "w"_a)
        .def(
            "state_pattern_at",
            [](const ParametricEnergy& model, const ConstVecRef& u, const ConstVecRef& p) {
                return dynamic_cast<const PyParametricEnergy&>(model).default_state_pattern(u, p);
            },
            "u"_a, "p"_a)
        .def(
            "parameter_pattern_at",
            [](const ParametricEnergy& model, const ConstVecRef& u, const ConstVecRef& p) {
                return dynamic_cast<const PyParametricEnergy&>(model).default_parameter_pattern(u, p);
            },
            "u"_a, "p"_a)
        .def(
            "eval_feasible_step",
            [](const ParametricEnergy&, const ConstVecRef&, const ConstVecRef&, const ConstVecRef&) { return 1.; },
            "u"_a, "p"_a, "du"_a);
}
} // namespace litepgo::energy::binding
