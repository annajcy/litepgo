#pragma once
#include "litepgo/energy/response/mesh_response.hpp"
#include "litepgo/energy/scalar/mesh_energy.hpp"
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/vector.h>
#include <nanobind/trampoline.h>
namespace litepgo::energy::binding {
namespace nb = nanobind;
class PyMeshResponse : public MeshResponse {
  public:
    NB_TRAMPOLINE(MeshResponse);
    BatchMatrix default_jvp(const std::vector<int>& ids, const BatchMatrix& x, const BatchMatrix& d) const {
        return MeshResponse::eval_jvp_batch(ids, x, d);
    }
    BatchMatrix default_vjp(const std::vector<int>& ids, const BatchMatrix& x, const BatchMatrix& d) const {
        return MeshResponse::eval_vjp_batch(ids, x, d);
    }

  protected:
    BatchMatrix eval_batch(std::span<const int> ids, const BatchMatrix& x, BatchMatrix* J) const override {
        nb::detail::ticket ticket(nb_trampoline, "eval_batch", nb::detail::str_hash("eval_batch"), true);
        nb::object result = nb_trampoline.base().attr(ticket.key)(std::vector<int>(ids.begin(), ids.end()),
                                                                  nb::cast(x, nb::rv_policy::copy), J != nullptr);
        if (result.is_none())
            throw std::domain_error("MeshResponse: infeasible batch");
        if (!J)
            return nb::cast<BatchMatrix>(result);
        if (!nb::isinstance<nb::tuple>(result) || nb::len(result) != 2)
            throw nb::type_error("Expected (values, Jacobians)");
        auto pair = nb::cast<nb::tuple>(result);
        *J = nb::cast<BatchMatrix>(pair[1]);
        return nb::cast<BatchMatrix>(pair[0]);
    }
    BatchMatrix eval_jvp_batch(std::span<const int> ids, const BatchMatrix& x, const BatchMatrix& d) const override {
        nb::detail::ticket ticket(nb_trampoline, "eval_jvp_batch", nb::detail::str_hash("eval_jvp_batch"), false);
        if (!ticket.key.is_valid())
            return MeshResponse::eval_jvp_batch(ids, x, d);
        return nb::cast<BatchMatrix>(nb_trampoline.base().attr(ticket.key)(std::vector<int>(ids.begin(), ids.end()),
                                                                           nb::cast(x, nb::rv_policy::copy),
                                                                           nb::cast(d, nb::rv_policy::copy)));
    }
    BatchMatrix eval_vjp_batch(std::span<const int> ids, const BatchMatrix& x, const BatchMatrix& d) const override {
        nb::detail::ticket ticket(nb_trampoline, "eval_vjp_batch", nb::detail::str_hash("eval_vjp_batch"), false);
        if (!ticket.key.is_valid())
            return MeshResponse::eval_vjp_batch(ids, x, d);
        return nb::cast<BatchMatrix>(nb_trampoline.base().attr(ticket.key)(std::vector<int>(ids.begin(), ids.end()),
                                                                           nb::cast(x, nb::rv_policy::copy),
                                                                           nb::cast(d, nb::rv_policy::copy)));
    }
};
class PyMeshEnergy : public MeshEnergy {
  public:
    NB_TRAMPOLINE(MeshEnergy);
    Vec default_step(const std::vector<int>& ids, const BatchMatrix& x, const BatchMatrix& d) const {
        return MeshEnergy::eval_feasible_step_batch(ids, x, d);
    }

  protected:
    Vec eval_batch(std::span<const int> ids, const BatchMatrix& x, BatchMatrix* g, BatchMatrix* H) const override {
        nb::detail::ticket ticket(nb_trampoline, "eval_batch", nb::detail::str_hash("eval_batch"), true);
        nb::object result = nb_trampoline.base().attr(ticket.key)(
            std::vector<int>(ids.begin(), ids.end()), nb::cast(x, nb::rv_policy::copy), g != nullptr, H != nullptr);
        if (result.is_none())
            throw std::domain_error("MeshEnergy: infeasible batch");
        if (!nb::isinstance<nb::tuple>(result) || nb::len(result) != 3)
            throw nb::type_error("Expected (values, gradients, Hessians)");
        auto tuple = nb::cast<nb::tuple>(result);
        if (g)
            *g = nb::cast<BatchMatrix>(tuple[1]);
        if (H)
            *H = nb::cast<BatchMatrix>(tuple[2]);
        return nb::cast<Vec>(tuple[0]);
    }
    Vec eval_feasible_step_batch(std::span<const int> ids, const BatchMatrix& x, const BatchMatrix& dx) const override {
        nb::detail::ticket ticket(nb_trampoline, "eval_feasible_step_batch",
                                  nb::detail::str_hash("eval_feasible_step_batch"), false);
        if (!ticket.key.is_valid())
            return MeshEnergy::eval_feasible_step_batch(ids, x, dx);
        return nb::cast<Vec>(nb_trampoline.base().attr(ticket.key)(std::vector<int>(ids.begin(), ids.end()),
                                                                   nb::cast(x, nb::rv_policy::copy),
                                                                   nb::cast(dx, nb::rv_policy::copy)));
    }
};
} // namespace litepgo::energy::binding
