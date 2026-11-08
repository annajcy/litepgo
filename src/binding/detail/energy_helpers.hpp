#pragma once
#include "detail/sparse_matrix.hpp"
#include "litepgo/energy/scalar/energy.hpp"
#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/trampoline.h>
#include <stdexcept>
namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::energy::binding {
// Canonical CSC inputs: merge each column in linear time, preserving output structure.
inline void copy_derivative_values(const SpMat& source, SpMat& output, const char* invalid, const char* outside) {
    if (source.rows() != output.rows() || source.cols() != output.cols() || !source.coeffs().allFinite())
        throw std::logic_error(invalid);
    output.coeffs().setZero();
    for (int c = 0; c < source.outerSize(); ++c) {
        SpMat::InnerIterator slot(output, c);
        for (SpMat::InnerIterator it(source, c); it; ++it) {
            while (slot && slot.row() < it.row())
                ++slot;
            if (slot && slot.row() == it.row())
                slot.valueRef() = it.value();
            else if (it.value() != 0)
                throw std::logic_error(outside);
        }
    }
}

// Both bases share scalar evaluation and feasibility; only the dynamic base requests hessian_pattern_at.
template <class Base> class PyEnergy : public Base {
  public:
    NB_TRAMPOLINE(Base);
    explicit PyEnergy(const linalg::binding::SparseArgument& pattern) : Base(pattern.matrix) {}
    explicit PyEnergy(int n) : Base(n) {}

  private:
    double eval_values(const ConstVecRef& x, Vec* g, SpMat* H, HessianKind kind) const override {
        nb::gil_scoped_acquire gil;
        if (!nb::hasattr(this->nb_trampoline.base(), "eval_values"))
            throw std::logic_error("Energy: missing eval_values");
        nb::object out = this->nb_trampoline.base().attr("eval_values")(nb::cast(Vec(x)), g != nullptr,
                                                                        H ? nb::cast(kind) : nb::none());
        if (out.is_none())
            throw std::domain_error("Energy is infeasible");
        if (!nb::isinstance<nb::tuple>(out) || nb::len(out) != 3)
            throw nb::type_error("Expected (value, gradient, Hessian)");
        nb::tuple values = nb::cast<nb::tuple>(out);
        if (g) {
            *g = nb::cast<Vec>(values[1]);
            if (g->size() != this->num_dofs() || !g->allFinite())
                throw std::logic_error("Energy: invalid gradient");
        }
        if (H) {
            const SpMat source = nb::cast<linalg::binding::SparseArgument>(values[2]).matrix;
            copy_derivative_values(source, *H, "Energy: invalid Hessian", "Energy: Hessian outside pattern");
        }
        return nb::cast<double>(values[0]);
    }
    double feasible_step(const ConstVecRef& x, const ConstVecRef& dx) const override {
        nb::detail::ticket ticket(this->nb_trampoline, "feasible_step", nb::detail::str_hash("feasible_step"), false);
        if (!ticket.key.is_valid())
            return 1.0;
        return nb::cast<double>(this->nb_trampoline.base().attr(ticket.key)(nb::cast(Vec(x)), nb::cast(Vec(dx))));
    }
};

template <class Base> class PyResponse : public Base {
  public:
    NB_TRAMPOLINE(Base);
    explicit PyResponse(const linalg::binding::SparseArgument& pattern,
                        const std::optional<linalg::binding::SparseArgument>& hessian_pattern = std::nullopt)
        : Base(pattern.matrix, hessian_pattern ? std::optional<SpMat>(hessian_pattern->matrix) : std::nullopt) {}
    PyResponse(int num_inputs, int num_outputs) : Base(num_inputs, num_outputs) {}
    Vec default_jvp(const ConstVecRef& x, const ConstVecRef& dx) {
        return Base::eval_jvp_impl(x, dx);
    }
    Vec default_vjp(const ConstVecRef& x, const ConstVecRef& gy) {
        return Base::eval_vjp_impl(x, gy);
    }

  private:
    void eval_hessian_impl(const ConstVecRef& x, const ConstVecRef& weights, SpMat& H) override {
        nb::gil_scoped_acquire gil;
        nb::handle self = this->nb_trampoline.base();
        if (!nb::hasattr(self, "eval_hessian_impl"))
            throw std::logic_error("Response: missing eval_hessian_impl");
        nb::object result = self.attr("eval_hessian_impl")(nb::cast(Vec(x)), nb::cast(Vec(weights)));
        if (result.is_none())
            throw std::domain_error("Response is infeasible");
        const SpMat source = nb::cast<linalg::binding::SparseArgument>(result).matrix;
        copy_derivative_values(source, H, "Response: invalid Hessian", "Response: Hessian outside pattern");
    }
    Vec eval_jvp_impl(const ConstVecRef& x, const ConstVecRef& dx) override {
        nb::gil_scoped_acquire gil;
        // The bound base method selects the fallback; Python overrides may reenter normally.
        nb::object result = this->nb_trampoline.base().attr("eval_jvp_impl")(nb::cast(Vec(x)), nb::cast(Vec(dx)));
        if (result.is_none())
            throw std::domain_error("Response is infeasible");
        return nb::cast<Vec>(result);
    }
    Vec eval_vjp_impl(const ConstVecRef& x, const ConstVecRef& gy) override {
        nb::gil_scoped_acquire gil;
        // The bound base method selects the fallback; Python overrides may reenter normally.
        nb::object result = this->nb_trampoline.base().attr("eval_vjp_impl")(nb::cast(Vec(x)), nb::cast(Vec(gy)));
        if (result.is_none())
            throw std::domain_error("Response is infeasible");
        return nb::cast<Vec>(result);
    }
    Vec eval_impl(const ConstVecRef& x, SpMat* J) override {
        using linalg::binding::SparseArgument;
        nb::gil_scoped_acquire gil;
        nb::handle self = this->nb_trampoline.base();
        if (!nb::hasattr(self, "eval_impl"))
            throw std::logic_error("Response: missing eval_impl");
        nb::object value = self.attr("eval_impl")(nb::cast(Vec(x)), J != nullptr);
        if (value.is_none())
            throw std::domain_error("Response is infeasible");
        if (!J)
            return nb::cast<Vec>(value);
        if (!nb::isinstance<nb::tuple>(value) || nb::len(value) != 2)
            throw nb::type_error("Expected (value, Jacobian)");
        nb::tuple pair = nb::cast<nb::tuple>(value);
        const SpMat source = nb::cast<SparseArgument>(pair[1]).matrix;
        copy_derivative_values(source, *J, "Response: invalid Jacobian", "Response: Jacobian outside pattern");
        return nb::cast<Vec>(pair[0]);
    }
};
} // namespace litepgo::energy::binding
