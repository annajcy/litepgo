#include "litepgo/energy/response/dynamic_pattern_response.hpp"
#include "bindings.hpp"
#include "detail/energy_helpers.hpp"
#include "detail/sparse_matrix.hpp"
#include <nanobind/eigen/dense.h>
#include <nanobind/trampoline.h>
namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::energy::binding {
namespace {
class PyDynamicResponse : public PyResponse<DynamicPatternResponse> {
  public:
    using PyResponse<DynamicPatternResponse>::PyResponse;
    bool has_hessian() const override {
        nb::detail::ticket ticket(nb_trampoline, "has_hessian", nb::detail::str_hash("has_hessian"), false);
        if (!ticket.key.is_valid())
            return false;
        return nb::cast<bool>(nb_trampoline.base().attr(ticket.key));
    }
    SpMat default_hessian_pattern(const ConstVecRef& x) const {
        return DynamicPatternResponse::hessian_pattern_at(x);
    }

  private:
    SpMat hessian_pattern_at(const ConstVecRef& x) const override {
        nb::detail::ticket ticket(nb_trampoline, "hessian_pattern_at", nb::detail::str_hash("hessian_pattern_at"),
                                  false);
        if (!ticket.key.is_valid())
            return DynamicPatternResponse::hessian_pattern_at(x);
        return nb::cast<linalg::binding::SparseArgument>(nb_trampoline.base().attr(ticket.key)(nb::cast(Vec(x))))
            .matrix;
    }
    SpMat jacobian_pattern_at(const ConstVecRef& x) const override {
        nb::detail::ticket ticket(nb_trampoline, "jacobian_pattern_at", nb::detail::str_hash("jacobian_pattern_at"),
                                  true);
        return nb::cast<linalg::binding::SparseArgument>(nb_trampoline.base().attr(ticket.key)(nb::cast(Vec(x))))
            .matrix;
    }
};
} // namespace

void bind_dynamic_pattern_response(nb::module_& m) {
    using linalg::binding::SparseArgument;
    nb::class_<DynamicPatternResponse, Response, PyDynamicResponse>(m, "DynamicPatternResponse")
        .def(
            "hessian_pattern_at",
            [](DynamicPatternResponse& r, const ConstVecRef& x) {
                return dynamic_cast<const PyDynamicResponse&>(r).default_hessian_pattern(x);
            },
            "x"_a)
        .def(
            "eval_jvp_impl",
            [](DynamicPatternResponse& r, const ConstVecRef& x, const ConstVecRef& d) {
                return dynamic_cast<PyResponse<DynamicPatternResponse>&>(r).default_jvp(x, d);
            },
            "x"_a, "dx"_a)
        .def(
            "eval_vjp_impl",
            [](DynamicPatternResponse& r, const ConstVecRef& x, const ConstVecRef& d) {
                return dynamic_cast<PyResponse<DynamicPatternResponse>&>(r).default_vjp(x, d);
            },
            "x"_a, "weights"_a)
        .def(nb::init<int, int>(), "num_inputs"_a, "num_outputs"_a,
             "Subclass and override jacobian_pattern_at(x) and eval_impl(x, need_jacobian). Dimensions remain fixed.");
}
} // namespace litepgo::energy::binding
