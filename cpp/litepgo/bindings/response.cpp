#include "litepgo/bindings/bindings.hpp"
#include "litepgo/bindings/sparse_matrix.hpp"
#include "litepgo/dof/constant_dof_map.hpp"
#include "litepgo/dof/element_dof_map.hpp"
#include "litepgo/dof/stencil_dof_map.hpp"
#include "litepgo/dof/vertex_dof_map.hpp"
#include "litepgo/response/dynamic_pattern_response.hpp"
#include "litepgo/response/fixed_pattern_response.hpp"
#include "litepgo/response/mesh_response.hpp"
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/vector.h>
#include <nanobind/trampoline.h>
namespace nb = nanobind;
using namespace nb::literals;
namespace litepgo::response::binding {
template <class Base> class PyResponse : public Base {
  public:
    NB_TRAMPOLINE(Base);
    explicit PyResponse(const linalg::binding::SparseArgument& pattern) : Base(pattern.matrix) {}

  private:
    Vec eval_values(const ConstVecRef& x, SpMat* J) const override {
        using linalg::binding::SparseArgument;
        nb::detail::ticket ticket(this->nb_trampoline, "eval_values", nb::detail::str_hash("eval_values"), true);
        nb::object value = this->nb_trampoline.base().attr(ticket.key)(nb::cast(Vec(x)), J != nullptr);
        if (value.is_none())
            throw std::domain_error("Response is infeasible");
        if (!J)
            return nb::cast<Vec>(value);
        if (!nb::isinstance<nb::tuple>(value) || nb::len(value) != 2)
            throw nb::type_error("Expected (value, Jacobian)");
        nb::tuple pair = nb::cast<nb::tuple>(value);
        const SpMat source = nb::cast<SparseArgument>(pair[1]).matrix;
        if (source.rows() != J->rows() || source.cols() != J->cols() || !source.coeffs().allFinite())
            throw std::logic_error("Response: invalid Jacobian");
        // Both patterns are canonical: merge each column in linear time, retaining zero slots.
        J->coeffs().setZero();
        for (int c = 0; c < source.outerSize(); ++c) {
            SpMat::InnerIterator slot(*J, c);
            for (SpMat::InnerIterator it(source, c); it; ++it) {
                while (slot && slot.row() < it.row())
                    ++slot;
                if (slot && slot.row() == it.row())
                    slot.valueRef() = it.value();
                else if (it.value() != 0)
                    throw std::logic_error("Response: Jacobian outside pattern");
            }
        }
        return nb::cast<Vec>(pair[0]);
    }
};

class PyDynamicResponse : public PyResponse<DynamicPatternResponse> {
  public:
    using PyResponse<DynamicPatternResponse>::PyResponse;

  private:
    SpMat pattern_at(const ConstVecRef& x) const override {
        nb::detail::ticket ticket(nb_trampoline, "pattern_at", nb::detail::str_hash("pattern_at"), true);
        return nb::cast<linalg::binding::SparseArgument>(nb_trampoline.base().attr(ticket.key)(nb::cast(Vec(x))))
            .matrix;
    }
};

template <class Input, class Output, class Cls> void bind_batch_init(Cls& cls) {
    cls.def(
        "__init__",
        [](MeshResponse* self, const Input& input, const Output& output, nb::callable evaluator, int batch_size) {
            new (self) MeshResponse(
                input, output,
                [evaluator](std::span<const int> ids, const BatchMatrix& x, BatchMatrix* J) -> BatchMatrix {
                    nb::gil_scoped_acquire acquire;
                    nb::object value = evaluator(std::vector<int>(ids.begin(), ids.end()),
                                                 nb::cast(x, nb::rv_policy::copy), J != nullptr);
                    if (value.is_none())
                        throw std::domain_error("MeshResponse: infeasible batch");
                    if (!J)
                        return nb::cast<BatchMatrix>(value);
                    if (!nb::isinstance<nb::tuple>(value) || nb::len(value) != 2)
                        throw nb::type_error("Expected (batch_values, batch_Jacobians)");
                    nb::tuple pair = nb::cast<nb::tuple>(value);
                    *J = nb::cast<BatchMatrix>(pair[1]);
                    return nb::cast<BatchMatrix>(pair[0]);
                },
                batch_size);
        },
        "input_map"_a, "output_map"_a, "eval_batch"_a, "batch_size"_a = 256);
}
template <class Input, class Cls> void bind_batch_outputs(Cls& cls) {
    bind_batch_init<Input, dof::ConstantDofMap>(cls);
    bind_batch_init<Input, dof::ElementDofMap>(cls);
    bind_batch_init<Input, dof::VertexDofMap>(cls);
    bind_batch_init<Input, dof::StencilDofMap>(cls);
}

void bind(nb::module_& m) {
    using linalg::binding::SparseArgument;
    nb::class_<Response>(m, "Response")
        .def_prop_ro("num_inputs", &Response::num_inputs)
        .def_prop_ro("num_outputs", &Response::num_outputs)
        .def_prop_ro("has_fixed_pattern", &Response::has_fixed_pattern)
        .def_prop_ro("jacobian_pattern", [](const Response& r) { return r.jacobian_pattern(); })
        .def("update_pattern", &Response::update_pattern, "x"_a, nb::call_guard<nb::gil_scoped_release>())
        .def("eval_value", &Response::eval_value, "x"_a, nb::call_guard<nb::gil_scoped_release>())
        .def("eval_jacobian", &Response::eval_jacobian, "x"_a, nb::call_guard<nb::gil_scoped_release>())
        .def(
            "eval",
            [](const Response& r, const ConstVecRef& x, bool need_jacobian) {
                SpMat J = r.jacobian_pattern();
                Vec value;
                {
                    nb::gil_scoped_release release;
                    value = r.eval(x, need_jacobian ? &J : nullptr);
                }
                return nb::make_tuple(value, need_jacobian ? nb::cast(J) : nb::none());
            },
            "x"_a, "need_jacobian"_a = false);
    nb::class_<FixedPatternResponse, Response, PyResponse<FixedPatternResponse>>(m, "FixedPatternResponse")
        .def(nb::init<const SparseArgument&>(), "pattern"_a,
             "Subclass and override eval_values(x, need_jacobian). Return value or (value, J); None means infeasible.");
    nb::class_<DynamicPatternResponse, Response, PyDynamicResponse>(m, "DynamicPatternResponse")
        .def(nb::init<const SparseArgument&>(), "initial_pattern"_a,
             "Subclass and override pattern_at(x) and eval_values(x, need_jacobian). Dimensions remain fixed.");
    auto batch = nb::class_<MeshResponse, Response>(m, "MeshResponse");
    bind_batch_outputs<dof::ConstantDofMap>(batch);
    bind_batch_outputs<dof::ElementDofMap>(batch);
    bind_batch_outputs<dof::VertexDofMap>(batch);
    bind_batch_outputs<dof::StencilDofMap>(batch);
}
} // namespace litepgo::response::binding
