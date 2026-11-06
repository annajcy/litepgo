#pragma once
#include "litepgo/common/types.hpp"
#include "litepgo/fem/material/shell/shell_material_inputs.hpp"
#include <concepts>
#include <stdexcept>
#include <string>

namespace litepgo::fem {

// Column n of each input holds one independent face. Forms are packed (a11,a12,a22,b11,b12,b22);
// mixed entries are independent and unscaled, matching the pointwise shell material derivatives.
struct ShellMaterialBatch {
    Eigen::Matrix<double, 6, Eigen::Dynamic> ab;
    Eigen::Matrix<double, 6, Eigen::Dynamic> ab_bar;
    Vec thickness;
    Eigen::MatrixXd theta; // num_parameters rows
    Eigen::Index size() const {
        return ab.cols();
    }
};

// Non-null outputs are requested; the material resizes and overwrites them. Column n stores
// column-major matrices whose rows follow packed current forms: d2W_dab2 and d2W_dab_dabarbbar
// are 6x6, d2W_dab_dtheta is 6 x num_parameters.
struct ShellMaterialBatchOutputs {
    Vec* W = nullptr;
    Eigen::Matrix<double, 6, Eigen::Dynamic>* dW_dab = nullptr;
    Eigen::Matrix<double, 36, Eigen::Dynamic>* d2W_dab2 = nullptr;
    Eigen::MatrixXd* d2W_dab_dtheta = nullptr;
    Eigen::Matrix<double, 36, Eigen::Dynamic>* d2W_dab_dabarbbar = nullptr;
};

// A shell material evaluated once per batch of faces. Static pointwise materials such as KoiterStVK
// are used directly. Line searches use only the geometric shell step, so no feasible-step hook exists.
template <class M>
concept ShellBatchMaterial =
    requires(const M& material, const ShellMaterialBatch& inputs, ShellMaterialBatchOutputs& outputs) {
        { material.num_parameters() } -> std::convertible_to<int>;
        material.eval(inputs, outputs);
    };

inline Eigen::Matrix<double, 6, 1> pack_shell_forms(const Eigen::Matrix2d& a, const Eigen::Matrix2d& b) {
    return (Eigen::Matrix<double, 6, 1>() << a(0, 0), a(0, 1), a(1, 1), b(0, 0), b(0, 1), b(1, 1)).finished();
}

namespace detail {
template <class M> int shell_num_parameters(const M& material) {
    if constexpr (ShellBatchMaterial<M>)
        return material.num_parameters();
    else
        return M::num_parameters;
}

// Validates material-supplied batch outputs before assembly reads them.
inline void check_shell_batch_outputs(const ShellMaterialBatchOutputs& out, Eigen::Index count, int k) {
    auto check = [&](const auto* value, Eigen::Index rows, const char* name) {
        if (!value)
            return;
        if (value->rows() != rows || value->cols() != count)
            throw std::logic_error(std::string("Shell batch material: invalid ") + name + " dimensions");
        if (!value->allFinite())
            throw std::domain_error(std::string("Shell batch material: non-finite ") + name);
    };
    if (out.W && out.W->size() != count)
        throw std::logic_error("Shell batch material: invalid W dimensions");
    if (out.W && !out.W->allFinite())
        throw std::domain_error("Shell batch material: non-finite W");
    check(out.dW_dab, 6, "dW_dab");
    check(out.d2W_dab2, 36, "d2W_dab2");
    check(out.d2W_dab_dtheta, 6 * k, "d2W_dab_dtheta");
    check(out.d2W_dab_dabarbbar, 36, "d2W_dab_dabarbbar");
}

inline void set_shell_point(ShellMaterialBatch& batch, Eigen::Index n, const ShellMaterialInputs& inputs) {
    batch.ab.col(n) = pack_shell_forms(inputs.a, inputs.b);
    batch.ab_bar.col(n) = pack_shell_forms(inputs.a_bar, inputs.b_bar);
    batch.thickness[n] = inputs.thickness;
    batch.theta.col(n) = inputs.theta;
}

inline ShellMaterialBatch make_shell_batch(Eigen::Index count, int parameters) {
    ShellMaterialBatch batch;
    batch.ab.resize(6, count);
    batch.ab_bar.resize(6, count);
    batch.thickness.resize(count);
    batch.theta.resize(parameters, count);
    return batch;
}
} // namespace detail
} // namespace litepgo::fem
