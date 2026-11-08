#pragma once
#include "litepgo/common/types.hpp"
#include <concepts>
#include <stdexcept>
#include <string>

namespace litepgo::fem {

// Column n of each input holds one independent material point. Matrices use column-major vec, i+3*j.
struct VolumetricMaterialBatch {
    Eigen::Matrix<double, 9, Eigen::Dynamic> F;
    Eigen::Matrix<double, 9, Eigen::Dynamic> Fp;
    Eigen::MatrixXd theta; // num_parameters rows
    Eigen::Index size() const {
        return F.cols();
    }
};

// Non-null outputs are requested; the material resizes and overwrites them. Column n stores:
// dW_dF as vec(dW/dF); d2W_dF2 and d2W_dF_dFp as column-major 9x9 matrices with rows vec(dW/dF);
// d2W_dF_dtheta as a column-major 9 x num_parameters matrix.
struct VolumetricMaterialBatchOutputs {
    Vec* W = nullptr;
    Eigen::Matrix<double, 9, Eigen::Dynamic>* dW_dF = nullptr;
    Eigen::Matrix<double, 81, Eigen::Dynamic>* d2W_dF2 = nullptr;
    Eigen::MatrixXd* d2W_dF_dtheta = nullptr;
    Eigen::Matrix<double, 81, Eigen::Dynamic>* d2W_dF_dFp = nullptr;
};

// A material evaluated once per batch of points instead of once per point. Static pointwise
// materials such as NeoHookean are used directly; assembly integrates either kind identically.
// eval_feasible_step receives dF only (Fp and theta are fixed) and returns one [0,1] bound per point.
template <class M>
concept VolumetricBatchMaterial =
    requires(const M& material, const VolumetricMaterialBatch& inputs, VolumetricMaterialBatchOutputs& outputs,
             const Eigen::Matrix<double, 9, Eigen::Dynamic>& dF) {
        { material.num_parameters() } -> std::convertible_to<int>;
        material.eval(inputs, outputs);
        { material.eval_feasible_step(inputs, dF) } -> std::convertible_to<Vec>;
    };

namespace detail {
template <class M> int volumetric_num_parameters(const M& material) {
    if constexpr (VolumetricBatchMaterial<M>)
        return material.num_parameters();
    else
        return M::num_parameters;
}

// Validates material-supplied batch outputs before assembly reads them.
inline void check_volumetric_batch_outputs(const VolumetricMaterialBatchOutputs& out, Eigen::Index count, int k) {
    auto check = [&](const auto* value, Eigen::Index rows, const char* name) {
        if (!value)
            return;
        if (value->rows() != rows || value->cols() != count)
            throw std::logic_error(std::string("Volumetric batch material: invalid ") + name + " dimensions");
        if (!value->allFinite())
            throw std::domain_error(std::string("Volumetric batch material: non-finite ") + name);
    };
    if (out.W && out.W->size() != count)
        throw std::logic_error("Volumetric batch material: invalid W dimensions");
    if (out.W && !out.W->allFinite())
        throw std::domain_error("Volumetric batch material: non-finite W");
    check(out.dW_dF, 9, "dW_dF");
    check(out.d2W_dF2, 81, "d2W_dF2");
    check(out.d2W_dF_dtheta, 9 * k, "d2W_dF_dtheta");
    check(out.d2W_dF_dFp, 81, "d2W_dF_dFp");
}
} // namespace detail
} // namespace litepgo::fem
