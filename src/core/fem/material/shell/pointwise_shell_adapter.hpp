#pragma once
#include "litepgo/fem/material/shell/shell_material_batch.hpp"

namespace litepgo::fem {
// Material must implement every pointwise derivative, including both mixed derivatives.
// Only requested outputs are evaluated.
template <class Material> class PointwiseShellAdapter final : public ShellBatchMaterial {
  public:
    int num_parameters() const override {
        return Material::num_parameters;
    }
    void eval(const ShellMaterialBatch& in, ShellMaterialBatchOutputs& out) const override {
        detail::check_shell_batch_inputs(in, num_parameters());
        const Eigen::Index n = in.size();
        if (out.W)
            out.W->resize(n);
        if (out.dW_dab)
            out.dW_dab->resize(6, n);
        if (out.d2W_dab2)
            out.d2W_dab2->resize(36, n);
        if (out.d2W_dab_dtheta)
            out.d2W_dab_dtheta->resize(6 * num_parameters(), n);
        if (out.d2W_dab_dabarbbar)
            out.d2W_dab_dabarbbar->resize(36, n);
        ShellMaterialInputs input;
        input.theta.resize(num_parameters());
        for (Eigen::Index p = 0; p < n; ++p) {
            input.a = unpack(in.ab.col(p).head<3>());
            input.b = unpack(in.ab.col(p).tail<3>());
            input.a_bar = unpack(in.ab_bar.col(p).head<3>());
            input.b_bar = unpack(in.ab_bar.col(p).tail<3>());
            input.thickness = in.thickness[p];
            input.theta = in.theta.col(p);
            if (out.W)
                (*out.W)[p] = Material::eval_W(input);
            if (out.dW_dab)
                out.dW_dab->col(p) = Material::eval_dW_dab(input);
            if (out.d2W_dab2)
                out.d2W_dab2->col(p) = Material::eval_d2W_dab2(input).reshaped();
            if (out.d2W_dab_dtheta)
                out.d2W_dab_dtheta->col(p) = Material::eval_d2W_dab_dtheta(input).reshaped();
            if (out.d2W_dab_dabarbbar)
                out.d2W_dab_dabarbbar->col(p) = Material::eval_d2W_dab_dabarbbar(input).reshaped();
        }
    }

  private:
    static Eigen::Matrix2d unpack(const Eigen::Vector3d& packed) {
        return (Eigen::Matrix2d() << packed[0], packed[1], packed[1], packed[2]).finished();
    }
};
} // namespace litepgo::fem
