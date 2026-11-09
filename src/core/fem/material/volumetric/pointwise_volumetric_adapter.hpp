#pragma once
#include "litepgo/fem/material/volumetric/volumetric_material_batch.hpp"
#include "litepgo/fem/material/volumetric/volumetric_material_inputs.hpp"

namespace litepgo::fem {
// Material must implement every pointwise derivative, including both mixed derivatives.
// Only requested outputs are evaluated.
template <class Material> class PointwiseVolumetricAdapter final : public VolumetricBatchMaterial {
  public:
    int num_parameters() const override {
        return Material::num_parameters;
    }
    void eval(const VolumetricMaterialBatch& in, VolumetricMaterialBatchOutputs& out) const override {
        detail::check_volumetric_batch_inputs(in, num_parameters());
        const Eigen::Index n = in.size();
        if (out.W)
            out.W->resize(n);
        if (out.dW_dF)
            out.dW_dF->resize(9, n);
        if (out.d2W_dF2)
            out.d2W_dF2->resize(81, n);
        if (out.d2W_dF_dtheta)
            out.d2W_dF_dtheta->resize(9 * num_parameters(), n);
        if (out.d2W_dF_dFp)
            out.d2W_dF_dFp->resize(81, n);
        VolumetricMaterialInputs input;
        input.theta.resize(num_parameters());
        for (Eigen::Index p = 0; p < n; ++p) {
            input.F = in.F.col(p).reshaped(3, 3);
            input.Fp = in.Fp.col(p).reshaped(3, 3);
            input.theta = in.theta.col(p);
            if (out.W)
                (*out.W)[p] = Material::eval_W(input);
            if (out.dW_dF)
                out.dW_dF->col(p) = Material::eval_dW_dF(input).reshaped();
            if (out.d2W_dF2)
                out.d2W_dF2->col(p) = Material::eval_d2W_dF2(input).reshaped();
            if (out.d2W_dF_dtheta)
                out.d2W_dF_dtheta->col(p) = Material::eval_d2W_dF_dtheta(input).reshaped();
            if (out.d2W_dF_dFp)
                out.d2W_dF_dFp->col(p) = Material::eval_d2W_dF_dFp(input).reshaped();
        }
    }
    Vec eval_feasible_step(const VolumetricMaterialBatch& in,
                           const Eigen::Matrix<double, 9, Eigen::Dynamic>& dF) const override {
        detail::check_volumetric_batch_inputs(in, num_parameters());
        if (dF.cols() != in.size())
            throw std::invalid_argument("Volumetric batch material: incompatible increment dimensions");
        Vec steps(in.size());
        for (Eigen::Index p = 0; p < in.size(); ++p) {
            const VolumetricMaterialInputs input = point(in, p);
            const VolumetricMaterialInputs direction{dF.col(p).reshaped(3, 3), Eigen::Matrix3d::Zero(),
                                                     Vec::Zero(input.theta.size())};
            steps[p] = Material::eval_feasible_step(input, direction);
        }
        return steps;
    }

  private:
    static VolumetricMaterialInputs point(const VolumetricMaterialBatch& in, Eigen::Index p) {
        return {in.F.col(p).reshaped(3, 3), in.Fp.col(p).reshaped(3, 3), in.theta.col(p)};
    }
};
} // namespace litepgo::fem
