#pragma once

#include "litepgo/energy/energy.hpp"

#include <memory>
#include <vector>

namespace litepgo {

// E(z) = F(T z + offset): the inner energy F seen through an affine map, such as
// periodic or kinematic boundary conditions acting on independent fluctuations.
// The gradient is T^T g and the Hessian T^T H T. The pattern is fixed if F's is.
//
// T^T H T is not formed by sparse products. A contribution table, built once per
// inner pattern, lists for each stored entry of the result which entries of H
// contribute and with which coefficient T(i, a) T(j, b); evaluating sums over it
// without allocating. With T's rows holding one or two entries (selection and
// periodic maps), the table is about as long as H has stored entries.
class AffineMappedEnergy final : public Energy {
  public:
    // T must have energy->num_dofs() rows; z has T.cols() entries. An empty offset
    // means zero. Independence of T's columns is not checked.
    AffineMappedEnergy(std::shared_ptr<Energy> energy, SpMat T, Vec offset = Vec());

  private:
    struct Contribution {
        int output;
        int input;
        double coefficient;
    };
    struct HessianPlan {
        SpMat pattern;
        std::vector<Contribution> contributions;
    };
    struct Setup {
        std::shared_ptr<Energy> energy;
        SpMat map;
        Vec offset;
        HessianPlan plan;
    };

    static Setup checked_setup(std::shared_ptr<Energy> energy, SpMat T, Vec offset);
    static HessianPlan plan_hessian(const SpMat& inner_pattern, const SpMat& map);
    explicit AffineMappedEnergy(Setup setup);

    std::optional<SpMat> refresh_pattern(const ConstVecRef& z) override;
    double evaluate_values(const ConstVecRef& z, Vec* gradient, SpMat* hessian, HessianKind kind) const override;
    double feasible_step(const ConstVecRef& z, const ConstVecRef& dz) const override;

    std::shared_ptr<Energy> energy_;
    SpMat map_;
    Vec offset_;
    std::vector<Contribution> contributions_;

    // Scratch in the inner energy's coordinates. hessian_full_ has the inner
    // pattern the contribution table was planned for.
    mutable Vec x_full_;
    mutable Vec dx_full_;
    mutable Vec gradient_full_;
    mutable SpMat hessian_full_;
};

} // namespace litepgo
