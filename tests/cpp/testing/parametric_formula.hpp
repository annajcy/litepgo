#pragma once
#include "litepgo/energy/parametric/parametric_energy.hpp"
#include <functional>
namespace litepgo::testing {
class FormulaModel final : public energy::ParametricEnergy {
  public:
    using Factory = std::function<std::shared_ptr<energy::Energy>(const Vec&)>;
    using Partial = std::function<SpMat(const Vec&, const Vec&)>;
    FormulaModel(int n, int size, Factory factory, Partial partial)
        : ParametricEnergy(Eigen::MatrixXd::Ones(size, size).sparseView(), Eigen::MatrixXd::Ones(size, n).sparseView()),
          factory_(std::move(factory)), partial_(std::move(partial)) {}

  private:
    double eval_state_values(const ConstVecRef& u, const ConstVecRef& p, Vec* g, SpMat* K,
                             energy::HessianKind kind) const override {
        auto potential = factory_(p);
        if (!potential)
            throw std::invalid_argument("invalid test energy");
        SpMat source = potential->hessian_pattern();
        const double value = potential->eval(u, g, K ? &source : nullptr, kind);
        if (K) {
            for (int c = 0; c < K->outerSize(); ++c)
                for (SpMat::InnerIterator it(*K, c); it; ++it)
                    it.valueRef() = source.coeff(it.row(), c);
        }
        return value;
    }
    void eval_parameter_jacobian(const ConstVecRef& u, const ConstVecRef& p, SpMat& B) const override {
        const auto source = partial_(p, u);
        if (source.rows() != B.rows() || source.cols() != B.cols())
            throw std::invalid_argument("invalid test partial");
        for (int c = 0; c < B.outerSize(); ++c)
            for (SpMat::InnerIterator it(B, c); it; ++it)
                it.valueRef() = source.coeff(it.row(), c);
    }
    double eval_feasible_step(const ConstVecRef& u, const ConstVecRef& p, const ConstVecRef& du) const override {
        auto potential = factory_(p);
        if (!potential)
            throw std::invalid_argument("invalid test energy");
        return potential->max_feasible_step(u, du);
    }
    Factory factory_;
    Partial partial_;
};
} // namespace litepgo::testing
