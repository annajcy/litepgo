#pragma once
#include "litepgo/common/types.hpp"
#include <memory>
#include <optional>
namespace litepgo::energy {
class Energy;
struct EnergyValues {
    double value;
    std::optional<Vec> gradient;
    std::optional<SpMat> hessian;
};
// One copied input; borrows its model. Not concurrently callable.
class EnergyEvaluation {
  public:
    virtual ~EnergyEvaluation() = default;
    double value();
    Vec gradient();
    SpMat hessian();
    SpMat projected_hessian();
    double max_feasible_step(const ConstVecRef& direction) const;
    int num_dofs() const noexcept;

  protected:
    EnergyEvaluation(const Energy& energy, Vec x);
    virtual EnergyValues compute(bool gradient, bool hessian);
    virtual EnergyValues compute_projected();
    virtual double feasible_step_impl(const ConstVecRef& direction) const;
    const Energy& energy_;
    Vec x_;

  private:
    friend class Energy;
    void save(EnergyValues result, bool gradient, bool hessian, bool projected);
    std::optional<double> value_;
    std::optional<Vec> gradient_;
    std::optional<SpMat> hessian_, projected_;
};
// Evaluations borrow this model, which must outlive them and remain unchanged.
class Energy {
  public:
    virtual ~Energy() = default;
    Energy(const Energy&) = delete;
    Energy& operator=(const Energy&) = delete;
    int num_dofs() const noexcept {
        return num_dofs_;
    }
    std::shared_ptr<EnergyEvaluation> evaluate(const ConstVecRef& x) const;

  protected:
    explicit Energy(int num_dofs);
    virtual std::shared_ptr<EnergyEvaluation> make_evaluation(const ConstVecRef& x) const;
    virtual EnergyValues eval_impl(const ConstVecRef& x, bool gradient, bool hessian) const;
    virtual EnergyValues eval_projected_values(const ConstVecRef& x, EnergyEvaluation& exact) const;
    virtual double feasible_step(const ConstVecRef& x, const ConstVecRef& dx) const;

  private:
    friend class EnergyEvaluation;
    int num_dofs_;
};
// Spectral PSD projection; may produce a dense sparsity pattern.
SpMat project_hessian(const SpMat& hessian);
} // namespace litepgo::energy
