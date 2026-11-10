#include "litepgo/energy/scalar/affine_mapped_energy.hpp"

#include "litepgo/common/busy_scope.hpp"
#include "litepgo/energy/scalar/detail/evaluation_access.hpp"
#include "litepgo/linalg/sparse_pattern.hpp"

#include <algorithm>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace litepgo::energy {

AffineMappedEnergy::AffineMappedEnergy(std::shared_ptr<Energy> energy, SpMat T, Vec offset)
    : Energy(static_cast<int>(T.cols())), energy_(std::move(energy)), map_(std::move(T)), offset_(std::move(offset)) {
    if (energy_ == nullptr) {
        throw std::invalid_argument("AffineMappedEnergy requires an energy");
    }
    if (map_.rows() != energy_->num_dofs()) {
        throw std::invalid_argument("AffineMappedEnergy: T must have energy.num_dofs rows");
    }
    if (offset_.size() == 0) {
        offset_ = Vec::Zero(map_.rows());
    }
    if (offset_.size() != map_.rows()) {
        throw std::invalid_argument("AffineMappedEnergy: offset must have energy.num_dofs entries");
    }
    map_.makeCompressed();
    if (!linalg::is_canonical(map_) || !map_.coeffs().allFinite() || !offset_.allFinite())
        throw std::invalid_argument("AffineMappedEnergy: invalid finite map or offset");
}

AffineMappedEnergy::HessianPlan AffineMappedEnergy::plan_hessian(const SpMat& inner_pattern, const SpMat& map) {
    // T's rows are walked, so plan from a row-major copy made only for planning.
    using RowMajorSpMat = Eigen::SparseMatrix<double, Eigen::RowMajor, int>;
    const RowMajorSpMat map_rows = map;
    const int num_dofs = static_cast<int>(map.cols());
    // (T^T H T)(a, b) = sum over stored H(i, j) of T(i, a) H(i, j) T(j, b).
    struct Entry {
        int column;
        int row;
        int input;
        double coefficient;
    };
    std::vector<Entry> entries;
    for (int j = 0; j < inner_pattern.outerSize(); ++j) {
        for (int k = inner_pattern.outerIndexPtr()[j]; k < inner_pattern.outerIndexPtr()[j + 1]; ++k) {
            const int i = inner_pattern.innerIndexPtr()[k];
            for (RowMajorSpMat::InnerIterator t_ia(map_rows, i); t_ia; ++t_ia) {
                for (RowMajorSpMat::InnerIterator t_jb(map_rows, j); t_jb; ++t_jb) {
                    entries.push_back(
                        {static_cast<int>(t_jb.col()), static_cast<int>(t_ia.col()), k, t_ia.value() * t_jb.value()});
                }
            }
        }
    }
    std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
        return std::tie(a.column, a.row, a.input) < std::tie(b.column, b.row, b.input);
    });

    std::vector<int> outer(static_cast<size_t>(num_dofs) + 1, 0);
    std::vector<int> inner;
    std::vector<Contribution> contributions;
    contributions.reserve(entries.size());
    for (size_t e = 0; e < entries.size(); ++e) {
        const bool new_position =
            e == 0 || entries[e].column != entries[e - 1].column || entries[e].row != entries[e - 1].row;
        if (new_position) {
            inner.push_back(entries[e].row);
            outer[static_cast<size_t>(entries[e].column) + 1] += 1;
        }
        contributions.push_back({static_cast<int>(inner.size()) - 1, entries[e].input, entries[e].coefficient});
    }
    for (int column = 0; column < num_dofs; ++column) {
        outer[static_cast<size_t>(column) + 1] += outer[static_cast<size_t>(column)];
    }

    const std::vector<double> values(inner.size(), 0.0);
    SpMat pattern = Eigen::Map<const SpMat>(num_dofs, num_dofs, static_cast<Eigen::Index>(inner.size()), outer.data(),
                                            inner.data(), values.data());
    return {std::move(pattern), std::move(contributions)};
}

class AffineMappedEnergy::Evaluation final : public EnergyEvaluation {
  public:
    Evaluation(const AffineMappedEnergy& model, const ConstVecRef& x)
        : EnergyEvaluation(model, x), model_(model), child_(model.energy_->evaluate(model.map_ * x + model.offset_)) {}

  private:
    double feasible_step_impl(const ConstVecRef& dx) const override {
        return child_->max_feasible_step(model_.map_ * dx);
    }
    EnergyValues compute(bool gradient, bool hessian) override {
        return assemble(gradient, hessian, false);
    }
    EnergyValues compute_projected() override {
        return assemble(true, true, true);
    }
    EnergyValues assemble(bool gradient, bool hessian, bool project) {
        ::litepgo::detail::BusyScope scope(model_.busy_, "AffineMappedEnergy");
        const SpMat* H = nullptr;
        if (hessian)
            H = &(project ? detail::EvaluationAccess::projected_hessian(*child_)
                          : detail::EvaluationAccess::hessian(*child_));
        EnergyValues result{0., std::nullopt, std::nullopt};
        if (gradient)
            result.gradient = model_.map_.transpose() * detail::EvaluationAccess::gradient(*child_);
        result.value = child_->value();
        if (hessian) {
            if (!model_.plan_ || !linalg::same_pattern(*H, model_.plan_->inner_pattern)) {
                CachedPlan next{linalg::pattern_of(*H), model_.plan_hessian(*H, model_.map_)};
                model_.plan_ = std::move(next);
            }
            result.hessian = model_.plan_->hessian.pattern;
            for (const auto& contribution : model_.plan_->hessian.contributions)
                result.hessian->valuePtr()[contribution.output] +=
                    contribution.coefficient * H->valuePtr()[contribution.input];
        }
        return result;
    }
    const AffineMappedEnergy& model_;
    std::shared_ptr<EnergyEvaluation> child_;
};
std::shared_ptr<EnergyEvaluation> AffineMappedEnergy::make_evaluation(const ConstVecRef& x) const {
    return std::make_shared<Evaluation>(*this, x);
}

} // namespace litepgo::energy
