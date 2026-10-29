#include "litepgo/energy/affine_mapped_energy.hpp"

#include "litepgo/linalg/sparse_pattern.hpp"

#include <algorithm>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace litepgo {

AffineMappedEnergy::Setup AffineMappedEnergy::checked_setup(std::shared_ptr<Energy> energy, SpMat T, Vec offset) {
    if (energy == nullptr) {
        throw std::invalid_argument("AffineMappedEnergy requires an energy");
    }
    if (T.rows() != energy->num_dofs()) {
        throw std::invalid_argument("AffineMappedEnergy: T must have energy.num_dofs rows");
    }
    if (offset.size() == 0) {
        offset = Vec::Zero(T.rows());
    }
    if (offset.size() != T.rows()) {
        throw std::invalid_argument("AffineMappedEnergy: offset must have energy.num_dofs entries");
    }
    T.makeCompressed();
    HessianPlan plan = plan_hessian(energy->hessian_pattern(), T);
    return {std::move(energy), std::move(T), std::move(offset), std::move(plan)};
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

AffineMappedEnergy::AffineMappedEnergy(std::shared_ptr<Energy> energy, SpMat T, Vec offset)
    : AffineMappedEnergy(checked_setup(std::move(energy), std::move(T), std::move(offset))) {}

AffineMappedEnergy::AffineMappedEnergy(Setup setup)
    : Energy(static_cast<int>(setup.map.cols()), setup.plan.pattern, setup.energy->has_fixed_pattern()),
      energy_(std::move(setup.energy)), map_(std::move(setup.map)), offset_(std::move(setup.offset)),
      contributions_(std::move(setup.plan.contributions)), hessian_full_(energy_->hessian_pattern()) {}

std::optional<SpMat> AffineMappedEnergy::refresh_pattern(const ConstVecRef& z) {
    x_full_ = map_ * z + offset_;
    energy_->update_pattern(x_full_);
    // Compare patterns rather than trust the return value: the inner energy may
    // have been updated elsewhere, in which case its own update reports no change.
    if (same_pattern(energy_->hessian_pattern(), hessian_full_)) {
        return std::nullopt;
    }
    hessian_full_ = energy_->hessian_pattern();
    HessianPlan plan = plan_hessian(hessian_full_, map_);
    contributions_ = std::move(plan.contributions);
    return std::move(plan.pattern);
}

double AffineMappedEnergy::eval_values(const ConstVecRef& z, Vec* gradient, SpMat* hessian, HessianKind kind) const {
    if (hessian != nullptr && !energy_->has_fixed_pattern() &&
        !same_pattern(energy_->hessian_pattern(), hessian_full_)) {
        throw std::logic_error("AffineMappedEnergy: the inner energy's pattern changed after this energy's last "
                               "update_pattern(); call update_pattern() on the outermost energy");
    }
    x_full_ = map_ * z + offset_;
    const double value = energy_->eval(x_full_, gradient != nullptr ? &gradient_full_ : nullptr,
                                       hessian != nullptr ? &hessian_full_ : nullptr, kind);
    if (gradient != nullptr) {
        *gradient = map_.transpose() * gradient_full_;
    }
    if (hessian != nullptr) {
        double* values = hessian->valuePtr();
        const double* inner_values = hessian_full_.valuePtr();
        std::fill(values, values + hessian->nonZeros(), 0.0);
        for (const Contribution& contribution : contributions_) {
            values[contribution.output] += contribution.coefficient * inner_values[contribution.input];
        }
    }
    return value;
}

double AffineMappedEnergy::feasible_step(const ConstVecRef& z, const ConstVecRef& dz) const {
    x_full_ = map_ * z + offset_;
    dx_full_ = map_ * dz;
    return energy_->max_feasible_step(x_full_, dx_full_);
}

} // namespace litepgo
