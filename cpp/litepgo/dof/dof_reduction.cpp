#include "litepgo/dof/dof_reduction.hpp"

#include "litepgo/core/memory.hpp"
#include "litepgo/linalg/reduced_matrix.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace litepgo::dof {

DofReduction::DofReduction(int num_dofs, std::vector<int> fixed_dofs)
    : num_dofs_(num_dofs), fixed_dofs_(std::move(fixed_dofs)) {
    if (num_dofs_ < 0) {
        throw std::invalid_argument("DofReduction requires a non-negative DOF count");
    }
    std::sort(fixed_dofs_.begin(), fixed_dofs_.end());
    if (std::adjacent_find(fixed_dofs_.begin(), fixed_dofs_.end()) != fixed_dofs_.end()) {
        throw std::invalid_argument("DofReduction fixed DOFs must be unique");
    }
    if (!fixed_dofs_.empty() && (fixed_dofs_.front() < 0 || fixed_dofs_.back() >= num_dofs_)) {
        throw std::invalid_argument("DofReduction fixed DOFs must lie in [0, num_dofs)");
    }

    free_index_.assign(static_cast<size_t>(num_dofs_), -1);
    free_dofs_.reserve(static_cast<size_t>(num_dofs_) - fixed_dofs_.size());
    auto next_fixed = fixed_dofs_.begin();
    for (int dof = 0; dof < num_dofs_; ++dof) {
        if (next_fixed != fixed_dofs_.end() && *next_fixed == dof) {
            ++next_fixed;
            continue;
        }
        free_index_[static_cast<size_t>(dof)] = static_cast<int>(free_dofs_.size());
        free_dofs_.push_back(dof);
    }
}

int DofReduction::free_index(int dof) const {
    if (dof < 0 || dof >= num_dofs_) {
        throw std::out_of_range("DofReduction::free_index: DOF " + std::to_string(dof) + " is out of range");
    }
    return free_index_[static_cast<size_t>(dof)];
}

void DofReduction::reduce(const ConstVecRef& full, VecRef reduced) const {
    if (full.size() != num_dofs_ || reduced.size() != num_free()) {
        throw std::invalid_argument("DofReduction::reduce vector sizes must be num_dofs and num_free");
    }
    if (overlaps(full.data(), full.size(), reduced.data(), reduced.size())) {
        throw std::invalid_argument("DofReduction::reduce requires full and reduced not to overlap");
    }
    for (int i = 0; i < num_free(); ++i) {
        reduced[i] = full[free_dofs_[static_cast<size_t>(i)]];
    }
}

Vec DofReduction::reduce(const ConstVecRef& full) const {
    Vec reduced(num_free());
    reduce(full, reduced);
    return reduced;
}

void DofReduction::expand(const ConstVecRef& reduced, VecRef full) const {
    if (reduced.size() != num_free() || full.size() != num_dofs_) {
        throw std::invalid_argument("DofReduction::expand vector sizes must be num_free and num_dofs");
    }
    if (overlaps(reduced.data(), reduced.size(), full.data(), full.size())) {
        throw std::invalid_argument("DofReduction::expand requires reduced and full not to overlap");
    }
    for (int i = 0; i < num_free(); ++i) {
        full[free_dofs_[static_cast<size_t>(i)]] = reduced[i];
    }
}

SpMat DofReduction::reduce(const SpMat& A) const {
    return linalg::ReducedMatrix(*this, A).matrix();
}

} // namespace litepgo::dof
