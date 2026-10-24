#include "litepgo/dof/assembler.hpp"
#include "litepgo/linalg/sparse_pattern.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace litepgo {

Assembler::Assembler(int num_dofs, std::vector<std::vector<int>> element_dof_indices)
    : num_dofs_(num_dofs), element_dof_indices_(std::move(element_dof_indices)) {
    if (num_dofs < 0 || element_dof_indices_.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("Assembler: invalid DOF or element count");

    std::size_t count = 0;
    for (const auto& indices : element_dof_indices_) {
        const auto n = indices.size();
        if (n != 0 && n > (static_cast<std::size_t>(std::numeric_limits<int>::max()) - count) / n)
            throw std::invalid_argument("Assembler: local Hessian entry count must fit in int");
        count += n * n;
        for (int index : indices)
            if (index < 0 || index >= num_dofs)
                throw std::invalid_argument("Assembler: DOF index out of range");
    }

    std::vector<Triplet> entries;
    entries.reserve(count);
    for (const auto& indices : element_dof_indices_)
        for (int column : indices)
            for (int row : indices)
                entries.emplace_back(row, column, 1.0);
    pattern_.resize(num_dofs, num_dofs);
    pattern_.setFromTriplets(entries.begin(), entries.end());
    pattern_ = pattern_of(pattern_);

    hessian_offsets_.reserve(element_dof_indices_.size());
    for (const auto& indices : element_dof_indices_) {
        auto& offsets = hessian_offsets_.emplace_back();
        offsets.reserve(indices.size() * indices.size());
        for (int column : indices) {
            const auto* begin = pattern_.innerIndexPtr() + pattern_.outerIndexPtr()[column];
            const auto* end = pattern_.innerIndexPtr() + pattern_.outerIndexPtr()[column + 1];
            for (int row : indices)
                offsets.push_back(static_cast<int>(std::lower_bound(begin, end, row) - pattern_.innerIndexPtr()));
        }
    }
}

int Assembler::num_elements() const noexcept {
    return static_cast<int>(element_dof_indices_.size());
}
std::span<const int> Assembler::dof_indices(int element) const {
    check_element(element);
    return element_dof_indices_[static_cast<std::size_t>(element)];
}

const SpMat& Assembler::hessian_pattern() const noexcept {
    return pattern_;
}

void Assembler::check_element(int element) const {
    if (element < 0 || static_cast<std::size_t>(element) >= element_dof_indices_.size())
        throw std::out_of_range("Assembler: element index out of range");
}

void Assembler::add_gradient(int element, const ConstVecRef& local, VecRef global) const {
    check_element(element);
    const auto& indices = element_dof_indices_[static_cast<std::size_t>(element)];
    if (local.size() != static_cast<Eigen::Index>(indices.size()) || global.size() != num_dofs_ || !local.allFinite())
        throw std::invalid_argument("Assembler: invalid gradient dimensions or values");
    for (std::size_t i = 0; i < indices.size(); ++i)
        global[indices[i]] += local[static_cast<Eigen::Index>(i)];
}

void Assembler::add_hessian(int element, const Eigen::Ref<const Eigen::MatrixXd>& local, SpMat& global) const {
    check_element(element);
    const auto n = static_cast<Eigen::Index>(element_dof_indices_[static_cast<std::size_t>(element)].size());
    if (local.rows() != n || local.cols() != n || !local.allFinite() || !same_pattern(global, pattern_))
        throw std::invalid_argument("Assembler: invalid Hessian dimensions, values, or pattern");
    const auto& offsets = hessian_offsets_[static_cast<std::size_t>(element)];
    for (Eigen::Index j = 0; j < n; ++j)
        for (Eigen::Index i = 0; i < n; ++i)
            global.valuePtr()[offsets[static_cast<std::size_t>(i + n * j)]] += local(i, j);
}

} // namespace litepgo
