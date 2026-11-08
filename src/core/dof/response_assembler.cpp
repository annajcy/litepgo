#include "litepgo/dof/response_assembler.hpp"
#include "litepgo/linalg/sparse_pattern.hpp"
#include <algorithm>
#include <limits>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

namespace litepgo::dof {
ResponseAssembler::ResponseAssembler(int outputs, int inputs, Indices rows, Indices columns) {
    if (rows.size() != columns.size() || rows.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("ResponseAssembler: element counts differ or exceed index range");
    if (inputs < 0 || outputs < 0)
        throw std::invalid_argument("ResponseAssembler: negative DOF count");
    auto validate = [](const std::vector<int>& indices, int size) {
        for (int index : indices)
            if (index < 0 || index >= size)
                throw std::invalid_argument("ResponseAssembler: element DOF index out of range");
    };
    std::vector<Eigen::Triplet<double>> entries;
    for (std::size_t e = 0; e < rows.size(); ++e) {
        validate(rows[e], outputs);
        validate(columns[e], inputs);
        const auto limit = static_cast<std::size_t>(std::numeric_limits<int>::max());
        if (!columns[e].empty() && rows[e].size() > (limit - entries.size()) / columns[e].size())
            throw std::invalid_argument("ResponseAssembler: scatter plan exceeds sparse index range");
        for (int c : columns[e])
            for (int r : rows[e])
                entries.emplace_back(r, c, 1.);
    }
    pattern_.resize(outputs, inputs);
    pattern_.setFromTriplets(entries.begin(), entries.end());
    pattern_ = linalg::pattern_of(pattern_);
    rows_ = std::move(rows);
    columns_ = std::move(columns);
    offsets_.resize(rows_.size());
    for (std::size_t e = 0; e < rows_.size(); ++e)
        for (int c : columns_[e])
            for (int r : rows_[e]) {
                const int begin = pattern_.outerIndexPtr()[c], end = pattern_.outerIndexPtr()[c + 1];
                const auto position =
                    std::lower_bound(pattern_.innerIndexPtr() + begin, pattern_.innerIndexPtr() + end, r);
                offsets_[e].push_back(static_cast<int>(position - pattern_.innerIndexPtr()));
            }
    contributions_.resize(outputs);
    for (int e = 0; e < num_elements(); ++e)
        for (int r = 0; r < static_cast<int>(rows_[e].size()); ++r)
            contributions_[rows_[e][r]].push_back({e, r});
}
void ResponseAssembler::assemble(std::span<const ElementView> elements, Vec& value, SpMat* J) const {
    if (elements.size() != rows_.size())
        throw std::invalid_argument("ResponseAssembler: expected one result per element");
    if (J && !linalg::same_pattern(*J, pattern_))
        throw std::invalid_argument("ResponseAssembler: mismatched Jacobian pattern");
    tbb::parallel_for(std::size_t(0), elements.size(), [&](std::size_t e) {
        const auto local_value = elements[e].value();
        const auto local_jacobian = elements[e].jacobian();
        if (local_value.size() != static_cast<Eigen::Index>(rows_[e].size()) || !local_value.allFinite() ||
            (J &&
             (local_jacobian.rows() != local_value.size() ||
              local_jacobian.cols() != static_cast<Eigen::Index>(columns_[e].size()) || !local_jacobian.allFinite())))
            throw std::invalid_argument("ResponseAssembler: invalid local value or Jacobian");
    });
    value.resize(pattern_.rows());
    if (J)
        J->coeffs().setZero();
    tbb::parallel_for(tbb::blocked_range<int>(0, static_cast<int>(pattern_.rows()), 64), [&](const auto& range) {
        for (int row = range.begin(); row < range.end(); ++row) {
            double sum = 0;
            for (const auto& [e, r] : contributions_[row]) {
                const auto& local = elements[e];
                sum += local.value()[r];
                if (J) {
                    const auto jacobian = local.jacobian();
                    const auto& offsets = offsets_[e];
                    for (Eigen::Index c = 0; c < jacobian.cols(); ++c)
                        J->valuePtr()[offsets[c * jacobian.rows() + r]] += jacobian(r, c);
                }
            }
            value[row] = sum;
        }
    });
    if (!value.allFinite() || (J && !J->coeffs().allFinite()))
        throw std::domain_error("ResponseAssembler: assembly overflow");
}
} // namespace litepgo::dof
