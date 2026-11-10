#include "litepgo/dof/reduced_matrix.hpp"

#include <algorithm>
#include <stdexcept>

namespace litepgo::dof {

namespace {

void check_matrix(const DofReduction& reduction, const SpMat& A) {
    if (A.rows() != reduction.num_dofs() || A.cols() != reduction.num_dofs()) {
        throw std::invalid_argument("ReducedMatrix requires a num_dofs x num_dofs matrix");
    }
    if (!A.isCompressed()) {
        throw std::invalid_argument("ReducedMatrix requires a compressed matrix; call makeCompressed()");
    }
}

} // namespace

ReducedMatrix::ReducedMatrix(const DofReduction& reduction, const SpMat& A) {
    check_matrix(reduction, A);
    outer_indices_.assign(A.outerIndexPtr(), A.outerIndexPtr() + A.outerSize() + 1);
    inner_indices_.assign(A.innerIndexPtr(), A.innerIndexPtr() + A.nonZeros());

    // Walk the free columns in order and keep their free rows. free_index is
    // increasing, so rows stay sorted within each column; the diagonal entry is
    // inserted at its sorted place when A does not store it.
    const int num_free = reduction.num_free();
    std::vector<int> outer(static_cast<size_t>(num_free) + 1, 0);
    std::vector<int> inner;
    diagonal_positions_.resize(static_cast<size_t>(num_free));
    for (int column = 0; column < num_free; ++column) {
        const int full_column = reduction.free_dofs()[static_cast<size_t>(column)];
        bool has_diagonal = false;
        const auto add_entry = [&](int row, int source) {
            if (row == column) {
                diagonal_positions_[static_cast<size_t>(column)] = static_cast<int>(inner.size());
                has_diagonal = true;
            }
            inner.push_back(row);
            source_positions_.push_back(source);
        };
        for (int k = outer_indices_[static_cast<size_t>(full_column)];
             k < outer_indices_[static_cast<size_t>(full_column) + 1]; ++k) {
            const int row = reduction.free_index(inner_indices_[static_cast<size_t>(k)]);
            if (row < 0) {
                continue;
            }
            if (!has_diagonal && row > column) {
                add_entry(column, -1);
            }
            add_entry(row, k);
        }
        if (!has_diagonal) {
            add_entry(column, -1);
        }
        outer[static_cast<size_t>(column) + 1] = static_cast<int>(inner.size());
    }

    const std::vector<double> values(inner.size(), 0.0);
    matrix_ = Eigen::Map<const SpMat>(num_free, num_free, static_cast<Eigen::Index>(inner.size()), outer.data(),
                                      inner.data(), values.data());
    matrix_.makeCompressed();
    assign(A);
}

bool ReducedMatrix::has_pattern(const SpMat& A) const {
    return A.isCompressed() && A.rows() == A.cols() &&
           A.outerSize() + 1 == static_cast<Eigen::Index>(outer_indices_.size()) &&
           A.nonZeros() == static_cast<Eigen::Index>(inner_indices_.size()) &&
           std::equal(outer_indices_.begin(), outer_indices_.end(), A.outerIndexPtr()) &&
           std::equal(inner_indices_.begin(), inner_indices_.end(), A.innerIndexPtr());
}

void ReducedMatrix::assign(const SpMat& A) {
    if (!has_pattern(A)) {
        throw std::invalid_argument("ReducedMatrix::assign requires the sparsity pattern given at construction");
    }
    const double* source = A.valuePtr();
    double* target = matrix_.valuePtr();
    for (size_t k = 0; k < source_positions_.size(); ++k) {
        target[k] = source_positions_[k] >= 0 ? source[source_positions_[k]] : 0.0;
    }
}

void ReducedMatrix::add_to_diagonal(double shift) {
    double* values = matrix_.valuePtr();
    for (const int position : diagonal_positions_) {
        values[position] += shift;
    }
}

} // namespace litepgo::dof
