#pragma once

#include "litepgo/common/types.hpp"

#include <algorithm>
#include <vector>

namespace litepgo::linalg::detail {

// PARDISO's symmetric matrix types require every diagonal entry to be stored, even
// when it is zero; with one missing, the analysis reads out of bounds. Returns A if
// it stores its whole diagonal, else a copy in storage with explicit zeros added
// there. The added entries depend only on the pattern, so analysis and
// factorization see the same one. A must be square with sorted row indices.
inline const SpMat& with_stored_diagonal(const SpMat& A, SpMat& storage) {
    std::vector<int> missing;
    for (int column = 0; column < A.outerSize(); ++column) {
        const int* rows = A.innerIndexPtr();
        if (!std::binary_search(rows + A.outerIndexPtr()[column], rows + A.outerIndexPtr()[column + 1], column)) {
            missing.push_back(column);
        }
    }
    if (missing.empty()) {
        return A;
    }
    std::vector<Triplet> entries;
    entries.reserve(static_cast<size_t>(A.nonZeros()) + missing.size());
    for (int column = 0; column < A.outerSize(); ++column) {
        for (SpMat::InnerIterator it(A, column); it; ++it) {
            entries.emplace_back(it.row(), it.col(), it.value());
        }
    }
    for (const int column : missing) {
        entries.emplace_back(column, column, 0.0);
    }
    storage.resize(A.rows(), A.cols());
    storage.setFromTriplets(entries.begin(), entries.end());
    return storage;
}

} // namespace litepgo::linalg::detail
