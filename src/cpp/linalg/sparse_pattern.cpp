#include "litepgo/linalg/sparse_pattern.hpp"

#include <algorithm>

namespace litepgo::linalg {

bool is_canonical(const SpMat& A) {
    if (!A.isCompressed()) {
        return false;
    }
    const int* outer = A.outerIndexPtr();
    const int* inner = A.innerIndexPtr();
    for (Eigen::Index column = 0; column < A.cols(); ++column) {
        for (int k = outer[column] + 1; k < outer[column + 1]; ++k) {
            if (inner[k - 1] >= inner[k]) {
                return false;
            }
        }
    }
    return true;
}

bool same_pattern(const SpMat& a, const SpMat& b) {
    if (!a.isCompressed() || !b.isCompressed() || a.rows() != b.rows() || a.cols() != b.cols() ||
        a.nonZeros() != b.nonZeros()) {
        return false;
    }
    return std::equal(a.outerIndexPtr(), a.outerIndexPtr() + a.outerSize() + 1, b.outerIndexPtr()) &&
           std::equal(a.innerIndexPtr(), a.innerIndexPtr() + a.nonZeros(), b.innerIndexPtr());
}

SpMat pattern_of(const SpMat& A) {
    SpMat pattern = A;
    pattern.makeCompressed();
    std::fill(pattern.valuePtr(), pattern.valuePtr() + pattern.nonZeros(), 0.0);
    return pattern;
}

} // namespace litepgo::linalg
