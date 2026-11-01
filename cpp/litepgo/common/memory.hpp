#pragma once

#include <Eigen/Core>

namespace litepgo::detail {

// Whether the arrays [a, a + a_size) and [b, b + b_size) share any element.
inline bool overlaps(const double* a, Eigen::Index a_size, const double* b, Eigen::Index b_size) {
    return a_size > 0 && b_size > 0 && a < b + b_size && b < a + a_size;
}

} // namespace litepgo::detail
