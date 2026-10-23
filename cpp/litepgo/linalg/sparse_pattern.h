#pragma once

#include "litepgo/core/types.h"

namespace litepgo {

// Whether A is compressed with strictly increasing row indices in every column:
// sorted, with no duplicate entries.
bool is_canonical(const SpMat& A);

// Whether a and b have the same dimensions and store the same positions; values
// are ignored. Uncompressed matrices never match.
bool same_pattern(const SpMat& a, const SpMat& b);

// A compressed copy of A's pattern with every stored value set to zero.
SpMat pattern_of(const SpMat& A);

} // namespace litepgo
