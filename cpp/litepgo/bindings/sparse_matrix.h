#pragma once

#include "litepgo/core/types.h"

#include <nanobind/eigen/sparse.h>
#include <nanobind/nanobind.h>

namespace litepgo::bindings {

// Converts a scipy.sparse matrix (or anything scipy.sparse.csc_matrix accepts)
// to SpMat. Requires canonical format, as SparseSolver does: nanobind would copy
// duplicate entries straight into Eigen, which asserts on them. Never modifies
// the caller's object. Needs the GIL.
inline SpMat sparse_matrix_from_python(nanobind::handle matrix) {
    nanobind::object csc_matrix = nanobind::module_::import_("scipy.sparse").attr("csc_matrix");
    nanobind::object csc = matrix.type().is(csc_matrix) ? nanobind::borrow(matrix) : csc_matrix(matrix);
    if (!nanobind::cast<bool>(csc.attr("has_canonical_format"))) {
        throw nanobind::value_error("sparse matrix must have sorted indices and no duplicate entries; "
                                    "call sum_duplicates() first");
    }
    return nanobind::cast<SpMat>(csc);
}

} // namespace litepgo::bindings
