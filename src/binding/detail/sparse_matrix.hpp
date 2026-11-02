#pragma once

#include "litepgo/common/types.hpp"

#include <nanobind/eigen/sparse.h>
#include <nanobind/nanobind.h>

namespace litepgo::linalg::binding {

// A sparse-matrix argument from Python, converted to a canonical SpMat.
//
// Accepts scipy.sparse matrices and arrays in any format, and, where implicit
// conversion is allowed, dense 2-D arrays. (Only 2-D: scipy would turn a 1-D
// input into a 1 x n matrix, so an overload taking a vector must win for those.) Duplicate entries are summed, which is
// what scipy defines them to mean; the caller's object is never modified. Binding a parameter as SparseArgument instead
// of SpMat matters: nanobind's own SpMat conversion copies duplicate entries straight into Eigen, which asserts on
// them.
struct SparseArgument {
    SpMat matrix;
};

} // namespace litepgo::linalg::binding

namespace nanobind::detail {

template <> struct type_caster<litepgo::linalg::binding::SparseArgument> {
    NB_TYPE_CASTER(litepgo::linalg::binding::SparseArgument, const_name("scipy.sparse.csc_matrix"))

    bool from_python(handle src, uint32_t flags, cleanup_list*) noexcept {
        try {
            module_ sparse = module_::import_("scipy.sparse");
            if (!cast<bool>(sparse.attr("issparse")(src))) {
                const bool may_convert = flags & static_cast<uint32_t>(cast_flags::convert);
                if (!may_convert || cast<int>(module_::import_("numpy").attr("ndim")(src)) != 2) {
                    return false;
                }
            }
            object csc_matrix = sparse.attr("csc_matrix");
            object csc = src.type().is(csc_matrix) ? borrow(src) : csc_matrix(src);
            if (!cast<bool>(csc.attr("has_canonical_format"))) {
                csc = csc.attr("copy")();
                csc.attr("sum_duplicates")();
            }
            value.matrix = cast<litepgo::SpMat>(csc);
            return true;
        } catch (const std::exception&) {
            PyErr_Clear();
            return false;
        }
    }
};

} // namespace nanobind::detail
