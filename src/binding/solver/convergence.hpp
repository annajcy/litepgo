#pragma once
#include "litepgo/solver/newton.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>

namespace litepgo::solver::binding {
inline ConvergenceCheck convergence_check(nanobind::handle value) {
    if (nanobind::isinstance<AbsoluteConvergence>(value))
        return nanobind::cast<AbsoluteConvergence>(value);
    if (!PyCallable_Check(value.ptr()))
        throw nanobind::type_error("convergence must be callable");
    return nanobind::cast<ConvergenceCheck>(value);
}
} // namespace litepgo::solver::binding
