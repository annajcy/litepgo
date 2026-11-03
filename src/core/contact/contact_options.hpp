#pragma once
#include "litepgo/common/types.hpp"
#include <limits>
namespace litepgo::contact {
// Native ACCD stops when another safe increment is below ccd_tolerance (time),
// or ccd_max_iterations is exhausted; both return the last conservative bound.
struct IPCContactOptions {
    double dhat = std::numeric_limits<double>::quiet_NaN();
    double min_distance = 0.0;
    double ccd_tolerance = 1e-6;
    long ccd_max_iterations = 100'000;
};

// Tight-Inclusion time-of-impact tolerance and conservative iteration budget.
struct IPCTKContactOptions {
    double dhat = std::numeric_limits<double>::quiet_NaN();
    double min_distance = 0.0;
    double ccd_tolerance = 1e-6;
    long ccd_max_iterations = 100'000;
};
} // namespace litepgo::contact
