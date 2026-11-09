#pragma once
#include "litepgo/common/types.hpp"
#include <limits>
namespace litepgo::contact {
// Native ACCD stops when another safe increment is below ccd_tolerance (time),
// or ccd_max_iterations is exhausted; both return the last conservative bound.
struct IPCContactOptions {
    // Physical activation width above min_distance, matching IPC Toolkit's dhat.
    double dhat = std::numeric_limits<double>::quiet_NaN();
    double min_distance = 0.0;
    double ccd_tolerance = 1e-6;
    long ccd_max_iterations = 100'000;
    // Fixed reference areas; set false for the original unweighted candidate sum.
    bool use_area_weighting = true;
    // IPC Toolkit's physical normalization makes the external stiffness a pressure
    // when reference-area weighting is enabled. False selects its default raw barrier.
    bool use_physical_barrier = false;
};

// Tight-Inclusion time-of-impact tolerance and conservative iteration budget.
struct IPCTKContactOptions {
    double dhat = std::numeric_limits<double>::quiet_NaN();
    double min_distance = 0.0;
    double ccd_tolerance = 1e-6;
    long ccd_max_iterations = 100'000;
    bool use_area_weighting = true;
    bool use_physical_barrier = false;
};
} // namespace litepgo::contact
