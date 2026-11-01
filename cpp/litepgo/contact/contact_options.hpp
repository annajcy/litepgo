#pragma once
#include "litepgo/common/types.hpp"
#include <limits>
namespace litepgo::contact {
// A static infinite plane through point with the given normal (nonzero, normalized
// on construction). Vertices must stay on the side the normal points to.
struct Plane {
    Eigen::Vector3d point = Eigen::Vector3d::Zero();
    Eigen::Vector3d normal = Eigen::Vector3d::UnitZ();
};

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
