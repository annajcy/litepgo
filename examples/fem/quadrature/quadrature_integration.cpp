// Integrate polynomials and constants on reference elements.
// Build: cmake --build build/release --target quadrature_integration
// Run: build/release/examples/fem/quadrature/quadrature_integration
#include "litepgo/fem/quadrature/1d/gauss_legendre_1d.hpp"
#include "litepgo/fem/quadrature/2d/gauss_legendre_2d.hpp"
#include "litepgo/fem/quadrature/2d/triangle_centroid_quadrature.hpp"
#include "litepgo/fem/quadrature/3d/gauss_legendre_3d.hpp"
#include "litepgo/fem/quadrature/3d/tet_centroid_quadrature.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace fem = litepgo::fem;

template <class Q, class Integrand> void run(const char* name, Integrand f, double expected) {
    const auto& points = Q::points();
    const auto& weights = Q::weights();
    double integral = 0;
    for (int q = 0; q < Q::num_points; ++q)
        integral += weights[q] * f(points.row(q));
    if (!std::isfinite(integral) || std::abs(integral - expected) > 1e-12)
        throw std::runtime_error("Quadrature does not reproduce the analytic integral");
    std::cout << name << ": points=" << Q::num_points << ", integral=" << std::fixed << std::setprecision(6) << integral
              << ", verified=1\n";
}

int main() {
    const auto squared_norm = [](const auto& xi) { return xi.squaredNorm(); };
    run<fem::GaussLegendre1D<2>>("GaussLegendre1D[2]", squared_norm, 1.0 / 3);
    run<fem::GaussLegendre2D<2>>("GaussLegendre2D[2]", squared_norm, 2.0 / 3);
    run<fem::GaussLegendre3D<2>>("GaussLegendre3D[2]", squared_norm, 1);

    const auto constant = [](const auto&) { return 1.0; };
    run<fem::TriangleCentroidQuadrature>("TriangleCentroid", constant, 0.5);
    run<fem::TetCentroidQuadrature>("TetCentroid", constant, 1.0 / 6);
}
