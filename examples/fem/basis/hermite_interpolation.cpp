// Build Hermite coefficients from sample descriptions for u(xi) = sum(xi^2).
// Build: cmake --build build/release --target hermite_interpolation
// Run: build/release/examples/fem/basis/hermite_interpolation
#include "litepgo/fem/basis/2d/quad_hermite_basis.hpp"
#include "litepgo/fem/basis/3d/hex_hermite_basis.hpp"
#include "litepgo/fem/field/interpolated_field.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace fem = litepgo::fem;

template <class B, int Dim> void run(const typename fem::CoefficientInfo<Dim>::Coordinate& xi, const char* name) {
    const auto target = [](const Eigen::Matrix<double, Dim, 1>& point) { return point.squaredNorm(); };
    Eigen::VectorXd coefficients = Eigen::VectorXd::Zero(B::num_basis_functions);
    for (int i = 0; i < B::num_basis_functions; ++i) {
        const auto info = B::coefficient_info(i);
        Eigen::Matrix<double, Dim, 1> point;
        if constexpr (Dim == 1)
            point[0] = info.point;
        else
            point = info.point;
        const int degree = std::accumulate(info.derivative_order.begin(), info.derivative_order.end(), 0);
        if (degree == 0)
            coefficients[i] = target(point);
        else if (degree == 1)
            for (int d = 0; d < Dim; ++d)
                if (info.derivative_order[d] == 1)
                    coefficients[i] = 2 * point[d];
    }

    Eigen::Matrix<double, Dim, 1> point;
    if constexpr (Dim == 1)
        point[0] = xi;
    else
        point = xi;
    const double value = coefficients.dot(B::eval_N(xi));
    const Eigen::Matrix<double, Dim, 1> gradient = B::eval_dN_dxi(xi).transpose() * coefficients;
    const auto hessian = (B::eval_d2N_dxi2(xi).transpose() * coefficients).eval();
    if (std::abs(value - target(point)) > 1e-12 || !gradient.isApprox(2 * point, 1e-12))
        throw std::runtime_error("Hermite interpolation value or gradient check failed");
    int column = 0;
    for (int i = 0; i < Dim; ++i)
        for (int j = i; j < Dim; ++j)
            if (std::abs(hessian[column++] - (i == j ? 2 : 0)) > 1e-12)
                throw std::runtime_error("Hermite interpolation Hessian check failed");
    std::cout << name << ": value=" << std::fixed << std::setprecision(2) << value << ", verified=1\n";

    if constexpr (Dim > 1) {
        const fem::InterpolatedFieldT<B, Dim> field(coefficients);
        for (int sample = 0; sample < 3; ++sample) {
            Eigen::Matrix<double, Dim, 1> query;
            for (int d = 0; d < Dim; ++d)
                query[d] = 0.2 + 0.15 * d + 0.1 * sample;
            if (std::abs(field.eval_value(query)[0] - target(query)) > 1e-12)
                throw std::runtime_error("Hermite field does not reconstruct the target");
        }
        std::cout << name << " field: samples=3, verified=1\n";
    }
}

int main() {
    run<fem::CubicHermiteBasis, 1>(0.5, "CubicHermite");
    run<fem::QuadHermiteBasis, 2>(Eigen::Vector2d::Constant(0.5), "QuadHermite");
    run<fem::HexHermiteBasis, 3>(Eigen::Vector3d::Constant(0.5), "HexHermite");
}
