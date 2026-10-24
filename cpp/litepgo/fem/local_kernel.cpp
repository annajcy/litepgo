#include "litepgo/fem/local_kernel.hpp"

#include <stdexcept>
#include <utility>

namespace litepgo {

LocalKernel::LocalKernel(VolumeQuadratureCache cache) : cache_(std::move(cache)) {}

double LocalKernel::value(const LocalValue& energy) const {
    double result = 0;
    for (int q = 0; q < cache_.num_quadrature_points(); ++q) {
        const auto& sample = cache_.sample(q);
        result += sample.volume_weight * energy.value(sample.point);
    }
    return result;
}

Eigen::VectorXd LocalKernel::gradient(const LocalGradient& energy) const {
    const int n = energy.num_dofs();
    if (n <= 0)
        throw std::invalid_argument("LocalKernel requires positive num_dofs");
    Eigen::VectorXd result = Eigen::VectorXd::Zero(n);
    for (int q = 0; q < cache_.num_quadrature_points(); ++q) {
        const auto& sample = cache_.sample(q);
        const auto gradient = energy.gradient(sample.point);
        if (gradient.size() != n)
            throw std::invalid_argument("LocalKernel: invalid gradient size");
        result += sample.volume_weight * gradient;
    }
    return result;
}

Eigen::MatrixXd LocalKernel::hessian(const LocalHessian& energy) const {
    const int n = energy.num_dofs();
    if (n <= 0)
        throw std::invalid_argument("LocalKernel requires positive num_dofs");
    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(n, n);
    for (int q = 0; q < cache_.num_quadrature_points(); ++q) {
        const auto& sample = cache_.sample(q);
        const auto hessian = energy.hessian(sample.point);
        if (hessian.rows() != n || hessian.cols() != n)
            throw std::invalid_argument("LocalKernel: invalid Hessian size");
        result += sample.volume_weight * hessian;
    }
    return result;
}

} // namespace litepgo
