#pragma once

#include "litepgo/fem/quadrature.hpp"
#include "litepgo/fem/volume_reference_geometry.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace litepgo {

// The basis and quadrature must use the same parameter domain.
template <Basis3D BasisType, Quadrature3D QuadratureType> class VolumeQuadratureCache final {
  public:
    using ReferenceGeometry = VolumeReferenceGeometry<BasisType>;
    using PointData = typename ReferenceGeometry::PointData;

    static constexpr int num_quadrature_points = QuadratureType::num_points;

    struct Sample {
        PointData point;
        // w_q * det(J0(xi_q))
        double volume_weight;
    };

    // Owns the evaluated samples; retains no reference to reference_geometry.
    // Propagates point evaluation errors. A non-finite volume weight throws
    // std::invalid_argument. Validity is checked at quadrature points only.
    explicit VolumeQuadratureCache(const ReferenceGeometry& reference_geometry) {
        const auto& points = QuadratureType::points();
        const auto& weights = QuadratureType::weights();
        for (int q = 0; q < num_quadrature_points; ++q) {
            const Eigen::Vector3d xi = points.row(q).transpose();
            auto& result = samples_[static_cast<std::size_t>(q)];
            result.point = reference_geometry.evaluate(xi);
            result.volume_weight = weights[q] * result.point.jacobian_determinant;
            if (!std::isfinite(result.volume_weight)) {
                throw std::invalid_argument("VolumeQuadratureCache: non-finite volume weight at quadrature point " +
                                            std::to_string(q));
            }
        }
    }

    // Throws std::out_of_range for an invalid quadrature point index.
    const Sample& sample(int quadrature_point) const {
        if (quadrature_point < 0 || quadrature_point >= num_quadrature_points) {
            throw std::out_of_range("VolumeQuadratureCache::sample: quadrature point is out of range");
        }
        return samples_[static_cast<std::size_t>(quadrature_point)];
    }

  private:
    std::array<Sample, num_quadrature_points> samples_;
};

} // namespace litepgo
