#pragma once

#include "litepgo/fem/quadrature/quadrature.hpp"
#include "litepgo/fem/geometry/volume_reference_geometry.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace litepgo {

// The basis and quadrature must use the same parameter domain.
class VolumeQuadratureCache final {
  public:
    int num_quadrature_points() const {
        return static_cast<int>(samples_.size());
    }

    struct Sample {
        VolumePointData point;
        // w_q * det(J0(xi_q))
        double volume_weight;
    };

    // Owns the evaluated samples; retains no reference to reference_geometry.
    // Propagates point evaluation errors. A non-finite volume weight throws
    // std::invalid_argument. Validity is checked at quadrature points only.
    template <Basis3D BasisType, Quadrature3D QuadratureType>
    explicit VolumeQuadratureCache(const VolumeReferenceGeometry<BasisType>& reference_geometry, QuadratureType)
        : samples_(QuadratureType::num_points) {
        const auto& points = QuadratureType::points();
        const auto& weights = QuadratureType::weights();
        for (int q = 0; q < num_quadrature_points(); ++q) {
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
        if (quadrature_point < 0 || quadrature_point >= num_quadrature_points()) {
            throw std::out_of_range("VolumeQuadratureCache::sample: quadrature point is out of range");
        }
        return samples_[static_cast<std::size_t>(quadrature_point)];
    }

  private:
    std::vector<Sample> samples_;
};

} // namespace litepgo
