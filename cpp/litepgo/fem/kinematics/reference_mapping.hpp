#pragma once

#include <Eigen/Core>
#include <Eigen/LU>
#include <cmath>
#include <stdexcept>

namespace litepgo {

// Pointwise coordinate conversion; basic fields do not depend on this geometry operation.
class ReferenceMapping final {
  public:
    template <class XSample> explicit ReferenceMapping(const XSample& X_sample) {
        if (!X_sample.xi.allFinite() || X_sample.value.rows() != 3 || X_sample.value.cols() != 1 ||
            X_sample.dvalue_dxi.rows() != 3 || X_sample.dvalue_dxi.cols() != 3 || !X_sample.value.allFinite() ||
            !X_sample.dvalue_dxi.allFinite())
            throw std::invalid_argument("ReferenceMapping requires finite three-component coordinates");
        xi_ = X_sample.xi;
        X_ = X_sample.value;
        dX_dxi_ = X_sample.dvalue_dxi;
        det_dX_dxi_ = dX_dxi_.determinant();
        if (!std::isfinite(det_dX_dxi_) || det_dX_dxi_ <= 0)
            throw std::invalid_argument("ReferenceMapping: invalid reference mapping");
        dxi_dX_ = dX_dxi_.inverse();
        if (!dxi_dX_.allFinite())
            throw std::invalid_argument("ReferenceMapping: non-finite inverse");
    }
    const Eigen::Vector3d& xi() const {
        return xi_;
    }
    const Eigen::Vector3d& X() const {
        return X_;
    }
    const Eigen::Matrix3d& dX_dxi() const {
        return dX_dxi_;
    }
    const Eigen::Matrix3d& dxi_dX() const {
        return dxi_dX_;
    }
    double det_dX_dxi() const {
        return det_dX_dxi_;
    }

  private:
    Eigen::Vector3d xi_, X_;
    Eigen::Matrix3d dX_dxi_, dxi_dX_;
    double det_dX_dxi_;
};

} // namespace litepgo
