#pragma once

#include "litepgo/fem/geometry/volume_point_data.hpp"

namespace litepgo {

// Rows are basis coefficients, columns are field components; DOFs flatten by row.
class FieldInterpolation final {
  public:
    explicit FieldInterpolation(Eigen::MatrixXd coefficients);

    int num_dofs() const;
    int num_components() const;

    Eigen::MatrixXd value_matrix(const VolumePointData& point) const;
    Eigen::MatrixXd gradient_matrix(const VolumePointData& point) const;
    Eigen::VectorXd value(const VolumePointData& point) const;
    Eigen::Matrix<double, Eigen::Dynamic, 3> gradient(const VolumePointData& point) const;

  private:
    void check_value_point(const VolumePointData& point) const;
    void check_gradient_point(const VolumePointData& point) const;

    Eigen::MatrixXd coefficients_;
};

} // namespace litepgo
