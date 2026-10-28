#pragma once

#include <Eigen/Core>
#include <stdexcept>

namespace litepgo {

// Vector-valued field, affine in its active coefficients. Gradient outputs flatten by column.
template <int ParamDim> class FieldT {
  public:
    static_assert(ParamDim == 2 || ParamDim == 3);
    static constexpr int num_second_derivatives = ParamDim * (ParamDim + 1) / 2;
    using Coordinate = Eigen::Matrix<double, ParamDim, 1>;
    using SpatialDerivative = Eigen::Matrix<double, Eigen::Dynamic, ParamDim>;
    // Upper triangle in row order: (11,12,22) in 2D, (11,12,13,22,23,33) in 3D.
    using SpatialHessian = Eigen::Matrix<double, Eigen::Dynamic, num_second_derivatives>;

    virtual ~FieldT() = default;
    virtual int num_components() const = 0;
    virtual int num_dofs() const = 0;
    virtual Eigen::VectorXd eval_value(const Coordinate& xi) const = 0;
    virtual Eigen::MatrixXd eval_dvalue_dcoeffs(const Coordinate& xi) const = 0;
    virtual SpatialDerivative eval_dvalue_dxi(const Coordinate&) const {
        throw std::logic_error("FieldT does not support reference-coordinate derivatives");
    }
    virtual Eigen::MatrixXd eval_dgrad_xi_dcoeffs(const Coordinate&) const {
        throw std::logic_error("FieldT does not support mixed reference-coordinate and coefficient derivatives");
    }
    virtual SpatialHessian eval_d2value_dxi2(const Coordinate&) const {
        throw std::logic_error("FieldT does not support second reference-coordinate derivatives");
    }
    virtual Eigen::MatrixXd eval_dhess_xi_dcoeffs(const Coordinate&) const {
        throw std::logic_error("FieldT does not support second reference-coordinate and coefficient derivatives");
    }
};

using Field2D = FieldT<2>;
using Field3D = FieldT<3>;

} // namespace litepgo
