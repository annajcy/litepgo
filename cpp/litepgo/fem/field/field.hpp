#pragma once

#include "litepgo/fem/geometry/rest.hpp"

#include <cmath>
#include <stdexcept>

namespace litepgo {

// Local immutable field snapshot, independent of global DOF numbering.
template <class Value> class Field {
  public:
    virtual ~Field() = default;
    virtual Value value(const RestPointData& point) const = 0;
};

// At fixed geometry, vec(value) = J * local_dofs + offset.
// vec is column-major. Affineness is a mathematical contract.
template <class Value> class AffineField : public Field<Value> {
  public:
    virtual int num_dofs() const = 0;
    virtual Eigen::MatrixXd jacobian(const RestPointData& point) const = 0;

    // Add weighted J^T g and J^T H J into initialized, correctly sized outputs.
    // Derivative inputs are only used for non-null outputs and must not alias them.
    // Structured fields can override this without materializing J.
    virtual void accumulate_derivatives(const RestPointData& point,
                                        const Eigen::Ref<const Eigen::MatrixXd>& value_gradient,
                                        const Eigen::Ref<const Eigen::MatrixXd>& value_hessian, double weight,
                                        Eigen::VectorXd* gradient, Eigen::MatrixXd* hessian) const {
        if (!std::isfinite(weight))
            throw std::invalid_argument("AffineField::accumulate_derivatives requires a finite weight");
        if (!gradient && !hessian)
            return;
        const auto J = jacobian(point);
        const int n = num_dofs();
        if (n <= 0 || J.rows() <= 0 || J.cols() != n || !J.allFinite() ||
            (gradient && (value_gradient.size() != J.rows() || gradient->size() != n)) ||
            (hessian && (value_hessian.rows() != J.rows() || value_hessian.cols() != J.rows() || hessian->rows() != n ||
                         hessian->cols() != n)))
            throw std::invalid_argument("AffineField::accumulate_derivatives: invalid Jacobian or dimensions");
        if (gradient)
            gradient->noalias() += weight * (J.transpose() * value_gradient.reshaped());
        if (hessian)
            hessian->noalias() += weight * (J.transpose() * value_hessian * J);
    }
};

} // namespace litepgo
