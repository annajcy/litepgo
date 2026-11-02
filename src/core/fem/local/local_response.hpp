#pragma once
#include "litepgo/common/types.hpp"

namespace litepgo::fem {
// Pointwise vector density and its derivative wrt one declared coefficient block.
// Coefficients live in the associated fields; xi is a reference coordinate.
template <int ParamDim> class LocalResponseT {
  public:
    static_assert(ParamDim == 2 || ParamDim == 3);
    using Coordinate = Eigen::Matrix<double, ParamDim, 1>;
    virtual ~LocalResponseT() = default;
    virtual int num_inputs() const = 0;
    virtual int num_outputs() const = 0;
    virtual Eigen::VectorXd eval_value(const Coordinate& xi) const = 0;
    virtual bool has_jacobian() const noexcept;
    virtual Eigen::MatrixXd eval_jacobian(const Coordinate& xi) const;

    // Adds into initialized buffers without resizing. The caller supplies the
    // complete weight; outputs must not alias each other or the local inputs.
    void accumulate(const Coordinate& xi, double weight, Eigen::VectorXd& value,
                    Eigen::MatrixXd* jacobian = nullptr) const;
    // Single implementation hook for owning buffers and strided views.
    virtual void accumulate_into(const Coordinate& xi, double weight, StridedVecRef value,
                                 StridedMatRef* jacobian = nullptr) const;
};
extern template class LocalResponseT<2>;
extern template class LocalResponseT<3>;
using LocalResponse2D = LocalResponseT<2>;
using LocalResponse3D = LocalResponseT<3>;
} // namespace litepgo::fem
