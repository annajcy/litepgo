#pragma once

#include "litepgo/core/types.hpp"

namespace litepgo::fem {

// Pointwise density W and derivatives with respect to one declared, affine coefficient block.
template <int ParamDim> class LocalEnergyT {
  public:
    static_assert(ParamDim == 2 || ParamDim == 3);
    using Coordinate = Eigen::Matrix<double, ParamDim, 1>;

    virtual ~LocalEnergyT() = default;
    virtual int num_dofs() const = 0;
    virtual double eval_value(const Coordinate& xi) const = 0;
    virtual bool has_gradient() const noexcept;
    virtual bool has_hessian() const noexcept;
    virtual Eigen::VectorXd eval_gradient(const Coordinate& xi) const;
    virtual Eigen::MatrixXd eval_hessian(const Coordinate& xi) const;

    // The caller supplies the complete weight; no geometry or measure is applied here.
    // Adds requested derivatives to initialized, non-aliasing buffers without resizing them.
    // Non-finite numerical results throw std::domain_error.
    double accumulate(const Coordinate& xi, double weight, Eigen::VectorXd* gradient, Eigen::MatrixXd* hessian) const;
    // Single implementation hook for owning buffers and strided views.
    virtual double accumulate_into(const Coordinate& xi, double weight, StridedVecRef* gradient,
                                   StridedMatRef* hessian) const;
};

extern template class LocalEnergyT<2>;
extern template class LocalEnergyT<3>;
using LocalEnergy2D = LocalEnergyT<2>;
using LocalEnergy3D = LocalEnergyT<3>;

} // namespace litepgo::fem
