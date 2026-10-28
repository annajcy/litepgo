#pragma once

#include <Eigen/Core>

namespace litepgo {

// Pointwise density W and derivatives with respect to one declared, affine coefficient block.
template <int ParamDim> class LocalEnergyT {
  public:
    static_assert(ParamDim == 2 || ParamDim == 3);
    using Coordinate = Eigen::Matrix<double, ParamDim, 1>;

    virtual ~LocalEnergyT() = default;
    virtual int num_dofs() const = 0;
    virtual double value(const Coordinate& xi) const = 0;
    virtual bool has_gradient() const noexcept;
    virtual bool has_hessian() const noexcept;
    virtual Eigen::VectorXd gradient(const Coordinate& xi) const;
    virtual Eigen::MatrixXd hessian(const Coordinate& xi) const;

    // The caller supplies the complete weight; no geometry or measure is applied here.
    // Adds requested derivatives to initialized, non-aliasing buffers without resizing them.
    virtual double accumulate(const Coordinate& xi, double weight, Eigen::VectorXd* gradient,
                              Eigen::MatrixXd* hessian) const;
};

extern template class LocalEnergyT<2>;
extern template class LocalEnergyT<3>;
using LocalEnergy2D = LocalEnergyT<2>;
using LocalEnergy3D = LocalEnergyT<3>;

} // namespace litepgo
