#pragma once

#include "litepgo/fem/detail/dof_derivatives.hpp"

#include <type_traits>
#include <utility>

// Test energy with a nonzero theta Hessian (NeoHookean is linear in theta).
template <class Fields> class QuadraticParameterLocalEnergy {
  public:
    static constexpr int num_dofs = std::remove_cvref_t<decltype(std::declval<Fields>().theta)>::num_dofs;
    explicit QuadraticParameterLocalEnergy(Fields fields) : fields_(std::move(fields)) {}
    template <class Point> double value(const Point& point) const {
        return 0.5 * fields_.theta.value(point).squaredNorm();
    }
    template <class Point> Eigen::Matrix<double, num_dofs, 1> gradient(const Point& point) const {
        return litepgo::detail::gradient_wrt_dofs(fields_.theta.jacobian(point), fields_.theta.value(point));
    }
    template <class Point> Eigen::Matrix<double, num_dofs, num_dofs> hessian(const Point& point) const {
        return litepgo::detail::hessian_wrt_dofs(fields_.theta.jacobian(point), Eigen::Matrix2d::Identity());
    }

  private:
    Fields fields_;
};
