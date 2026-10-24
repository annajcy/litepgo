#pragma once

#include "litepgo/fem/volume_field_inputs.hpp"

#include <type_traits>
#include <utility>

namespace litepgo {

// ActiveMember selects a local field representation, not a global storage kind.
// Its static Jacobian must be affine and depend only on the reference point.
// The caller must pair the selected input with the matching point energy.
template <class ReferencePointType, class FieldsType, auto ActiveMember>
    requires std::is_member_object_pointer_v<decltype(ActiveMember)> &&
             requires(const FieldsType& fields, const ReferencePointType& point) {
                 fields.*ActiveMember;
                 { fields.F.value(point) } -> std::same_as<Mat3>;
                 { fields.Fp.value(point) } -> std::same_as<Mat3>;
                 { fields.theta.value(point) } -> std::same_as<decltype(FieldsType::Point::theta)>;
             }
class VolumeFieldMapping final {
  public:
    using ReferencePoint = ReferencePointType;
    using Fields = FieldsType;
    using Point = typename Fields::Point;
    using ActiveField = std::remove_cvref_t<decltype(std::declval<Fields>().*ActiveMember)>;
    static constexpr int num_dofs = ActiveField::num_dofs;
    static constexpr int point_dofs = ActiveField::Value::SizeAtCompileTime;
    using PointGradient = typename ActiveField::Value;
    using PointHessian = Eigen::Matrix<double, point_dofs, point_dofs>;
    using Jacobian = typename ActiveField::Jacobian;
    static_assert(Jacobian::RowsAtCompileTime == point_dofs && Jacobian::ColsAtCompileTime == num_dofs);

    Point point_input(const Fields& fields, const ReferencePoint& point) const {
        return detail::volume_field_point(fields, point);
    }
    Jacobian active_jacobian(const ReferencePoint& point) const {
        return ActiveField::jacobian(point);
    }
};

} // namespace litepgo
