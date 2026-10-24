#pragma once

#include "litepgo/fem/field_value.hpp"
#include "litepgo/fem/volumetric_point.hpp"

namespace litepgo {

// Each member owns its local representation. F/Fp may be interpolated gradients
// or direct matrices; theta may be interpolated or direct (including empty).
template <class FField, class FpField, class ThetaField>
    requires std::same_as<typename FField::Value, Mat3> && std::same_as<typename FpField::Value, Mat3> &&
             FixedFieldValue<typename ThetaField::Value> && (ThetaField::Value::ColsAtCompileTime == 1)
struct VolumeFieldInputs {
    FField F;
    FpField Fp;
    ThetaField theta;

    static constexpr int num_parameters = ThetaField::Value::RowsAtCompileTime;
    using Point = VolumetricPoint<num_parameters>;
};

namespace detail {
template <class Fields, class ReferencePoint>
typename Fields::Point volume_field_point(const Fields& fields, const ReferencePoint& point) {
    return {fields.F.value(point), fields.Fp.value(point), fields.theta.value(point)};
}
} // namespace detail

} // namespace litepgo
