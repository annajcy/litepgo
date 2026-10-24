#pragma once

#include "litepgo/fem/field_value.hpp"
#include "litepgo/fem/material_inputs.hpp"

namespace litepgo {

// Each member owns its local representation. F/Fp may be interpolated gradients
// or direct matrices; theta may be interpolated or direct.
template <class FField, class FpField, class ThetaField>
    requires std::same_as<typename FField::FieldValue, Mat3> && std::same_as<typename FpField::FieldValue, Mat3> &&
             FixedFieldValue<typename ThetaField::FieldValue> && (ThetaField::FieldValue::ColsAtCompileTime == 1)
struct VolumeFieldInputs {
    FField F;
    FpField Fp;
    ThetaField theta;

    static constexpr int num_parameters = ThetaField::FieldValue::RowsAtCompileTime;
    template <class GeometryPoint>
        requires requires(const FField& deformation, const FpField& plastic, const ThetaField& parameters,
                          const GeometryPoint& point) {
            { deformation.value(point) } -> std::same_as<Mat3>;
            { plastic.value(point) } -> std::same_as<Mat3>;
            { parameters.value(point) } -> std::same_as<typename ThetaField::FieldValue>;
        }
    MaterialInputs<num_parameters> evaluate(const GeometryPoint& point) const {
        return {F.value(point), Fp.value(point), theta.value(point)};
    }
};

} // namespace litepgo
