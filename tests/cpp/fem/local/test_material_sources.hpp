#pragma once

#include "litepgo/fem/field/identity_field.hpp"
#include "litepgo/fem/field/interpolated_field.hpp"
#include "litepgo/fem/kinematics/deformation_gradient.hpp"
#include "litepgo/fem/local/material_local_energy.hpp"

namespace litepgo::test {
template <class Source> auto value_at(const Source& source, const Eigen::Vector3d& xi) {
    return detail::material_value(detail::material_sample(source, xi));
}
template <class FSource, class FpSource, class ThetaSource> struct MaterialSources {
    FSource F;
    FpSource Fp;
    ThetaSource theta;
    MaterialInputs evaluate(const Eigen::Vector3d& xi) const {
        return {value_at(F, xi), value_at(Fp, xi), value_at(theta, xi)};
    }
};
template <class Material, MaterialVariable Active, class Sources> auto bind_energy(const Sources& fields) {
    return make_material_local_energy<Material, Active>(fields.F, fields.Fp, fields.theta);
}
template <class Material, MaterialVariable Active, class Sources, class XField>
auto bind_energy(const Sources& fields, const XField& X_field) {
    return make_material_local_energy<Material, Active>(fields.F, fields.Fp, fields.theta, X_field);
}
} // namespace litepgo::test
