#include "litepgo/fem/material_fields.hpp"

#include <stdexcept>

namespace litepgo {

void MaterialFields::validate() const {
    if (!F || !Fp || !theta)
        throw std::invalid_argument("MaterialFields requires three non-null fields");
}

MaterialInputs MaterialFields::evaluate(const VolumePointData& point) const {
    validate();
    MaterialInputs inputs{F->value(point), Fp->value(point), theta->value(point)};
    if (inputs.theta.size() == 0 || !inputs.F.allFinite() || !inputs.Fp.allFinite() || !inputs.theta.allFinite())
        throw std::invalid_argument("MaterialFields requires finite values and nonempty parameters");
    return inputs;
}

} // namespace litepgo
