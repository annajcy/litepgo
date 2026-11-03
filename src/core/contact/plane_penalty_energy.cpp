#include "litepgo/contact/plane_penalty_energy.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace litepgo::contact {
namespace {
SpMat checked_pattern(int num_vertices, int axis, double height, double stiffness) {
    if (num_vertices < 0 || num_vertices > std::numeric_limits<int>::max() / 3 || axis < 0 || axis > 2 ||
        !std::isfinite(height) || !std::isfinite(stiffness) || stiffness <= 0) {
        throw std::invalid_argument("PlanePenaltyEnergy requires a valid vertex count, axis in [0, 2], finite height "
                                    "and positive finite stiffness");
    }
    SpMat pattern(3 * num_vertices, 3 * num_vertices);
    pattern.reserve(num_vertices);
    for (int vertex = 0; vertex < num_vertices; ++vertex) {
        const int dof = 3 * vertex + axis;
        pattern.insert(dof, dof) = 1;
    }
    pattern.makeCompressed();
    return pattern;
}
} // namespace

PlanePenaltyEnergy::PlanePenaltyEnergy(int num_vertices, int axis, double height, double stiffness, bool positive_side)
    : PlanePenaltyEnergy(checked_pattern(num_vertices, axis, height, stiffness), axis, height, stiffness,
                         positive_side) {}

PlanePenaltyEnergy::PlanePenaltyEnergy(const SpMat& pattern, int axis, double height, double stiffness,
                                       bool positive_side)
    : FixedPatternEnergy(static_cast<int>(pattern.rows()), pattern), axis_(axis), height_(height),
      stiffness_(stiffness), side_(positive_side ? 1.0 : -1.0) {}

double PlanePenaltyEnergy::eval_values(const ConstVecRef& x, Vec* gradient, SpMat* hessian, energy::HessianKind) const {
    if (gradient)
        gradient->setZero();
    double value = 0;
    for (int vertex = 0; vertex < num_dofs() / 3; ++vertex) {
        const int dof = 3 * vertex + axis_;
        const double gap = x[dof] - height_;
        const bool active = side_ * gap < 0;
        if (active) {
            const double derivative = stiffness_ * gap;
            value += 0.5 * gap * derivative;
            if (gradient)
                (*gradient)[dof] = derivative;
        }
        if (hessian)
            hessian->valuePtr()[vertex] = active ? stiffness_ : 0.0;
    }
    return value;
}
} // namespace litepgo::contact
