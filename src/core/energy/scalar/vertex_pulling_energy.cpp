#include "litepgo/energy/scalar/vertex_pulling_energy.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace litepgo::energy {

namespace {

// Validate indices before using them to construct the fixed sparse pattern.
SpMat diagonal_pattern(int num_dofs, const std::vector<std::array<int, 3>>& position_dofs) {
    if (num_dofs < 0)
        throw std::invalid_argument("VertexPullingEnergy requires a non-negative DOF count");
    std::vector<int> sorted;
    sorted.reserve(3 * position_dofs.size());
    for (const auto& vertex : position_dofs)
        sorted.insert(sorted.end(), vertex.begin(), vertex.end());
    std::sort(sorted.begin(), sorted.end());
    if (!sorted.empty() && (sorted.front() < 0 || sorted.back() >= num_dofs))
        throw std::invalid_argument("VertexPullingEnergy position DOFs must be in [0, num_dofs)");
    if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end())
        throw std::invalid_argument("VertexPullingEnergy position DOFs must be distinct");
    std::vector<Triplet> entries;
    entries.reserve(sorted.size());
    for (int dof : sorted)
        entries.emplace_back(dof, dof, 1.0);
    SpMat pattern(num_dofs, num_dofs);
    pattern.setFromTriplets(entries.begin(), entries.end());
    return pattern;
}
} // namespace

VertexPullingEnergy::VertexPullingEnergy(int num_dofs, std::vector<std::array<int, 3>> position_dofs, Vec targets,
                                         double stiffness)
    : FixedPatternEnergy(diagonal_pattern(num_dofs, position_dofs)), position_dofs_(std::move(position_dofs)),
      targets_(std::move(targets)), stiffness_(stiffness) {
    if (!std::isfinite(stiffness_) || stiffness_ <= 0.0)
        throw std::invalid_argument("VertexPullingEnergy requires a finite positive stiffness");
    if (targets_.size() != 3 * static_cast<Eigen::Index>(position_dofs_.size()) || !targets_.allFinite())
        throw std::invalid_argument("VertexPullingEnergy requires three finite target entries per pulled vertex");
}

double VertexPullingEnergy::eval_values(const ConstVecRef& x, Vec* gradient, SpMat* hessian) const {
    if (gradient != nullptr) {
        gradient->setZero();
    }
    double squared_distance = 0.0;
    for (std::size_t i = 0; i < position_dofs_.size(); ++i) {
        for (int axis = 0; axis < 3; ++axis) {
            const int dof = position_dofs_[i][axis];
            const double gap = x[dof] - targets_[static_cast<Eigen::Index>(3 * i + axis)];
            squared_distance += gap * gap;
            if (gradient != nullptr)
                (*gradient)[dof] = stiffness_ * gap;
        }
    }
    if (hessian != nullptr) {
        // One stored entry per pulled DOF, all equal to the stiffness.
        std::fill_n(hessian->valuePtr(), hessian->nonZeros(), stiffness_);
    }
    return 0.5 * stiffness_ * squared_distance;
}

} // namespace litepgo::energy
